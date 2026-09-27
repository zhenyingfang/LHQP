import os
import sys

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import copy
import importlib
import json
from pathlib import Path
import torch
import torch.distributed as dist
import tqdm
from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler
from mmengine.config import Config, DictAction
from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader
from opentad.cores import train_one_epoch, val_one_epoch, eval_one_epoch, build_optimizer, build_scheduler
from opentad.cores.test_engine import gather_ddp_results
from opentad.datasets.base import SlidingWindowDataset
from opentad.models.utils.post_processing import build_classifier
from opentad.utils import (
    set_seed,
    update_workdir,
    create_folder,
    save_config,
    setup_logger,
    ModelEma,
    save_checkpoint,
    save_best_checkpoint,
)


def infer_dynamic_training_set(loader, model, model_ema, cfg, rank, world_size, use_amp, dynamic_cfg=None):
    """Infer the training subset in-process without resetting optimizer state."""
    if model_ema is not None:
        current_dict = copy.deepcopy(model.state_dict())
        model.load_state_dict(model_ema.module.state_dict())
    post_cfg = copy.deepcopy(cfg.post_processing)
    post_cfg.sliding_window = isinstance(loader.dataset, SlidingWindowDataset)
    # ActivityNet training inference is class-agnostic: its validation-only
    # external classifier does not contain entries for training videos.
    if dynamic_cfg is not None and dynamic_cfg.get("disable_external_cls", False):
        post_cfg.external_cls = None
    if "external_cls" in post_cfg and post_cfg.external_cls is not None:
        external_cls = build_classifier(post_cfg.external_cls)
    else:
        external_cls = loader.dataset.class_map
    infer_cfg = copy.deepcopy(cfg.inference)
    infer_cfg.save_raw_prediction = False
    model.eval()
    result_dict = {}
    for data_dict in tqdm.tqdm(loader, disable=(rank != 0), desc="dynamic train inference"):
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                results = model(
                    **data_dict,
                    return_loss=False,
                    infer_cfg=infer_cfg,
                    post_cfg=post_cfg,
                    ext_cls=external_cls,
                )
        for key, value in results.items():
            result_dict.setdefault(key, []).extend(value)
    result_dict = gather_ddp_results(world_size, result_dict, post_cfg)
    if model_ema is not None:
        model.load_state_dict(current_dict)
        del current_dict
    model.train()
    return result_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument("--resume", type=str, default=None, help="resume from a checkpoint")
    parser.add_argument("--not_eval", action="store_true", help="whether not to eval, only do inference")
    parser.add_argument("--disable_deterministic", action="store_true", help="disable deterministic for faster speed")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # DDP init
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["RANK"])
    print(f"Distributed init (rank {args.rank}/{args.world_size}, local rank {args.local_rank})")
    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)

    # set random seed, create work_dir, and save config
    set_seed(args.seed, args.disable_deterministic)
    cfg = update_workdir(cfg, args.id, args.world_size)
    if args.rank == 0:
        create_folder(cfg.work_dir)
        save_config(args.config, cfg.work_dir)

    # setup logger
    logger = setup_logger("Train", save_dir=cfg.work_dir, distributed_rank=args.rank)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    logger.info(f"Config: \n{cfg.pretty_text}")

    # build dataset
    train_dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=logger))
    train_loader = build_dataloader(
        train_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=True,
        drop_last=True,
        **cfg.solver.train,
    )

    val_dataset = build_dataset(cfg.dataset.val, default_args=dict(logger=logger))
    val_loader = build_dataloader(
        val_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=False,
        drop_last=False,
        **cfg.solver.val,
    )

    test_dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=logger))
    test_loader = build_dataloader(
        test_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=False,
        drop_last=False,
        **cfg.solver.test,
    )

    dynamic_cfg = cfg.get("dynamic_pseudo", None)
    if dynamic_cfg is not None and dynamic_cfg.get("enabled", False):
        dynamic_test_cfg = copy.deepcopy(cfg.dataset.test)
        dynamic_test_cfg.ann_file = cfg.dataset.train.ann_file
        dynamic_test_cfg.subset_name = "training"
        dynamic_test_cfg.data_path = cfg.dataset.train.data_path
        dynamic_test_cfg.test_mode = True
        dynamic_dataset = build_dataset(dynamic_test_cfg, default_args=dict(logger=logger))
        dynamic_loader = build_dataloader(
            dynamic_dataset,
            rank=args.rank,
            world_size=args.world_size,
            shuffle=False,
            drop_last=False,
            **cfg.solver.test,
        )
        dynamic_epochs = set(range(
            int(dynamic_cfg.start_epoch),
            int(dynamic_cfg.stop_epoch) + 1,
            int(dynamic_cfg.interval),
        ))
        current_pseudo = str(dynamic_cfg.initial_pseudo)
        initial_train_loader_len = len(train_loader)
        if args.rank == 0:
            Path(dynamic_cfg.output_dir).mkdir(parents=True, exist_ok=True)
            logger.info(
                "Dynamic pseudo labels enabled: epochs=%s beta=%s gamma=%s",
                sorted(dynamic_epochs), dynamic_cfg.beta, dynamic_cfg.gamma,
            )
    else:
        dynamic_loader = None
        dynamic_epochs = set()
        current_pseudo = ""


    # build model
    model = build_detector(cfg.model)

    # DDP
    use_static_graph = getattr(cfg.solver, "static_graph", False)
    model = model.to(args.local_rank)
    model = DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=False if use_static_graph else True,
        static_graph=use_static_graph,  # default is False, should be true when use activation checkpointing in E2E
    )
    logger.info(f"Using DDP with total {args.world_size} GPUS...")

    # FP16 compression
    use_fp16_compress = getattr(cfg.solver, "fp16_compress", False)
    if use_fp16_compress:
        logger.info("Using FP16 compression ...")
        model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)

    # Model EMA
    use_ema = getattr(cfg.solver, "ema", False)
    if use_ema:
        logger.info("Using Model EMA...")
        model_ema = ModelEma(model)
    else:
        model_ema = None

    # AMP: automatic mixed precision
    use_amp = getattr(cfg.solver, "amp", False)
    if use_amp:
        logger.info("Using Automatic Mixed Precision...")
        scaler = GradScaler()
    else:
        scaler = None

    # build optimizer and scheduler
    optimizer = build_optimizer(cfg.optimizer, model, logger)
    scheduler, max_epoch = build_scheduler(cfg.scheduler, optimizer, len(train_loader))

    # override the max_epoch
    max_epoch = cfg.workflow.get("end_epoch", max_epoch)

    # resume: reset epoch, load checkpoint / best rmse
    if args.resume != None:
        logger.info("Resume training from: {}".format(args.resume))
        device = f"cuda:{args.local_rank}"
        checkpoint = torch.load(args.resume, map_location=device)
        resume_epoch = checkpoint["epoch"]
        logger.info("Resume epoch is {}".format(resume_epoch))
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if model_ema != None:
            model_ema.module.load_state_dict(checkpoint["state_dict_ema"])

        del checkpoint  #  save memory if the model is very large such as ViT-g
        torch.cuda.empty_cache()
    else:
        resume_epoch = -1

    if dynamic_cfg is not None and dynamic_cfg.get("enabled", False) and resume_epoch >= 0:
        completed_updates = [
            update_epoch
            for update_epoch in sorted(dynamic_epochs)
            if update_epoch <= resume_epoch
            and (Path(dynamic_cfg.output_dir) / f"epoch_{update_epoch:03d}" / "pseudo_labels.json").is_file()
        ]
        if completed_updates:
            latest_update = completed_updates[-1]
            latest_dir = Path(dynamic_cfg.output_dir) / f"epoch_{latest_update:03d}"
            current_pseudo = str(latest_dir / "pseudo_labels.json")
            resumed_train_cfg = copy.deepcopy(cfg.dataset.train)
            resumed_train_cfg.ann_file = str(latest_dir / "actionformer_annotations.json")
            train_dataset = build_dataset(resumed_train_cfg, default_args=dict(logger=logger))
            train_loader = build_dataloader(
                train_dataset,
                rank=args.rank,
                world_size=args.world_size,
                shuffle=True,
                drop_last=True,
                **cfg.solver.train,
            )
            if len(train_loader) != initial_train_loader_len:
                raise RuntimeError("Resumed dynamic annotation changed train-loader length")


    # train the detector
    logger.info("Training Starts...\n")
    val_loss_best = 1e6
    val_start_epoch = cfg.workflow.get("val_start_epoch", 0)
    for epoch in range(resume_epoch + 1, max_epoch):
        if epoch in dynamic_epochs:
            result_dict = infer_dynamic_training_set(
                dynamic_loader, model, model_ema, cfg, args.rank, args.world_size, use_amp, dynamic_cfg
            )
            version_dir = Path(dynamic_cfg.output_dir) / f"epoch_{epoch:03d}"
            prediction_path = version_dir / "result_detection.json"
            pseudo_path = version_dir / "pseudo_labels.json"
            annotation_path = version_dir / "actionformer_annotations.json"
            audit_path = version_dir / "audit.json"
            if args.rank == 0:
                version_dir.mkdir(parents=True, exist_ok=True)
                with prediction_path.open("w", encoding="utf-8") as handle:
                    json.dump(dict(results=result_dict), handle)
                updater_module = str(dynamic_cfg.get("updater_module", "dynamic_thumos_pseudo"))
                update_dynamic_labels = importlib.import_module(updater_module).update_dynamic_labels
                updater_options = dict(dynamic_cfg.get("updater_options", {}))
                min_agreement_start = float(dynamic_cfg.min_agreement)
                min_agreement_end = float(
                    dynamic_cfg.get("min_agreement_end", min_agreement_start)
                )
                first_update_epoch = min(dynamic_epochs)
                last_update_epoch = max(dynamic_epochs)
                if last_update_epoch > first_update_epoch:
                    agreement_progress = (
                        (epoch - first_update_epoch)
                        / (last_update_epoch - first_update_epoch)
                    )
                    min_agreement = (
                        min_agreement_start
                        + agreement_progress
                        * (min_agreement_end - min_agreement_start)
                    )
                else:
                    min_agreement = min_agreement_start
                audit = update_dynamic_labels(
                    pipeline_config=str(dynamic_cfg.pipeline_config),
                    teacher_prediction=str(prediction_path),
                    current_pseudo=current_pseudo,
                    output_pseudo=str(pseudo_path),
                    output_annotation=str(annotation_path),
                    audit_path=str(audit_path),
                    epoch=epoch,
                    beta=float(dynamic_cfg.beta),
                    gamma=float(dynamic_cfg.gamma),
                    top_k=int(dynamic_cfg.top_k),
                    min_agreement=min_agreement,
                    update_fraction=float(dynamic_cfg.update_fraction),
                    update_momentum=float(dynamic_cfg.update_momentum),
                    **updater_options,
                )
            dist.barrier()
            current_pseudo = str(pseudo_path)
            refreshed_train_cfg = copy.deepcopy(cfg.dataset.train)
            refreshed_train_cfg.ann_file = str(annotation_path)
            train_dataset = build_dataset(refreshed_train_cfg, default_args=dict(logger=logger))
            train_loader = build_dataloader(
                train_dataset,
                rank=args.rank,
                world_size=args.world_size,
                shuffle=True,
                drop_last=True,
                **cfg.solver.train,
            )
            if len(train_loader) != initial_train_loader_len:
                raise RuntimeError(
                    "Dynamic pseudo update changed train-loader length; the iteration-based "
                    f"scheduler would be invalid ({initial_train_loader_len} -> {len(train_loader)})"
                )
        train_loader.sampler.set_epoch(epoch)

        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            logger,
            model_ema=model_ema,
            clip_grad_l2norm=cfg.solver.clip_grad_norm,
            logging_interval=cfg.workflow.logging_interval,
            scaler=scaler,
        )

        # # save checkpoint
        # if (epoch == max_epoch - 1) or ((epoch + 1) % cfg.workflow.checkpoint_interval == 0):
        #     if args.rank == 0:
        #         save_checkpoint(model, model_ema, optimizer, scheduler, epoch, work_dir=cfg.work_dir)

        # val for one epoch
        if epoch >= val_start_epoch:
            if (cfg.workflow.val_loss_interval > 0) and ((epoch + 1) % cfg.workflow.val_loss_interval == 0):
                val_loss = val_one_epoch(
                    val_loader,
                    model,
                    logger,
                    args.rank,
                    epoch,
                    model_ema=model_ema,
                    use_amp=use_amp,
                )

                # save checkpoint
                if (epoch == max_epoch - 1) or ((epoch + 1) % cfg.workflow.checkpoint_interval == 0):
                    if args.rank == 0:
                        save_checkpoint(model, model_ema, optimizer, scheduler, epoch, work_dir=cfg.work_dir)

                # save the best checkpoint
                if val_loss < val_loss_best:
                    logger.info(f"New best epoch {epoch}")
                    val_loss_best = val_loss
                    if args.rank == 0:
                        save_best_checkpoint(model, model_ema, epoch, work_dir=cfg.work_dir)

        # eval for one epoch
        if epoch >= val_start_epoch:
            if (cfg.workflow.val_eval_interval > 0) and ((epoch + 1) % cfg.workflow.val_eval_interval == 0):
                eval_one_epoch(
                    test_loader,
                    model,
                    cfg,
                    logger,
                    args.rank,
                    model_ema=model_ema,
                    use_amp=use_amp,
                    world_size=args.world_size,
                    not_eval=args.not_eval,
                )
    logger.info("Training Over...\n")


if __name__ == "__main__":
    main()
