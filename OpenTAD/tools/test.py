"""Inference entry point adapted from the upstream OpenTAD tools/test.py."""

import argparse
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
OPENTAD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = OPENTAD_ROOT.parent
sys.path.insert(0, str(OPENTAD_ROOT))
os.environ.setdefault("PTAL_REPO_ROOT", str(REPO_ROOT))

# MMEngine inspects CUDA while importing. Hide GPUs before those imports for
# CPU inference, including CPU process groups with more ranks than GPUs.
device_parser = argparse.ArgumentParser(add_help=False)
device_parser.add_argument("--device")
device_args, _ = device_parser.parse_known_args()
if device_args.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.distributed as dist
from mmengine.config import Config, DictAction
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from opentad.cores import eval_one_epoch
from opentad.datasets import build_dataset
from opentad.datasets.builder import collate
from opentad.models import build_detector
from opentad.utils import create_folder, set_seed, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Infer and evaluate a temporal action detector")
    parser.add_argument("config", help="path to an OpenTAD config")
    parser.add_argument("--checkpoint", help="override the checkpoint in the config")
    parser.add_argument("--work-dir", help="directory for predictions, metrics and logs")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:N, or cpu (torchrun uses LOCAL_RANK)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--not-eval", "--not_eval", dest="not_eval", action="store_true",
                        help="save detections without computing mAP")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override config settings")
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path, use_ema, logger):
    # These are locally trained checkpoints, including metadata and EMA weights.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    key = "state_dict_ema" if use_ema else "state_dict"
    if key not in checkpoint:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain {key}")
    # Training saved the DDP wrapper's state; plain Python inference has no wrapper.
    state = {name.removeprefix("module."): value for name, value in checkpoint[key].items()}
    model.load_state_dict(state, strict=True)
    logger.info(f"Loaded {checkpoint_path}, epoch={checkpoint.get('epoch', 'unknown')}, weights={key}")


def main():
    args = parse_args()
    # Resolve caller-supplied paths before using OpenTAD-relative paths from configs.
    config_path = Path(args.config).resolve()
    checkpoint_override = str(Path(args.checkpoint).resolve()) if args.checkpoint else None
    work_dir_override = str(Path(args.work_dir).resolve()) if args.work_dir else None
    os.chdir(OPENTAD_ROOT)
    cfg = Config.fromfile(str(config_path))
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if work_dir_override:
        cfg.work_dir = work_dir_override
    cfg.post_processing.save_dict = True
    checkpoint_path = checkpoint_override or cfg.get("checkpoint")
    if not cfg.inference.load_from_raw_predictions:
        if not checkpoint_path or not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}; use --checkpoint")

    distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if distributed and args.device != "cpu" else args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu for CPU inference")
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")

    try:
        set_seed(args.seed)
        create_folder(cfg.work_dir)
        logger = setup_logger("Test", save_dir=cfg.work_dir, distributed_rank=rank)
        logger.info(f"Inference device={device}, world_size={world_size}")
        logger.info(f"Config:\n{cfg.pretty_text}")
        if rank == 0:
            cfg.dump(os.path.join(cfg.work_dir, "inference_config.py"))
        dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=logger))
        # Disjoint shards avoid DistributedSampler's duplicated final videos.
        loader_options = dict(cfg.solver.test)
        batch_size = loader_options.pop("batch_size")
        if batch_size % world_size:
            raise ValueError(f"Test batch size {batch_size} must be divisible by world size {world_size}")
        loader = DataLoader(dataset, batch_size=batch_size // world_size,
                            sampler=range(rank, len(dataset), world_size), collate_fn=collate,
                            pin_memory=device.type == "cuda", **loader_options)
        model = build_detector(cfg.model)
        if not cfg.inference.load_from_raw_predictions:
            load_checkpoint(model, checkpoint_path, cfg.solver.get("ema", False), logger)
        model = model.to(device)
        if distributed:
            model = DistributedDataParallel(
                model, device_ids=[device.index] if device.type == "cuda" else None,
                broadcast_buffers=False,
            )
        eval_one_epoch(loader, model, cfg, logger, rank, world_size=world_size,
                       use_amp=cfg.solver.get("amp", False) and device.type == "cuda",
                       not_eval=args.not_eval)
        logger.info(f"Detections saved to {cfg.work_dir}/result_detection.json")
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
