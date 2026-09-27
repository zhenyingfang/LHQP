# Rethinking Pseudo-Label Learning for Point-Supervised Temporal Action Localization

## Environment

> Our code is built upon the codebase from [OpenTAD](https://github.com/sming256/OpenTAD), and we would like to express our gratitude for their outstanding work.

## Data Process

### `opentad/data`

```text
opentad/data/
├── activitynet-1.3/
│   ├── annotations/
│   └── features/
└── thumos-14/
    ├── annotations/
    └── features/
```


- The THUMOS14 features are from [OpenTAD](https://github.com/sming256/OpenTAD).
- The ActivityNet-1.3 data is from [OpenTAD](https://github.com/sming256/OpenTAD) and was filtered based on `point_labels`; entries without point labels were removed.

### `inputs`

```text
inputs/
├── activitynet/
│   ├── classifier.json
│   ├── ground_truth.json
│   ├── point_labels.csv
│   ├── stage1_predictions.json
│   └── video_metadata.json
├── beoid/
│   ├── features
│   ├── ground_truth.json
│   ├── point_labels.csv
│   ├── stage1_predictions.json
│   └── video_metadata.json
├── gtea/
│   ├── features
│   ├── ground_truth.json
│   ├── point_labels.csv
│   ├── stage1_predictions.json
│   └── video_metadata.json
└── thumos/
    ├── ground_truth.json
    ├── point_labels.csv
    └── stage1_predictions.json
```

- The GTEA and BEOID features are from [HR-Pro](https://github.com/pipixin321/HR-Pro).

- The `stage1_predictions.json` files for all datasets are obtained by reproducing the Stage-1 results of [HR-Pro](https://github.com/pipixin321/HR-Pro).

## Training

Run the following command from the project root to train on THUMOS, ActivityNet, BEOID, and GTEA sequentially:

```bash
bash scripts/run_dynamic_all.sh
```
