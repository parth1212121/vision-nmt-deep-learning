# Deep Learning Systems: Vision and Neural Machine Translation

Author: Parth Verma

This repository contains two PyTorch-based deep learning pipelines:

- **ResNet image classification** with custom normalization layers, training/evaluation utilities, and Grad-CAM visualizations.
- **Neural machine translation** with attention-based seq2seq models, GloVe/BERT encoder variants, BPE tokenization, beam search, and BLEU/chrF evaluation.

## Modules

| Module | Focus | Key files |
| --- | --- | --- |
| `vision_resnet/` | Image classification and interpretability | `train.py`, `evaluate.py`, `generate_gradcam.py`, `vision/` |
| `nmt_seq2seq/` | English-to-Indic neural machine translation | `scripts/train.py`, `scripts/evaluate.py`, `inference.py`, `nmt/` |

## Repository Layout

```text
.
├── vision_resnet/
│   ├── vision/             # datasets, transforms, model, metrics, training engine
│   ├── train.py            # train ResNet-18 variants
│   ├── evaluate.py         # evaluate saved checkpoints
│   └── generate_gradcam.py # generate Grad-CAM visualizations
├── nmt_seq2seq/
│   ├── nmt/                # data, model, decoding, metrics, training utilities
│   ├── configs/            # experiment configs for GloVe/BERT/attention variants
│   ├── scripts/            # training, evaluation, and data inspection CLIs
│   └── inference.py        # batch translation entry point
└── requirements.txt
```

## Highlights

- ResNet-18 implementation in PyTorch without relying on torchvision model definitions.
- Interchangeable normalization strategies: BatchNorm, InstanceNorm, Batch-InstanceNorm, LayerNorm, GroupNorm, and no-normalization baselines.
- Image preprocessing and augmentation support for random crops, flips, Cutout, MixUp, AutoAugment-style policies, and optional PCA color augmentation.
- Grad-CAM tooling for inspecting successful and failed predictions.
- Attention-based seq2seq translation with configurable GloVe and BERT encoder variants.
- Tokenizer/vocabulary pipelines, teacher-forcing schedules, beam-search decoding, and generation metrics.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Datasets, checkpoints, and large generated artifacts are intentionally kept outside the repository. The training and inference commands expect local paths to those files.

## Vision Pipeline

Train a ResNet variant:

```bash
cd vision_resnet
python train.py \
  --data-root /path/to/imagenet-style-data \
  --output-dir outputs/resnet18_gn \
  --norm gn \
  --epochs 100 \
  --batch-size 128 \
  --amp
```

Evaluate a checkpoint:

```bash
python evaluate.py \
  --checkpoint outputs/resnet18_gn/checkpoint_best.pt \
  --data-root /path/to/imagenet-style-data \
  --split val
```

Generate Grad-CAM outputs:

```bash
python generate_gradcam.py \
  --checkpoint outputs/resnet18_gn/checkpoint_best.pt \
  --data-root /path/to/imagenet-style-data \
  --output-dir outputs/gradcam \
  --split val
```

## Translation Pipeline

Train a translation model:

```bash
cd nmt_seq2seq
python scripts/train.py \
  --config configs/en_hi_bert_finetune_attn_sched_final40.json
```

Evaluate a checkpoint:

```bash
python scripts/evaluate.py \
  --config configs/en_hi_bert_finetune_attn_sched_final40.json \
  --checkpoint outputs/en_hi_bert_finetune_attn_sched/checkpoint_best.pt
```

Run batch inference:

```bash
python inference.py \
  --input /path/to/input.csv \
  --output predictions.csv \
  --checkpoint /path/to/checkpoints \
  --mode 4 \
  --decoding_strategy beam
```

