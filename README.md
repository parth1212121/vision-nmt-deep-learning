# Deep Learning Systems: Vision and Neural Machine Translation

Author: Parth Verma

This repository brings together two end-to-end deep learning systems: an image classification and interpretability stack built around ResNet-18, and a neural machine translation stack for English-to-Indic generation. The focus is not only on model definitions, but also on the surrounding machinery needed to run experiments cleanly: dataset handling, preprocessing, configurable training, checkpoint loading, evaluation, decoding, and inference.

## System Overview

| Area | What it supports | Main components |
| --- | --- | --- |
| `vision_resnet/` | Image classification experiments and model interpretability | ResNet-18, custom normalization layers, augmentation pipeline, checkpoint evaluation, Grad-CAM |
| `nmt_seq2seq/` | Sequence-to-sequence translation experiments | GloVe/BERT encoders, attention decoder, BPE target tokenizer, beam search, BLEU/chrF evaluation |

## Repository Layout

```text
.
├── vision_resnet/
│   ├── vision/             # datasets, transforms, model, metrics, training engine
│   ├── train.py            # train ResNet-18 variants
│   ├── evaluate.py         # evaluate saved checkpoints
│   └── generate_gradcam.py # generate Grad-CAM visual explanations
├── nmt_seq2seq/
│   ├── nmt/                # data, model, decoding, metrics, training utilities
│   ├── configs/            # reproducible translation experiment configs
│   ├── scripts/            # training, evaluation, and data inspection CLIs
│   └── inference.py        # batch translation entry point
└── requirements.txt
```

## Vision System

The vision module implements a ResNet-18 classification workflow with enough flexibility to compare normalization and preprocessing choices under a common training/evaluation interface.

Key details:

- ResNet-18 is implemented directly in the repository, rather than depending on a prebuilt torchvision model.
- The normalization layer is configurable, with built-in BatchNorm and custom alternatives including BatchNorm, InstanceNorm, Batch-InstanceNorm, LayerNorm, GroupNorm, and a no-normalization baseline.
- The preprocessing stack includes resize/crop policies, random crops, horizontal flips, Cutout, MixUp, AutoAugment-style transforms, and optional PCA color augmentation.
- Evaluation reports loss, accuracy, micro-F1, and macro-F1 through a shared metrics path.
- Grad-CAM support makes it possible to inspect model attention for both successful and failed predictions.

Typical workflow:

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

```bash
python evaluate.py \
  --checkpoint outputs/resnet18_gn/checkpoint_best.pt \
  --data-root /path/to/imagenet-style-data \
  --split val
```

```bash
python generate_gradcam.py \
  --checkpoint outputs/resnet18_gn/checkpoint_best.pt \
  --data-root /path/to/imagenet-style-data \
  --output-dir outputs/gradcam \
  --split val
```

## Translation System

The translation module is a configurable encoder-decoder system for English-to-Indic translation experiments. It supports both classic word-level source representations and pretrained encoder variants, while keeping the target-side generation pipeline consistent.

Key details:

- Source encoders can use word embeddings with optional GloVe initialization or BERT-based representations.
- The decoder is an attention-based recurrent decoder with configurable hidden sizes, dropout, teacher forcing, and decoding settings.
- Target text is handled with a BPE tokenizer built and saved through the training pipeline.
- Experiment configs define data paths, model variants, optimizer settings, teacher-forcing schedules, checkpoint behavior, and generation settings.
- Evaluation includes generation quality metrics such as BLEU and chrF, with greedy or beam-search decoding.

Typical workflow:

```bash
cd nmt_seq2seq
python scripts/train.py \
  --config configs/en_hi_bert_finetune_attn_sched_final40.json
```

```bash
python scripts/evaluate.py \
  --config configs/en_hi_bert_finetune_attn_sched_final40.json \
  --checkpoint outputs/en_hi_bert_finetune_attn_sched/checkpoint_best.pt
```

```bash
python inference.py \
  --input /path/to/input.csv \
  --output predictions.csv \
  --checkpoint /path/to/checkpoints \
  --mode 4 \
  --decoding_strategy beam
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Datasets, checkpoints, tokenizer artifacts, and generated outputs are kept outside version control. The code expects those paths to be supplied through the command-line interfaces or JSON configs.

