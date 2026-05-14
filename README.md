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

Aim:

The vision system studies how architectural and training choices affect ImageNet-style classification with a ResNet-18 backbone. The main questions are whether custom normalization layers can match or improve a BatchNorm baseline, how preprocessing and augmentation change validation performance, and whether the trained classifier attends to meaningful object regions.

The vision module implements a ResNet-18 classification workflow with enough flexibility to compare normalization and preprocessing choices under a common training/evaluation interface.

Key details:

- ResNet-18 is implemented directly in the repository, rather than depending on a prebuilt torchvision model.
- The normalization layer is configurable, with built-in BatchNorm and custom alternatives including BatchNorm, InstanceNorm, Batch-InstanceNorm, LayerNorm, GroupNorm, and a no-normalization baseline.
- The preprocessing stack includes resize/crop policies, random crops, horizontal flips, Cutout, MixUp, AutoAugment-style transforms, and optional PCA color augmentation.
- Evaluation reports loss, accuracy, micro-F1, and macro-F1 through a shared metrics path.
- Grad-CAM support makes it possible to inspect model attention for both successful and failed predictions.

Reported results:

- The strongest preprocessing run used ResNet-style scale jitter with AutoAugment and reached **81.40% best validation accuracy**.
- Hyperparameter tuning with learning rate, label smoothing, and dropout improved the stable baseline to **82.78% best validation accuracy**.
- In the final normalization comparison, **Batch-Instance Normalization (BIN)** was the best model, reaching **83.20% best validation accuracy**, with **82.94% validation accuracy**, **82.94% micro-F1**, and **82.77% macro-F1** at the final checkpoint.
- The small-batch study showed that both BN and GN degraded at batch size 8 in this setup; BN retained **76.86%** validation accuracy, while GN dropped to **68.46%**.
- Grad-CAM visualizations indicated that the final model was generally object-centric, while failures often came from confusing visually similar classes or focusing on incomplete object evidence.

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

Aim:

The translation system studies English-to-Indic neural machine translation under multiple encoder-decoder designs. The project compares plain seq2seq models, attention-based decoders, frozen versus fine-tuned BERT encoders, beam-search decoding, scheduled teacher forcing, and cross-lingual transfer from English-Hindi to English-Marathi.

The translation module is a configurable encoder-decoder system for English-to-Indic translation experiments. It supports both classic word-level source representations and pretrained encoder variants, while keeping the target-side generation pipeline consistent.

Key details:

- Source encoders can use word embeddings with optional GloVe initialization or BERT-based representations.
- The decoder is an attention-based recurrent decoder with configurable hidden sizes, dropout, teacher forcing, and decoding settings.
- Target text is handled with a BPE tokenizer built and saved through the training pipeline.
- Experiment configs define data paths, model variants, optimizer settings, teacher-forcing schedules, checkpoint behavior, and generation settings.
- Evaluation includes generation quality metrics such as BLEU and chrF, with greedy or beam-search decoding.

Reported results:

| Experiment | Best beam | BLEU | chrF | TER |
| --- | ---: | ---: | ---: | ---: |
| English-Hindi Seq2Seq with GloVe | 1 | 9.36 | 26.13 | 99.10 |
| English-Hindi Seq2Seq + Attention with GloVe | 1 | 11.39 | 30.30 | 92.55 |
| English-Hindi Seq2Seq + Attention with frozen BERT | 1 | 10.62 | 29.87 | 93.50 |
| English-Hindi Seq2Seq + Attention with fine-tuned BERT | 20 | **24.26** | **45.08** | **75.07** |
| English-Hindi fine-tuned BERT with scheduled teacher forcing | 10 | 20.62 | 40.65 | 76.53 |
| English-Marathi scratch | 20 | 20.79 | 47.00 | **71.98** |
| English-Marathi transfer | 10 | **21.27** | **48.14** | 73.59 |

Beam search improved the fine-tuned English-Hindi BERT model substantially: BLEU increased from **20.60** with greedy decoding to **24.26** with beam size 20, while TER dropped from **80.88** to **75.07**. The Marathi transfer run gave a modest BLEU and chrF gain over training from scratch, suggesting useful but not uniform cross-lingual transfer.

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
