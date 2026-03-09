# ViT Kurdish vs Arabic Word-Image Classifier Scaffold

## Summary
- Initialize `VIT_PROJ_KU_AR` with the requested folders and all three runnable paths: `vit_baseline`, `proj_baseline`, and `vit_proj_fusion`.
- Use the external image folders directly: Kurdish from `E:\DSs\MLP\code\TRDG\data\350k\ku` and Arabic from `E:\DSs\MLP\code\TRDG\data\350k\ar`; do not copy images into the repo.
- Generate `data/train.csv`, `data/val.csv`, and `data/test.csv` once and reuse them for every experiment.
- Use group-aware splits by filename prefix `F*_S*` so the same generator/font group does not leak across train, validation, and test.
- Normalize every image to a common `64x256` canvas before modeling, because the raw folders already differ in image height by class (`ar` observed at 36 px, `ku` observed at 38 px).

## Key Changes
- Create the project tree exactly as requested under `configs/`, `data/`, `src/`, `src/models/`, `src/engine/`, `src/utils/`, and `outputs/`.
- Implement dataset indexing in `src/datasets.py` with CSV schema `image_path,label,label_name,source_group,split`.
- Use label mapping `ku=0` and `ar=1` to stay consistent with the earlier project.
- Extract `source_group` with regex `^(F\d+_S\d+)_`; if that fails, fall back to the filename stem.
- Persist a fixed-seed `70/15/15` split so all model variants train on identical samples.
- Implement `src/transforms.py` for train/eval image transforms. Default ViT path: load image, convert to grayscale, resize with aspect ratio preserved to height `64`, pad to width `256`, repeat to 3 channels, normalize, then tensorize.
- Keep `binarize` and `crop_to_text` as config flags. Default them `false` for `vit_baseline`, `true` for `proj_baseline`, and configurable for `vit_proj_fusion`.
- Reuse the existing projection definition in `src/projections.py`: split the normalized image into `K=4` width segments, compute per-part horizontal and vertical projections, max-normalize each part, and concatenate.
- Implement `src/models/vit_classifier.py` as a pretrained `timm` ViT wrapper for `img_size=(64, 256)` with a binary classification head.
- Implement `src/models/projection_mlp.py` as an MLP over the projection feature vector, with input dimension derived from `img_h`, `img_w`, and `K_parts`.
- Implement `src/models/vit_projection_fusion.py` to concatenate the ViT embedding and projection embedding, then classify with a fusion head.
- Implement `src/engine/train.py` for shared training logic across all model types, including validation each epoch, early stopping on macro F1, and best-checkpoint saving.
- Implement `src/engine/evaluate.py` for checkpoint-based evaluation on `val` or `test`.
- Implement `src/engine/metrics.py` for accuracy, precision, recall, macro F1, per-class F1, and confusion matrix.
- Implement `src/utils/seed.py`, `src/utils/logger.py`, and `src/utils/checkpoint.py` for reproducibility, logging, and checkpoint I/O.
- Implement `src/main.py` as the single CLI entrypoint with `--config`, `--mode`, `--checkpoint`, and optional `--split`.
- Use one shared YAML schema in all three config files with sections `project`, `data`, `split`, `transforms`, `projection`, `model`, `optimizer`, `scheduler`, `trainer`, and `logging`.
- Return a unified dataset item interface from `KurdishArabicDataset`: `image`, `projection_features`, `label`, `path`, and `source_group`.

## Test Plan
- Verify both source folders are discovered and counted correctly.
- Verify every row written to `train.csv`, `val.csv`, and `test.csv` points to an existing file.
- Verify no `image_path` appears in more than one split.
- Verify no `source_group` appears in more than one split.
- Verify per-split label balance is close to the requested ratio.
- Verify the ViT transform outputs shape `3x64x256`.
- Verify projection feature length matches the configured formula.
- Verify each YAML config builds its model successfully.
- Verify each model completes one dummy forward pass without tensor-shape errors.
- Verify a tiny training smoke test produces a checkpoint and metric files.
- Verify evaluation reloads a saved checkpoint and outputs metrics plus a confusion matrix.

## Assumptions
- This repo is binary only; the mixed `kuar` class from `first_pub` is out of scope.
- The images stay in their current external directories and the repo stores absolute paths in CSV files.
- `timm` is the ViT backend to target.
- A Python environment and dependency manifest will be added during implementation, because the current workspace does not have a working Python install.
- The requested folder layout is the canonical structure; any extra top-level file should be limited to the minimum needed to run the project.
