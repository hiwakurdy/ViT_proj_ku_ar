# ViT Kurdish vs Arabic Word-Image Classification

PyTorch project for classifying cropped word images as Kurdish or Arabic with:

- ViT-only classifier
- Projection-only classifier
- ViT + projection late-fusion classifier

## Project Layout

- `configs/`: experiment configs
- `src/`: training, evaluation, models, datasets, and utilities
- `data/`: generated split CSVs at runtime
- `outputs/`: checkpoints, logs, and metrics at runtime
- `evaluate_annotated_json.py`: evaluation script for COCO-style annotated line images

## Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Train

ViT baseline:

```powershell
python train.py --config configs/vit_baseline.yaml
```

ViT + projection fusion:

```powershell
python train.py --config configs/vit_proj_fusion.yaml
```

## Evaluate Annotated JSON

```powershell
python evaluate_annotated_json.py --config configs/vit_proj_fusion.yaml --checkpoint outputs/vit_proj_fusion/checkpoints/best.pt --json path\\to\\annotations.json --image-dir path\\to\\images --output outputs\\annotated_eval
```

## Notes

- The provided YAML configs use local Windows dataset paths and should be edited for a new machine.
- Generated CSV splits, checkpoints, and output artifacts are intentionally excluded from Git.

