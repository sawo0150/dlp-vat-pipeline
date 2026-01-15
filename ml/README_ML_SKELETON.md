# dlp-vat-ml-skeleton (drop-in patch)

This folder contains a minimal ML training skeleton that matches your repo layout:
- `ml/src/dlp_ml/...` : code
- `ml/configs/...`    : Hydra configs

## 1) Copy into your repo
Copy the contents of this patch into your repo root (merge folders):
- merge `ml/src/dlp_ml/*` into your existing `ml/src/dlp_ml/`
- merge `ml/configs/*` into your existing `ml/configs/`

(Nothing touches `pipeline/`.)

## 2) Dependencies (ml/requirements.txt)
Make sure you have:
- torch, torchvision
- hydra-core, omegaconf
- pillow, numpy, tqdm
- wandb (optional; switch to logger=none if you don't want it)

## 3) Prepare a manifest
Create a JSONL file where each line is:
{
  "id": "...",
  "input_path": "...",
  "target_path": "...",
  "meta": {...}
}

If your pipeline manifest uses `mask_path`, `light_path`, etc:
- set `data.input_key=mask_path`
- set `data.target_key=light_path`

## 4) Run
From repo root:
  python -m dlp_ml.cli \
    task=inverse model=unet loss=l1 logger=wandb \
    data.manifest_path=/ABS/PATH/manifest.jsonl \
    data.input_key=mask_path data.target_key=light_path

Hydra will create a run directory and save `config_resolved.yaml` + checkpoints.

## 5) Next step
- Add `tasks/forward.py`, `tasks/optimize_mask.py`
- Add pix2pix/transformer models as Hydra `model=` groups
- Add boundary-band metrics + image logging
