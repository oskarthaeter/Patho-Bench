"""
Extract patch-level features from pre-extracted CPTAC CLAM h5 files using any
Trident patch encoder (phikon, phikon_v2, uni_v1, uni_v2, virchow2, …).

Each CLAM h5 file contains:
  coords: [N, 2]           patch coordinates (preserved in output)
  imgs:   [N, 256, 256, 3] uint8 pre-extracted patches

Output per slide: <output_dir>/<slide_id>.h5  with keys:
  features: [N, D]   float32 patch embeddings
  coords:   [N, 2]   int64

Usage:
    # CCRCC with phikon (default)
    python extract_phikon_cptac_ccrcc.py

    # All cohorts, different model
    python extract_phikon_cptac_ccrcc.py \\
        --cohorts CCRCC LUAD LSCC HNSCC GBM PDA \\
        --model uni_v2 --device cuda:0 --batch_size 512 --num_workers 8
"""

import argparse
import glob
import os

import h5py
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from tqdm import tqdm

NFS_ROOT    = "/mnt/nfs03-R6/CPTAC"
OUTPUT_ROOT = "/mnt/data/ge96giq/patho-bench/features"

# Maps NFS directory name → Patho-Bench dataset key
COHORT_MAP = {
    "CCRCC": "cptac_ccrcc",
    "LUAD":  "cptac_luad",
    "LSCC":  "cptac_lscc",
    "HNSCC": "cptac_hnsc",
    "GBM":   "cptac_gbm",
    "PDA":   "cptac_pda",
}


class PatchDataset(torch.utils.data.Dataset):
    """Wraps a numpy array of uint8 patches and applies a torchvision transform."""

    def __init__(self, imgs: np.ndarray, transform):
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.transform(Image.fromarray(self.imgs[idx]))


def extract_slide(h5_path: str, out_path: str, encoder, transform, device: str,
                  batch_size: int, num_workers: int):
    with h5py.File(h5_path, "r") as f:
        imgs   = f["imgs"][:]
        coords = f["coords"][:]

    dataset = PatchDataset(imgs, transform)
    loader  = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    all_features = []
    with torch.inference_mode():
        for batch in loader:
            feats = encoder(batch.to(device, non_blocking=True))
            all_features.append(feats.cpu())

    features = torch.cat(all_features).numpy()

    with h5py.File(out_path, "w") as f:
        f.create_dataset("features", data=features)
        f.create_dataset("coords",   data=coords)


def extract_cohort(cohort_dir: str, output_dir: str, encoder, transform,
                   device: str, batch_size: int, num_workers: int):
    patches_dir = os.path.join(cohort_dir, "clam_20", "patches")
    h5_files    = sorted(glob.glob(os.path.join(patches_dir, "*.h5")))
    if not h5_files:
        print(f"  No h5 files found in {patches_dir}, skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)
    pending = [p for p in h5_files
               if not os.path.exists(os.path.join(output_dir, os.path.basename(p)))]
    print(f"  {len(h5_files)} slides total, {len(h5_files) - len(pending)} already done, "
          f"{len(pending)} to process → {output_dir}")

    for h5_path in tqdm(pending, desc=os.path.basename(cohort_dir), unit="slide"):
        slide_id = os.path.splitext(os.path.basename(h5_path))[0]
        out_path = os.path.join(output_dir, f"{slide_id}.h5")
        extract_slide(h5_path, out_path, encoder, transform, device, batch_size, num_workers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts",     nargs="+", default=["CCRCC"],
                        choices=list(COHORT_MAP.keys()),
                        help="NFS cohort directories to process")
    parser.add_argument("--model",       default="phikon",
                        help="Trident encoder name (phikon, phikon_v2, uni_v1, uni_v2, virchow2, …)")
    parser.add_argument("--device",      default="cuda:0")
    parser.add_argument("--batch_size",  type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader workers for parallel patch preprocessing")
    parser.add_argument("--nfs_root",    default=NFS_ROOT)
    parser.add_argument("--output_root", default=OUTPUT_ROOT)
    args = parser.parse_args()

    from trident.patch_encoder_models.load import encoder_factory

    print(f"Loading {args.model} encoder on {args.device} ...")
    encoder = encoder_factory(args.model)
    encoder.eval().to(args.device)
    encoder.compile()
    transform = encoder.eval_transforms

    yaml_lines = []
    for cohort in args.cohorts:
        pb_key     = COHORT_MAP[cohort]
        cohort_dir = os.path.join(args.nfs_root, cohort)
        output_dir = os.path.join(args.output_root, pb_key, args.model)

        print(f"\n=== {cohort} → {pb_key} ===")
        extract_cohort(cohort_dir, output_dir, encoder, transform,
                       args.device, args.batch_size, args.num_workers)
        yaml_lines.append(f"  {pb_key}:\n    {args.model}: {output_dir}")

    print("\n\nAdd to patch_embeddings_paths.yaml:")
    for line in yaml_lines:
        print(line)


if __name__ == "__main__":
    main()
