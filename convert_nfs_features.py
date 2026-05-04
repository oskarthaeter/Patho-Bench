"""
Convert pre-computed patch features from /mnt/nfs03-R6 into the .h5 format that
Patho-Bench expects, in two modes selectable via --mode:

  pooled (default)
    One {sample_id}.h5 per slide containing a 1-D pooled vector.
    Use with --pooled_dirs_yaml.

  patch
    One {sample_id}.h5 per slide containing the full [N_patches x D] matrix.
    Use with --patch_dirs_yaml (enables MIL training / custom poolers).

Datasets handled
----------------
bracs / uni      : BRACS/feats/feats_UNI/pt_files/*.pth        [N x 1024]
bracs / gigapath : BRACS/gigapath_embeddings/**/*.pt            already slide-level (pooled mode only)
imp   / uni      : IMP_CRC/feats/feats_UNI/pt_files/*.pth      [N x 1024]

Output layout (Patho-Bench appends by_{sample_col} automatically)
-----------------------------------------------------------------
pooled: <output_root>/<dataset>/<model>/by_slide_id/{sample_id}.h5   features shape [D]
patch : <output_root>/<dataset>/<model>/{sample_id}.h5               features shape [N x D]

Usage
-----
    # Pooled (mean) — for linprobe / coxnet
    python convert_nfs_features.py --mode pooled --nfs_root /mnt/nfs03-R6 --output_root /path/to/pooled

    # Patch-level — for MIL / finetune
    python convert_nfs_features.py --mode patch  --nfs_root /mnt/nfs03-R6 --output_root /path/to/patch

    # Both at once
    python convert_nfs_features.py --mode pooled patch --nfs_root /mnt/nfs03-R6 --output_root /path/to/out
"""

import argparse
import ctypes
import os
import glob
import h5py
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pth(fpath: str) -> torch.Tensor:
    data = torch.load(fpath, weights_only=True)
    return data["features"] if isinstance(data, dict) else data


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    # torch→numpy bridge is broken with NumPy 2.x; use ctypes direct pointer instead
    t = tensor.float().contiguous()
    storage = t.untyped_storage()
    buf = (ctypes.c_char * storage.nbytes()).from_address(storage.data_ptr())
    return np.frombuffer(buf, dtype=np.float32).reshape(t.shape).copy()


def _save_h5(path: str, array: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=array)


# ---------------------------------------------------------------------------
# Pooled conversions
# ---------------------------------------------------------------------------

def pooled_pth(src_dir: str, dst_dir: str):
    """Mean-pool [N x D] .pth files → [D] .h5 files."""
    files = sorted(glob.glob(os.path.join(src_dir, "*.pth")))
    if not files:
        print(f"  No .pth files in {src_dir}, skipping.")
        return
    converted = skipped = 0
    for fpath in files:
        slide_id = os.path.splitext(os.path.basename(fpath))[0]
        dst = os.path.join(dst_dir, f"{slide_id}.h5")
        if os.path.exists(dst):
            skipped += 1
            continue
        feats = _load_pth(fpath)
        _save_h5(dst, _to_numpy(feats.mean(dim=0)))
        converted += 1
    print(f"  {converted} converted, {skipped} skipped → {dst_dir}")


def pooled_gigapath(src_root: str, dst_dir: str, layer: str = "last_layer_embed"):
    """Extract one slide-level vector per .pt file (already pooled by GigaPath)."""
    files = sorted(glob.glob(os.path.join(src_root, "**", "*.pt"), recursive=True))
    if not files:
        print(f"  No .pt files under {src_root}, skipping.")
        return
    converted = skipped = 0
    for fpath in files:
        slide_id = os.path.splitext(os.path.basename(fpath))[0]
        dst = os.path.join(dst_dir, f"{slide_id}.h5")
        if os.path.exists(dst):
            skipped += 1
            continue
        data = torch.load(fpath, weights_only=True)
        _save_h5(dst, _to_numpy(data[layer].squeeze(0)))
        converted += 1
    print(f"  {converted} converted, {skipped} skipped → {dst_dir}")


# ---------------------------------------------------------------------------
# Patch-level conversions
# ---------------------------------------------------------------------------

def patch_pth(src_dir: str, dst_dir: str):
    """Copy [N x D] .pth patch matrices into [N x D] .h5 files (no pooling)."""
    files = sorted(glob.glob(os.path.join(src_dir, "*.pth")))
    if not files:
        print(f"  No .pth files in {src_dir}, skipping.")
        return
    converted = skipped = 0
    for fpath in files:
        slide_id = os.path.splitext(os.path.basename(fpath))[0]
        dst = os.path.join(dst_dir, f"{slide_id}.h5")
        if os.path.exists(dst):
            skipped += 1
            continue
        feats = _load_pth(fpath)
        _save_h5(dst, _to_numpy(feats))  # [N x D]
        converted += 1
    print(f"  {converted} converted, {skipped} skipped → {dst_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert NFS features to Patho-Bench .h5 format.")
    parser.add_argument("--nfs_root", default="/mnt/nfs03-R6")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--mode", nargs="+", choices=["pooled", "patch"], default=["pooled"],
                        help="pooled: mean-pool → use with pooled_dirs_yaml; "
                             "patch: keep all patches → use with patch_dirs_yaml (for MIL)")
    parser.add_argument("--datasets", nargs="+",
                        choices=["bracs_uni", "bracs_gigapath", "imp_uni"],
                        default=["bracs_uni", "bracs_gigapath", "imp_uni"])
    args = parser.parse_args()

    # Patho-Bench appends by_{sample_col} for pooled embeddings; patch dirs are used directly.
    POOLED_SUBDIR = "by_slide_id"

    if "pooled" in args.mode:
        print("──── POOLED MODE ────")
        if "bracs_uni" in args.datasets:
            print("=== BRACS / UNI ===")
            pooled_pth(
                src_dir=os.path.join(args.nfs_root, "BRACS", "feats", "feats_UNI", "pt_files"),
                dst_dir=os.path.join(args.output_root, "pooled", "bracs", "uni", POOLED_SUBDIR),
            )
        if "bracs_gigapath" in args.datasets:
            print("=== BRACS / GigaPath ===")
            pooled_gigapath(
                src_root=os.path.join(args.nfs_root, "BRACS", "gigapath_embeddings"),
                dst_dir=os.path.join(args.output_root, "pooled", "bracs", "gigapath", POOLED_SUBDIR),
            )
        if "imp_uni" in args.datasets:
            print("=== IMP / UNI ===")
            pooled_pth(
                src_dir=os.path.join(args.nfs_root, "IMP_CRC", "feats", "feats_UNI", "pt_files"),
                dst_dir=os.path.join(args.output_root, "pooled", "imp", "uni", POOLED_SUBDIR),
            )
        print(f"\nConfigure pooled_dirs_yaml:\n"
              f"bracs:\n"
              f"  uni:      {os.path.join(args.output_root, 'pooled', 'bracs', 'uni')}\n"
              f"  gigapath: {os.path.join(args.output_root, 'pooled', 'bracs', 'gigapath')}\n"
              f"imp:\n"
              f"  uni:      {os.path.join(args.output_root, 'pooled', 'imp', 'uni')}\n")

    if "patch" in args.mode:
        print("──── PATCH MODE ────")
        if "bracs_uni" in args.datasets:
            print("=== BRACS / UNI ===")
            patch_pth(
                src_dir=os.path.join(args.nfs_root, "BRACS", "feats", "feats_UNI", "pt_files"),
                dst_dir=os.path.join(args.output_root, "patch", "bracs", "uni"),
            )
        if "bracs_gigapath" in args.datasets:
            print("=== BRACS / GigaPath: patch mode not available (slide-level only), skipping. ===")
        if "imp_uni" in args.datasets:
            print("=== IMP / UNI ===")
            patch_pth(
                src_dir=os.path.join(args.nfs_root, "IMP_CRC", "feats", "feats_UNI", "pt_files"),
                dst_dir=os.path.join(args.output_root, "patch", "imp", "uni"),
            )
        print(f"\nConfigure patch_dirs_yaml:\n"
              f"bracs:\n"
              f"  uni: {os.path.join(args.output_root, 'patch', 'bracs', 'uni')}\n"
              f"imp:\n"
              f"  uni: {os.path.join(args.output_root, 'patch', 'imp', 'uni')}\n")


if __name__ == "__main__":
    main()
