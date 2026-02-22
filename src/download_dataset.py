import os
import shutil
from pathlib import Path

import yaml
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_kaggle_download(slug: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = f'kaggle datasets download -d "{slug}" -p "{out_dir}" --unzip'
    code = os.system(cmd)
    if code != 0:
        raise RuntimeError(
            "Kaggle download failed. Proveri da li imaš kaggle.json u C:\\Users\\nesto\\.kaggle\\kaggle.json"
        )


def find_dataset_root(raw_dir: Path) -> Path:
    candidate = raw_dir / "Dataset_BUSI_with_GT"
    if candidate.exists():
        return candidate

    for p in raw_dir.rglob("*"):
        if p.is_dir():
            if (p / "benign").exists() and (p / "malignant").exists() and (p / "normal").exists():
                return p
    raise FileNotFoundError("Ne mogu da pronađem dataset root (folder sa benign/malignant/normal).")


def remove_masks(dataset_root: Path, classes: list[str]) -> int:
    removed = 0
    for cls in classes:
        cls_dir = dataset_root / cls
        for f in cls_dir.glob("*"):
            if "mask" in f.name.lower():
                f.unlink(missing_ok=True)
                removed += 1
    return removed


def remove_corrupted_images(dataset_root: Path) -> int:
    removed = 0
    for img_path in tqdm(list(dataset_root.rglob("*.png")), desc="Checking images"):
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception:
            img_path.unlink(missing_ok=True)
            removed += 1
    return removed


def copy_clean_dataset(dataset_root: Path, clean_dir: Path, classes: list[str]):
    if clean_dir.exists():
        shutil.rmtree(clean_dir)
    for cls in classes:
        (clean_dir / cls).mkdir(parents=True, exist_ok=True)
        for img_path in (dataset_root / cls).glob("*.png"):
            shutil.copy2(img_path, clean_dir / cls / img_path.name)


def main():
    cfg = load_config()
    slug = cfg["dataset"]["kaggle_slug"]
    raw_dir = Path(cfg["dataset"]["raw_dir"])
    clean_dir = Path(cfg["dataset"]["clean_dir"])
    classes = cfg["dataset"]["classes"]

    print(f"📥 Downloading dataset: {slug}")
    run_kaggle_download(slug, raw_dir)

    dataset_root = find_dataset_root(raw_dir)
    print(f"✅ Found dataset root: {dataset_root}")

    masks = remove_masks(dataset_root, classes)
    print(f"🧽 Removed mask files: {masks}")

    corrupted = remove_corrupted_images(dataset_root)
    print(f"🧹 Removed corrupted images: {corrupted}")

    copy_clean_dataset(dataset_root, clean_dir, classes)
    print(f"✅ Clean dataset prepared at: {clean_dir}")
    print("Done.")


if __name__ == "__main__":
    main()