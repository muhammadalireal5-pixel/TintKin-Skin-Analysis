"""Train the face-age model on FairFace (CC BY 4.0) and export it to ONNX for CPU serving.

Usage: python scripts/train_age.py [--epochs 8] [--batch 128]
FairFace has 9 age bins; the model classifies the bin and age = softmax-weighted bin midpoint.
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "fairface"
TIGHT_FRAC = (1 + 0.25) / (1 + 1.25)  # centre crop that turns a 1.25-padding image into the 0.25-padding view
OUT = ROOT / "models"
MIDPOINTS = torch.tensor([1, 6, 15, 25, 35, 45, 55, 65, 75], dtype=torch.float32)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _build_cache(split: str, version: str = "0.25") -> Path:
    """Pack a split's JPEG bytes into one flat file so workers can memory-map it (RAM is tight)."""
    cdir = ROOT / "data" / "fairface_cache"
    cdir.mkdir(exist_ok=True)
    name = split if version == "0.25" else f"{split}_{version.replace('.', '')}"
    blob, meta = cdir / f"{name}.bin", cdir / f"{name}.npz"
    if blob.exists() and meta.exists():
        return cdir / name
    files = sorted((DATA_ROOT / version).glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {split} parquet in {DATA_ROOT / version}")
    offsets, ages, pos = [0], [], 0
    with open(blob, "wb") as out:
        for f in files:
            df = pd.read_parquet(f, columns=["image", "age"])
            for d, a in zip(df["image"], df["age"]):
                out.write(d["bytes"])
                pos += len(d["bytes"])
                offsets.append(pos)
                ages.append(a)
            del df
    np.savez(meta, offsets=np.array(offsets, np.int64), age=np.array(ages, np.int64))
    return cdir / name


class _TightCrop:
    """Centre crop a 1.25-padding image to the 0.25-padding view (a class, not a lambda, so workers can pickle it)."""

    def __call__(self, im: Image.Image) -> Image.Image:
        return T.functional.center_crop(im, int(im.width * TIGHT_FRAC))


class FairFaceAge(Dataset):
    def __init__(self, split: str, train: bool, version: str = "0.25"):
        self.base = _build_cache(split, version)
        m = np.load(f"{self.base}.npz")
        self.offsets, self.age = m["offsets"], m["age"]
        self.buf = None  # opened lazily per worker
        norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
        aug = [T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
               T.RandomHorizontalFlip(), T.ColorJitter(0.3, 0.3, 0.2, 0.02)]
        pre = [_TightCrop()] if version == "1.25" else []
        self.tf = T.Compose(pre + (aug if train else [T.Resize((224, 224))]) + norm)

    def __len__(self) -> int:
        return len(self.age)

    def __getitem__(self, i: int):
        if self.buf is None:
            self.buf = np.memmap(f"{self.base}.bin", dtype=np.uint8, mode="r")
        raw = self.buf[self.offsets[i]:self.offsets[i + 1]].tobytes()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return self.tf(img), int(self.age[i])


def build_model() -> nn.Module:
    return timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=9)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, dev: str) -> tuple[float, float]:
    model.eval()
    correct = n = 0
    abs_err = 0.0
    mids = MIDPOINTS.to(dev)
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        p = Fn.softmax(model(x), 1)
        correct += (p.argmax(1) == y).sum().item()
        abs_err += (p @ mids - mids[y]).abs().sum().item()
        n += len(y)
    return correct / n, abs_err / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--extra125", action="store_true", help="also train on the 448px 1.25 images (tight-cropped)")
    ap.add_argument("--init", type=Path, default=None, help="start from these weights (e.g. models/age_best.pt)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}", flush=True)
    tr, va = FairFaceAge("train", True), FairFaceAge("validation", False)
    if args.extra125:
        tr = torch.utils.data.ConcatDataset([tr, FairFaceAge("train", True, "1.25")])
    print(f"train={len(tr)} val={len(va)}", flush=True)
    tl = DataLoader(tr, args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                    persistent_workers=args.workers > 0)
    vl = DataLoader(va, args.batch * 2, num_workers=2)

    model = build_model()
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu"))
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tl))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    mids = MIDPOINTS.to(dev)
    best = 1e9
    OUT.mkdir(exist_ok=True)

    for ep in range(args.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for i, (x, y) in enumerate(tl):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev, enabled=dev == "cuda"):
                logits = model(x)
            logits = logits.float()
            p = Fn.softmax(logits, 1)
            # CE for the bin + L1 on the expected age so near-misses are cheaper than far ones
            loss = Fn.cross_entropy(logits, y, label_smoothing=0.05) \
                + 0.02 * (p @ mids - mids[y]).abs().mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            if i % 100 == 0:
                print(f"ep{ep} it{i}/{len(tl)} loss={loss.item():.3f}", flush=True)
        acc, mae = evaluate(model, vl, dev)
        print(f"== epoch {ep}: train_loss={tot / len(tl):.3f} val_acc={acc:.3f} val_MAE={mae:.2f}y "
              f"({time.time() - t0:.0f}s)", flush=True)
        if mae < best:
            best = mae
            torch.save(model.state_dict(), OUT / "age_best.pt")

    model.load_state_dict(torch.load(OUT / "age_best.pt", map_location="cpu"))
    model.cpu().eval()
    torch.onnx.export(model, torch.zeros(1, 3, 224, 224), str(OUT / "age.onnx"),
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "b"}, "logits": {0: "b"}}, opset_version=17)
    print(f"exported {OUT / 'age.onnx'} (best val MAE {best:.2f}y)", flush=True)


if __name__ == "__main__":
    main()
