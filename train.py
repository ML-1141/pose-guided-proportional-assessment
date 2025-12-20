from __future__ import annotations

"""
This script runs end-to-end training and then Grad-CAM inference.

Required CLI arguments:
  - root_dir:   dataset root that contains class folders
  - save_path:  where to save the trained model state_dict
  - ckpt_path:  which checkpoint to load for inference (if missing, falls back to save_path)
  - out_root:   output directory for saved heatmaps/overlays

python train.py <root_dir> <save_path> <ckpt_path> <out_root> --use_stn <true/false>
"""


import argparse
import glob
import os
import random
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader

# Ensure local imports work regardless of the current working directory.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from model import SiameseSTN_AMCNN
from loss import CosineEmbeddingLossMargin
from utils import evaluate_similarity
from utils import ResizeWithPad
from dataset import SiamesePoseDatasetSK

def create_model(
    in_channels: int = 3,
    feature_dim: int = 64, 
    margin: float = 0.7,
    use_stn: bool = True, # False for deterministic method
) -> Tuple[SiameseSTN_AMCNN, CosineEmbeddingLossMargin]:
    model = SiameseSTN_AMCNN(in_channels, feature_dim, use_stn=use_stn)
    loss_fn = CosineEmbeddingLossMargin(margin)
    return model, loss_fn


def train_step(
    model: SiameseSTN_AMCNN,
    loss_fn: CosineEmbeddingLossMargin,
    optimizer: torch.optim.Optimizer,
    batch_left: torch.Tensor,
    batch_right: torch.Tensor,
    labels: torch.Tensor
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    z1, z2 = model(batch_left, batch_right)
    loss = loss_fn(z1, z2, labels)
    loss.backward()
    optimizer.step()
    return float(loss.item())

def _init_wandb(batch_size: int, lr: float, num_epochs: int, margin: float, feature_dim: int, use_stn: bool):
    """Best-effort wandb init. If unavailable/misconfigured, training still runs."""
    try:
        import wandb  # type: ignore
    except Exception:
        print("[INFO] wandb not installed; wandb logging disabled.")
        return None

    key = os.environ.get("WANDB_API_KEY", "").strip()
    if not key:
        print("[INFO] WANDB_API_KEY not set; wandb logging disabled.")
        return None

    try:
        wandb.login(key=key)
        wandb.init(
            project="st_amcnn_pose",
            name="run_siamese_stamcnn_no_stn_bend",
            config={
                "batch_size": batch_size,
                "lr": lr,
                "num_epochs": num_epochs,
                "margin": margin,
                "feature_dim": feature_dim,
                "use_stn": use_stn,
            },
        )
        return wandb
    except Exception as e:
        print(f"[INFO] wandb init failed ({type(e).__name__}: {e}); wandb logging disabled.")
        return None
    
def _str2bool(v: str) -> bool:
    """Parse common boolean strings for argparse."""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v!r} (use true/false)")

    
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Siamese STN-AMCNN then run Grad-CAM inference.")
    parser.add_argument("root_dir", type=str, help="Dataset root directory")
    parser.add_argument("save_path", type=str, help="Path to save trained checkpoint (state_dict)")
    parser.add_argument("out_root", type=str, help="Directory to write inference outputs")
    parser.add_argument("--use_stn", type=_str2bool, default=True, help="Enable STN module (true/false). Default: true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    # --------------------------------------------------
    # CONFIG
    # --------------------------------------------------
    root_dir = args.root_dir
    use_stn = args.use_stn

    # support multiple classes now
    TARGET_CLASSES = [
        "6_bend_knee_120",
        "4_stand_arm_on_hip_141",
        "2_stand_open_arm_156",
    ]

    rng = random.Random(42)
    class_to_all_sk = {}

    # --------------------------------------------------
    # COLLECT ALL sk IMAGES FOR EACH TARGET CLASS
    # --------------------------------------------------
    for cls in TARGET_CLASSES:
        cls_sk_root = os.path.join(root_dir, cls, "sk")
        if not os.path.isdir(cls_sk_root):
            print(f"[WARN] sk folder not found for class: {cls}")
            continue

        all_paths = []
        # loop through every subfolder inside sk/ (original, swirl_180, etc.)
        for sub in os.listdir(cls_sk_root):
            sub_dir = os.path.join(cls_sk_root, sub)
            if not os.path.isdir(sub_dir):
                continue

            imgs = glob.glob(os.path.join(sub_dir, "*.*"))
            imgs = [p for p in imgs
                    if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
            all_paths.extend(imgs)

        if not all_paths:
            print(f"[WARN] no images found in sk/* for class: {cls}")
            continue

        rng.shuffle(all_paths)
        class_to_all_sk[cls] = all_paths
        print(f"{cls}: total SK images = {len(all_paths)}")

    # --------------------------------------------------
    # HARD SPLIT PER CLASS: TRAIN / VAL / TEST
    # --------------------------------------------------
    train_ratio = 0.7
    val_ratio   = 0.1   # test = 1 - train - val

    class_to_train = {}
    class_to_val   = {}
    class_to_test  = {}

    for cls, paths in class_to_all_sk.items():
        n = len(paths)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        class_to_train[cls] = paths[:n_train]
        class_to_val[cls]   = paths[n_train:n_train + n_val]
        class_to_test[cls]  = paths[n_train + n_val:]

        print(
            f"{cls}: total={n}, "
            f"train={len(class_to_train[cls])}, "
            f"val={len(class_to_val[cls])}, "
            f"test={len(class_to_test[cls])}"
        )



    transform = transforms.Compose([
        ResizeWithPad(156, fill=0),   # keep figure proportions
        transforms.ToTensor(),
    ])


    # create datasets from the pre-split dicts
    train_dataset = SiamesePoseDatasetSK(class_to_train, transform=transform, pos_ratio=0.5)
    val_dataset   = SiamesePoseDatasetSK(class_to_val,   transform=transform, pos_ratio=0.5)
    test_dataset  = SiamesePoseDatasetSK(class_to_test,  transform=transform, pos_ratio=0.5)

    print("Train images (sk only):", len(train_dataset))
    print("Val images   (sk only):", len(val_dataset))
    print("Test images  (sk only):", len(test_dataset))

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model, loss_fn = create_model(in_channels=3, feature_dim=64, margin=0.5, use_stn=use_stn)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    num_epochs = 100
    wandb = _init_wandb(batch_size=batch_size, lr=5e-4, num_epochs=num_epochs, margin=0.5, feature_dim=64)
    if wandb is not None:
        try:
            wandb.watch(model, loss_fn, log="gradients", log_freq=50)
        except Exception:
            # Non-critical.
            pass


    # --------------------RUN TRAINING--------------------

    def run_epoch(loader, train=True):
        if train:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        n = 0

        for img1, img2, labels in loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)

            if train:
                loss_val = train_step(model, loss_fn, optimizer, img1, img2, labels)
            else:
                with torch.no_grad():
                    z1, z2 = model(img1, img2)
                    loss_val = float(loss_fn(z1, z2, labels).item())

            total_loss += loss_val
            n += 1

        return total_loss / n

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    try:
        for epoch in range(1, num_epochs + 1):
            train_loss = run_epoch(train_loader, train=True)
            val_loss = run_epoch(val_loader, train=False)

            if wandb is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    }
                )

            print(f"Epoch {epoch:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    except KeyboardInterrupt:
        # --------------------SAVE CHECKPOINT ON INTERRUPT--------------------
        torch.save(model.state_dict(), save_path)
        print("\n[INFO] KeyboardInterrupt detected. Saved current model state at:", save_path)

    torch.save(model.state_dict(), save_path)
    print("✔ Saved current model state at:", save_path)

    # --------------------SAVE MODEL--------------------

    save_path = args.save_path
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print("✔ Saved current model state at:", save_path)


    # --------------------INFERENCE--------------------

    from gradcam import GradCAMWrapper

    ckpt_path = args.ckpt_path
    if not os.path.isfile(ckpt_path):
        print(f"[WARN] ckpt_path not found: {ckpt_path}. Falling back to save_path: {save_path}")
        ckpt_path = save_path
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    # Rebuild model (kept as in the original script) and load checkpoint.
    model, loss_fn = create_model(in_channels=3, feature_dim=64, margin=0.5, use_stn=use_stn)
    model.to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("Model loaded from:", ckpt_path)

    cam = GradCAMWrapper(model)  # model = SiameseSTAMCNN (the one you trained)

    out_root = args.out_root
    os.makedirs(out_root, exist_ok=True)
    print("Saving heatmaps to:", out_root)



    class_to_std_sk = {}       # class → standard reference images (ONLY sk/original)
    class_to_test_learner = {} # class → learner images (test split, excluding original)

    for cls in TARGET_CLASSES:
        cls_path = os.path.join(root_dir, cls)
        # ---------------------------------------------
        # STANDARD: sk/original/*
        # ---------------------------------------------
        orig_dir = os.path.join(cls_path, "sk", "original")
        std_paths = glob.glob(os.path.join(orig_dir, "*.*")) if os.path.isdir(orig_dir) else []
        std_paths = [
            p for p in std_paths
            if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
        class_to_std_sk[cls] = std_paths

        # ---------------------------------------------
        # LEARNER: test split paths, but EXCLUDE original/
        # ---------------------------------------------
        test_paths = class_to_test.get(cls, [])

        learner_paths = [
            p for p in test_paths
            if os.path.sep + "original" + os.path.sep not in p
        ]

        class_to_test_learner[cls] = learner_paths

        print(f"\nClass: {cls}")
        print("  #standard (sk/original):", len(class_to_std_sk[cls]))
        print("  #test learners (test split, non-original):", len(class_to_test_learner[cls]))




    import cv2
    import re
    pad_resize = ResizeWithPad(156)
    rng = random.Random(73)  # reproducible

    def load_img_eval(path):
        img = Image.open(path).convert("RGB")
        img = pad_resize(img)
        return transforms.ToTensor()(img).unsqueeze(0).to(device)

    def extract_id(path):
        filename = os.path.basename(path)
        nums = re.findall(r"\d+", filename)
        return nums[-1] if nums else filename.split(".")[0]


    for cls in class_to_test_learner.keys():
        learners = class_to_test_learner[cls]
        std_list = class_to_std_sk[cls]

        if len(learners) == 0 or len(std_list) == 0:
            print(f"Skip {cls}: no learners or standards.")
            continue

        print(f"\n=== Processing class: {cls} ===")

        for learner_path in learners:

            # -------------------------------------------------------
            # 0. learner ID
            # -------------------------------------------------------
            learner_id = extract_id(learner_path)

            # -------------------------------------------------------
            # 1. Randomly sample 5 standards, but EXCLUDE same ID
            # -------------------------------------------------------
            eligible_std = [p for p in std_list if extract_id(p) != learner_id]

            if len(eligible_std) < 5:
                chosen_std_paths = eligible_std  # take all if less than 5
            else:
                chosen_std_paths = rng.sample(eligible_std, 5)

            # -------------------------------------------------------
            # 2. Compute similarity vs these 5 and pick the best
            # -------------------------------------------------------
            img_learner = load_img_eval(learner_path)

            best_sim = -999
            best_std_path = None

            for std_path in chosen_std_paths:
                img_std = load_img_eval(std_path)
                sim = evaluate_similarity(model, img_std, img_learner)[0].item()
                if sim > best_sim:
                    best_sim = sim
                    best_std_path = std_path

            # load the chosen best standard (again for CAM)
            img_std_best = load_img_eval(best_std_path)

            # -------------------------------------------------------
            # 3. Augmentation name detection
            # -------------------------------------------------------
            parts = learner_path.split(os.sep)
            if "sk" in parts:
                sk_idx = parts.index("sk")
                mod_name = parts[sk_idx + 1]
            else:
                mod_name = "unknown"

            save_dir = os.path.join(out_root, mod_name)
            os.makedirs(save_dir, exist_ok=True)

            # -------------------------------------------------------
            # 4. Grad-CAM using best standard
            # -------------------------------------------------------
            heatmap = cam.heatmap_vs_reference(
                image_tensor=img_learner,
                reference_tensor=img_std_best
            )
            h_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

            # -------------------------------------------------------
            # 5. Prepare images for saving
            # -------------------------------------------------------
            def pad_resize_bgr(path):
                im = Image.open(path).convert("RGB")
                im = pad_resize(im)
                arr = np.array(im)[:, :, ::-1]
                return arr

            learner_bgr = pad_resize_bgr(learner_path)
            std_bgr = pad_resize_bgr(best_std_path)
            overlay_bgr = GradCAMWrapper.overlay_on_image(h_norm, learner_bgr)

            # -------------------------------------------------------
            # 6. Save
            # -------------------------------------------------------
            cv2.imwrite(os.path.join(save_dir, f"{learner_id}_learner.png"), learner_bgr)
            cv2.imwrite(os.path.join(save_dir, f"{learner_id}_standard.png"), std_bgr)
            cv2.imwrite(os.path.join(save_dir, f"{learner_id}_overlay.png"), overlay_bgr)

            print(f"Saved {learner_id} | best sim={best_sim:.4f} | standard={os.path.basename(best_std_path)}")
            



if __name__ == "__main__":
    main()
