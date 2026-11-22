"""
Training pipeline for Vanilla GAN and WGAN-GP on Ames Housing Tabular Data.
Workshop No. 5 — Generative Adversarial Networks.

Group 1:
- Chavez Porras Miguel Alejandro (20212020033)
- Guerrero Ordeñez Luisa Fernanda (20212020099)
- Arboleda Zambrano Dilan Stive (20212020105)
"""

from __future__ import annotations
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Resolve project imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import load_and_preprocess_data  # noqa: E402
from gan import TabularGenerator, TabularDiscriminator, TabularCritic, compute_gradient_penalty  # noqa: E402

RANDOM_SEED = 42
DEVICE = torch.device("cpu")


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TabularDataBundle:
    """Manages tabular data matrix [X, y_log], fitted MinMax normalization in [-1, 1], and inverse transforms."""

    def __init__(self, seed: int = RANDOM_SEED):
        set_seed(seed)
        bundle = load_and_preprocess_data(seed)
        
        self.feature_names = bundle["feature_names"] + ["target_log_saleprice"]
        self.preprocessor = bundle["preprocessor"]

        # Concatenate features and target
        X_tr = bundle["X_train"]
        y_tr = bundle["y_train_log"][:, None]
        train_raw = np.hstack([X_tr, y_tr])

        X_va = bundle["X_val"]
        y_va = bundle["y_val_log"][:, None]
        val_raw = np.hstack([X_va, y_va])

        X_te = bundle["X_test"]
        y_te = bundle["y_test_log"][:, None]
        test_raw = np.hstack([X_te, y_te])

        # Fit MinMax scaler strictly on train set to range [-1, 1]
        self.scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        self.train_norm = self.scaler.fit_transform(train_raw).astype(np.float32)
        self.val_norm = self.scaler.transform(val_raw).astype(np.float32)
        self.test_norm = self.scaler.transform(test_raw).astype(np.float32)

        self.input_dim = self.train_norm.shape[1]
        print(f"[TabularDataBundle] Initialized. Dimension: {self.input_dim} (260 features + 1 target).")
        print(f"  Train: {self.train_norm.shape}, Val: {self.val_norm.shape}, Test: {self.test_norm.shape}")

    def inverse_transform(self, normalized_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert normalized [-1, 1] samples back to (X_features, y_log, y_dollar)."""
        denorm = self.scaler.inverse_transform(normalized_samples)
        X_features = denorm[:, :-1]
        y_log = denorm[:, -1]
        # Invert log target
        y_dollar = np.expm1(np.clip(y_log, 0.0, 15.0))
        return X_features, y_log, y_dollar


def train_vanilla_gan(
    bundle: TabularDataBundle,
    latent_dim: int = 64,
    epochs: int = 150,
    batch_size: int = 64,
    lr: float = 2e-4,
    label_smoothing: float = 0.9,
    save_dir: Path = None,
) -> dict:
    """Train Vanilla GAN with BCE loss, Adam optimizers, and label smoothing."""
    print("=" * 80)
    print("TRAINING VANILLA GAN (Ames Housing Tabular)")
    print("=" * 80)
    set_seed(RANDOM_SEED)

    generator = TabularGenerator(latent_dim=latent_dim, output_dim=bundle.input_dim).to(DEVICE)
    discriminator = TabularDiscriminator(input_dim=bundle.input_dim).to(DEVICE)

    criterion = nn.BCELoss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    dataset = TensorDataset(torch.from_numpy(bundle.train_norm))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    history = {
        "epoch": [],
        "d_loss": [],
        "g_loss": [],
        "d_acc_real": [],
        "d_acc_fake": [],
    }

    best_g_loss = float("inf")
    best_gen_state = None

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()

        d_losses = []
        g_losses = []
        d_acc_reals = []
        d_acc_fakes = []

        for (real_batch,) in loader:
            real_batch = real_batch.to(DEVICE)
            b_size = real_batch.size(0)

            # Labels with one-sided label smoothing for real samples
            real_labels = torch.full((b_size,), label_smoothing, device=DEVICE)
            fake_labels = torch.zeros(b_size, device=DEVICE)

            # -------------------------------------------------------------
            # 1. Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            # -------------------------------------------------------------
            opt_d.zero_grad()

            pred_real = discriminator(real_batch)
            loss_d_real = criterion(pred_real, real_labels)

            z = torch.randn(b_size, latent_dim, device=DEVICE)
            fake_batch = generator(z)
            pred_fake = discriminator(fake_batch.detach())
            loss_d_fake = criterion(pred_fake, fake_labels)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()

            # Discriminator Accuracies
            acc_real = (pred_real >= 0.5).float().mean().item()
            acc_fake = (pred_fake < 0.5).float().mean().item()

            # -------------------------------------------------------------
            # 2. Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z)))
            # -------------------------------------------------------------
            opt_g.zero_grad()
            # Non-saturating heuristic: target = 1.0
            g_target_labels = torch.ones(b_size, device=DEVICE)
            pred_fake_for_g = discriminator(fake_batch)
            loss_g = criterion(pred_fake_for_g, g_target_labels)

            loss_g.backward()
            opt_g.step()

            d_losses.append(loss_d.item())
            g_losses.append(loss_g.item())
            d_acc_reals.append(acc_real)
            d_acc_fakes.append(acc_fake)

        mean_d = float(np.mean(d_losses))
        mean_g = float(np.mean(g_losses))
        mean_acc_real = float(np.mean(d_acc_reals))
        mean_acc_fake = float(np.mean(d_acc_fakes))

        history["epoch"].append(epoch)
        history["d_loss"].append(mean_d)
        history["g_loss"].append(mean_g)
        history["d_acc_real"].append(mean_acc_real)
        history["d_acc_fake"].append(mean_acc_fake)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"[Vanilla GAN | Epoch {epoch:>3}/{epochs}] D_Loss: {mean_d:.4f} | G_Loss: {mean_g:.4f} | D_Acc(Real): {mean_acc_real*100:.1f}% | D_Acc(Fake): {mean_acc_fake*100:.1f}%")

        if mean_g < best_g_loss and epoch >= 50:
            best_g_loss = mean_g
            best_gen_state = generator.state_dict()

    if best_gen_state is None:
        best_gen_state = generator.state_dict()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / "gan_vanilla_best.pt"
        torch.save({
            "generator": best_gen_state,
            "latent_dim": latent_dim,
            "output_dim": bundle.input_dim,
            "history": history,
            "scaler_min": bundle.scaler.min_,
            "scaler_scale": bundle.scaler.scale_,
        }, ckpt_path)
        print(f"Saved Vanilla GAN checkpoint to {ckpt_path}")

    return {"generator": generator, "history": history}


def train_wgan_gp(
    bundle: TabularDataBundle,
    latent_dim: int = 64,
    epochs: int = 150,
    batch_size: int = 64,
    lr: float = 2e-4,
    n_critic: int = 5,
    lambda_gp: float = 10.0,
    save_dir: Path = None,
) -> dict:
    """Train WGAN-GP with 1-Lipschitz gradient penalty and n_critic schedule."""
    print("=" * 80)
    print("TRAINING WGAN-GP (Ames Housing Tabular)")
    print("=" * 80)
    set_seed(RANDOM_SEED)

    generator = TabularGenerator(latent_dim=latent_dim, output_dim=bundle.input_dim).to(DEVICE)
    critic = TabularCritic(input_dim=bundle.input_dim).to(DEVICE)

    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))

    dataset = TensorDataset(torch.from_numpy(bundle.train_norm))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    history = {
        "epoch": [],
        "c_loss": [],
        "g_loss": [],
        "wasserstein_distance": [],
        "gp": [],
    }

    best_w_dist = -float("inf")
    best_gen_state = None

    for epoch in range(1, epochs + 1):
        generator.train()
        critic.train()

        c_losses = []
        g_losses = []
        w_dists = []
        gps = []

        batches = list(loader)
        i = 0
        while i < len(batches):
            # -------------------------------------------------------------
            # 1. Train Critic n_critic steps
            # -------------------------------------------------------------
            for _ in range(n_critic):
                if i >= len(batches):
                    break
                real_batch = batches[i][0].to(DEVICE)
                b_size = real_batch.size(0)
                i += 1

                opt_c.zero_grad()

                # Real scores
                critic_real = critic(real_batch)

                # Fake scores
                z = torch.randn(b_size, latent_dim, device=DEVICE)
                fake_batch = generator(z)
                critic_fake = critic(fake_batch.detach())

                # Gradient penalty
                gp = compute_gradient_penalty(critic, real_batch, fake_batch.detach(), DEVICE)

                # Wasserstein loss: E[D(x_fake)] - E[D(x_real)] + lambda * GP
                loss_c = critic_fake.mean() - critic_real.mean() + lambda_gp * gp
                loss_c.backward()
                opt_c.step()

                w_dist = (critic_real.mean() - critic_fake.mean()).item()
                c_losses.append(loss_c.item())
                w_dists.append(w_dist)
                gps.append(gp.item())

            # -------------------------------------------------------------
            # 2. Train Generator: min -E[D(G(z))]
            # -------------------------------------------------------------
            opt_g.zero_grad()
            z = torch.randn(batch_size, latent_dim, device=DEVICE)
            gen_fake = generator(z)
            critic_gen = critic(gen_fake)

            loss_g = -critic_gen.mean()
            loss_g.backward()
            opt_g.step()

            g_losses.append(loss_g.item())

        mean_c = float(np.mean(c_losses))
        mean_g = float(np.mean(g_losses))
        mean_w = float(np.mean(w_dists))
        mean_gp = float(np.mean(gps))

        history["epoch"].append(epoch)
        history["c_loss"].append(mean_c)
        history["g_loss"].append(mean_g)
        history["wasserstein_distance"].append(mean_w)
        history["gp"].append(mean_gp)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"[WGAN-GP     | Epoch {epoch:>3}/{epochs}] Critic Loss: {mean_c:.4f} | W_Dist: {mean_w:.4f} | G_Loss: {mean_g:.4f} | GP: {mean_gp:.4f}")

        if epoch >= 50 and mean_w > best_w_dist:
            best_w_dist = mean_w
            best_gen_state = generator.state_dict()

    if best_gen_state is None:
        best_gen_state = generator.state_dict()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / "gan_wgan_best.pt"
        torch.save({
            "generator": best_gen_state,
            "latent_dim": latent_dim,
            "output_dim": bundle.input_dim,
            "history": history,
            "scaler_min": bundle.scaler.min_,
            "scaler_scale": bundle.scaler.scale_,
        }, ckpt_path)
        print(f"Saved WGAN-GP checkpoint to {ckpt_path}")

    return {"generator": generator, "history": history}


def main():
    set_seed(RANDOM_SEED)
    bundle = TabularDataBundle(seed=RANDOM_SEED)

    w5_dir = ROOT / "workshop5_gans"
    ckpt_dir = w5_dir / "checkpoints"
    runs_dir = w5_dir / "runs" / "workshop5"
    results_dir = w5_dir / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Train Vanilla GAN
    res_vanilla = train_vanilla_gan(
        bundle=bundle,
        latent_dim=64,
        epochs=150,
        batch_size=64,
        lr=2e-4,
        label_smoothing=0.9,
        save_dir=ckpt_dir,
    )

    # 2. Train WGAN-GP
    res_wgan = train_wgan_gp(
        bundle=bundle,
        latent_dim=64,
        epochs=150,
        batch_size=64,
        lr=2e-4,
        n_critic=5,
        lambda_gp=10.0,
        save_dir=ckpt_dir,
    )

    # Also copy checkpoints to project root checkpoints/
    import shutil
    root_ckpt = ROOT / "checkpoints"
    root_ckpt.mkdir(parents=True, exist_ok=True)
    shutil.copy(ckpt_dir / "gan_vanilla_best.pt", root_ckpt / "gan_vanilla_best.pt")
    shutil.copy(ckpt_dir / "gan_wgan_best.pt", root_ckpt / "gan_wgan_best.pt")

    # Save histories to JSON in results/
    with open(results_dir / "vanilla_gan_history.json", "w", encoding="utf-8") as f:
        json.dump(res_vanilla["history"], f, indent=2)

    with open(results_dir / "wgan_gp_history.json", "w", encoding="utf-8") as f:
        json.dump(res_wgan["history"], f, indent=2)

    print("\nTraining completed successfully! Histories and checkpoints persisted.")


if __name__ == "__main__":
    main()
