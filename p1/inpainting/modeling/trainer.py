import json
from copy import copy
from pathlib import Path
from random import Random

import torch
from hydra.utils import instantiate
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from p1.inpainting.data import DIV2KDataset


def split_train_val(dataset: DIV2KDataset, val_split: float = 0.2, seed: int = 42):
    paths = list(dataset.image_paths)
    Random(seed).shuffle(paths)
    valid_count = int(len(paths) * val_split)
    if not 1 <= valid_count < len(paths):
        raise ValueError("val_split must leave at least one image in each split")

    train_ds, valid_ds = copy(dataset), copy(dataset)
    train_ds.image_paths = paths[valid_count:]
    valid_ds.image_paths = paths[:valid_count]
    valid_ds.num_samples = None
    valid_ds.random_images = False
    train_ds.set_seed(seed)
    valid_ds.set_seed(seed)
    return train_ds, valid_ds


class InpaintingTrainer:
    def __init__(self, model: nn.Module, criterion: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.device = device

    def train(
        self,
        train_ds: DIV2KDataset,
        output_dir: Path,
        epochs: int = 10,
        batch_size: int = 4,
        lr: float = 2e-4,
        seed: int = 42,
        val_size: float = 0.2,
        save_epochs: int = 10,
        resume: bool = False,
        num_workers: int = 4,
        log_every: int = 20,
        scheduler: dict | None = None,
    ):
        model, criterion, device = self.model, self.criterion, self.device
        train_ds, valid_ds = split_train_val(train_ds, val_size, seed)
        if not 1 <= batch_size <= len(train_ds):
            raise ValueError("batch_size must be at least 1 and no larger than the training dataset")

        torch.manual_seed(seed)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = instantiate(scheduler, optimizer=optimizer) if scheduler is not None else None
        component_names = list(criterion.weights)
        print(f"Loss weights: {criterion.weights}")
        pin_memory = torch.device(device).type == "cuda"
        loader_options = dict(
            batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
        train_loader = DataLoader(train_ds, generator=torch.Generator().manual_seed(seed), **loader_options)
        valid_loader = DataLoader(valid_ds, generator=torch.Generator().manual_seed(seed), **loader_options)
        history = []
        best_loss = float("inf")
        start_epoch = 0
        state_path = output_dir / "state.pt"
        if resume:
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            if scheduler is not None and state.get("scheduler") is not None:
                scheduler.load_state_dict(state["scheduler"])
            else:
                for group in optimizer.param_groups:
                    group["lr"] = lr
            history = state["history"]
            best_loss = state["best_loss"]
            start_epoch = state["epoch"]
            torch.set_rng_state(state["torch_rng_state"])
            print(f"Resuming after epoch {start_epoch}, target: {epochs} epochs.")
        else:
            torch.save(model.state_dict(), checkpoints_dir / "0.pt")

        for epoch in range(start_epoch, epochs):
            model.train()
            train_loss = torch.zeros((), device=device)
            grad_norm_sum = torch.zeros((), device=device)
            train_components = torch.zeros(len(component_names), device=device)
            train_count = 0
            progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for step, (cut, mask, target) in enumerate(progress, 1):
                cut, mask, target = (tensor.to(device, non_blocking=pin_memory) for tensor in (cut, mask, target))
                optimizer.zero_grad()
                prediction = model(cut, mask)
                loss, components = criterion(prediction, mask, target, return_components=True)
                loss.backward()
                grad_norm = torch.nn.utils.get_total_norm(
                    [parameter.grad for parameter in model.parameters() if parameter.grad is not None],
                    foreach=True,
                )
                optimizer.step()
                train_loss += loss.detach() * cut.shape[0]
                grad_norm_sum += grad_norm
                train_components += torch.stack([components[name].detach() for name in component_names]) * cut.shape[0]
                train_count += cut.shape[0]
                if step % log_every == 0 or step == len(train_loader):
                    loss_value, norm_value = torch.stack((loss.detach(), grad_norm)).tolist()
                    progress.set_postfix(loss=f"{loss_value:.4f}", grad_norm=f"{norm_value:.4f}")

            model.eval()
            valid_loss = torch.zeros((), device=device)
            valid_components = torch.zeros(len(component_names), device=device)
            hole_error = torch.zeros((), device=device)
            hole_values = torch.zeros((), device=device)
            with torch.no_grad():
                for cut, mask, target in valid_loader:
                    cut, mask, target = (tensor.to(device, non_blocking=pin_memory) for tensor in (cut, mask, target))
                    prediction = model(cut, mask)
                    loss, components = criterion(prediction, mask, target, return_components=True)
                    valid_loss += loss * cut.shape[0]
                    valid_components += torch.stack([components[name] for name in component_names]) * cut.shape[0]
                    hole_error += ((prediction - target).abs() * mask).sum()
                    hole_values += mask.sum() * target.shape[1]
            train_loss, valid_loss, valid_hole_mae, mean_grad_norm = torch.stack((
                train_loss / train_count, valid_loss / len(valid_ds), hole_error / hole_values,
                grad_norm_sum / len(train_loader),
            )).tolist()
            component_values = torch.stack((train_components / train_count, valid_components / len(valid_ds))).tolist()
            component_history = {}
            for split, values in zip(("train_components", "valid_components"), component_values):
                component_history[split] = {
                    name: {"raw": value, "weighted": value * criterion.weights[name]}
                    for name, value in zip(component_names, values)
                }
            current_lr = optimizer.param_groups[0]["lr"]
            history.append({
                "epoch": epoch + 1, "train_loss": train_loss, "valid_loss": valid_loss,
                "valid_hole_mae": valid_hole_mae, "lr": current_lr,
                "grad_norm": mean_grad_norm, **component_history,
            })
            print(
                f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}, "
                f"hole_mae={valid_hole_mae:.4f}, lr={current_lr:g}, grad_norm={mean_grad_norm:.4f}"
            )
            if scheduler is not None:
                scheduler.step()

            if (epoch + 1) % save_epochs == 0:
                torch.save(model.state_dict(), checkpoints_dir / f"{epoch + 1}.pt")
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.state_dict(), output_dir / "best.pt")
            (output_dir / "stats.json").write_text(json.dumps(history, indent=2))
            temporary_state = state_path.with_suffix(".pt.tmp")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "epoch": epoch + 1,
                "best_loss": best_loss,
                "history": history,
                "torch_rng_state": torch.get_rng_state(),
            }, temporary_state)
            temporary_state.replace(state_path)

        torch.save(model.state_dict(), output_dir / "final.pt")
        return history
