import json
from copy import copy
from pathlib import Path
from random import Random

import torch
from torch import nn
from torch.utils.data import DataLoader, get_worker_info
from tqdm import tqdm

from p1.inpainting.data import DIV2KDataset


def seed_worker(worker_id):
    get_worker_info().dataset.set_seed(torch.initial_seed())


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
    ):
        model, criterion, device = self.model, self.criterion, self.device
        train_ds, valid_ds = split_train_val(train_ds, val_size, seed)
        if not 1 <= batch_size <= len(train_ds):
            raise ValueError("batch_size must be at least 1 and no larger than the training dataset")

        torch.manual_seed(seed)
        rng = Random(seed)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        pin_memory = torch.device(device).type == "cuda"
        train_generator = torch.Generator()
        valid_generator = torch.Generator()
        loader_options = dict(
            batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=seed_worker,
        )
        train_loader = DataLoader(train_ds, generator=train_generator, **loader_options)
        valid_loader = DataLoader(valid_ds, generator=valid_generator, **loader_options)
        history = []
        best_loss = float("inf")
        start_epoch = 0
        state_path = output_dir / "state.pt"
        if resume:
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            history = state["history"]
            best_loss = state["best_loss"]
            start_epoch = state["epoch"]
            rng.setstate(state["random_state"])
            torch.set_rng_state(state["torch_rng_state"])
            print(f"Resuming after epoch {start_epoch}, target: {epochs} epochs.")
        else:
            torch.save(model.state_dict(), checkpoints_dir / "0.pt")

        for epoch in range(start_epoch, epochs):
            train_ds.set_seed(rng.randrange(2**32))
            train_generator.manual_seed(train_ds.seed)
            model.train()
            train_loss = torch.zeros((), device=device)
            train_count = 0
            progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for step, (cut, mask, target) in enumerate(progress, 1):
                cut, mask, target = (tensor.to(device, non_blocking=pin_memory) for tensor in (cut, mask, target))
                optimizer.zero_grad()
                prediction = model(cut, mask)
                loss = criterion(prediction, mask, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.detach() * cut.shape[0]
                train_count += cut.shape[0]
                if step % log_every == 0 or step == len(train_loader):
                    progress.set_postfix(loss=f"{loss.item():.4f}")

            model.eval()
            valid_ds.set_seed(seed)
            valid_generator.manual_seed(seed)
            valid_loss = torch.zeros((), device=device)
            hole_error = torch.zeros((), device=device)
            hole_values = torch.zeros((), device=device)
            with torch.no_grad():
                for cut, mask, target in valid_loader:
                    cut, mask, target = (tensor.to(device, non_blocking=pin_memory) for tensor in (cut, mask, target))
                    prediction = model(cut, mask)
                    valid_loss += criterion(prediction, mask, target) * cut.shape[0]
                    hole_error += ((prediction - target).abs() * mask).sum()
                    hole_values += mask.sum() * target.shape[1]
            train_loss, valid_loss, valid_hole_mae = torch.stack((
                train_loss / train_count, valid_loss / len(valid_ds), hole_error / hole_values,
            )).tolist()
            current_lr = optimizer.param_groups[0]["lr"]
            history.append({
                "epoch": epoch + 1, "train_loss": train_loss, "valid_loss": valid_loss,
                "valid_hole_mae": valid_hole_mae, "lr": current_lr,
            })
            print(
                f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}, "
                f"hole_mae={valid_hole_mae:.4f}, lr={current_lr:g}"
            )

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
                "epoch": epoch + 1,
                "best_loss": best_loss,
                "history": history,
                "random_state": rng.getstate(),
                "torch_rng_state": torch.get_rng_state(),
            }, temporary_state)
            temporary_state.replace(state_path)

        torch.save(model.state_dict(), output_dir / "final.pt")
        return history
