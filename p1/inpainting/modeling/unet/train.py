import argparse
import json
from datetime import datetime
from pathlib import Path
from random import Random
from time import perf_counter

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torchmetrics.functional import mean_absolute_error
from torchmetrics.functional.image import peak_signal_noise_ratio
from torchvision.utils import save_image
from tqdm import tqdm

from p1.inpainting.evaluate import run_evaluation


@torch.inference_mode()
def validate(model, loader, device, loss_fn, examples_path=None):
    model.eval()
    psnr, mae, examples = [], [], []
    loss_sum, sample_count = 0.0, 0
    for cut_images, masks, targets in loader:
        cut_images, masks, targets = (tensor.to(device) for tensor in (cut_images, masks, targets))
        predictions = model(torch.cat([cut_images, masks], dim=1))
        loss_sum += loss_fn(predictions, targets, masks).item() * len(targets)
        sample_count += len(targets)
        for cut_image, prediction, target, mask in zip(cut_images, predictions, targets, masks):
            hole = mask.bool().expand_as(target)
            psnr.append(peak_signal_noise_ratio(prediction[hole], target[hole], data_range=1.0).item())
            mae.append(mean_absolute_error(prediction[hole], target[hole]).item())
            if examples_path is not None and len(examples) < 15:
                restored = cut_image * (1 - mask) + prediction * mask
                examples.extend([cut_image.cpu(), restored.cpu(), target.cpu()])

    scores = {"PSNR_hole": sum(psnr) / len(psnr), "MAE_hole": sum(mae) / len(mae)}
    scores["loss"] = loss_sum / sample_count
    if examples_path is not None:
        save_image(torch.stack(examples), examples_path, nrow=3)
    return scores


def train(config, config_name="unet"):
    torch.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = config.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)
    output_dir = Path(config.output_dir or f"output/{datetime.now():%Y-%m-%d_%H-%M-%S}-{config_name}")
    config.output_dir = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "config.yaml")

    train_dataset = instantiate(
        config.dataset, num_samples=config.steps * config.loader.batch_size, augment=True,
    )
    image_paths = train_dataset.image_paths.copy()
    Random(config.seed).shuffle(image_paths)
    validation_paths = image_paths[:config.validation_images]
    train_paths = image_paths[config.validation_images:]
    if not train_paths or not validation_paths:
        raise ValueError("The image split must contain both training and validation images")
    (output_dir / "split.json").write_text(json.dumps({
        "train": [str(path) for path in train_paths],
        "validation": [str(path) for path in validation_paths],
    }, indent=2))

    train_dataset.image_paths = train_paths
    validation_dataset = instantiate(
        config.dataset, num_samples=len(validation_paths) * config.masks_per_image, random_images=False,
    )
    validation_dataset.image_paths = validation_paths
    loader_options = {
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.loader.num_workers > 0,
    }
    train_loader = instantiate(config.loader, dataset=train_dataset, **loader_options)
    validation_loader = instantiate(config.loader, dataset=validation_dataset, **loader_options)

    model = instantiate(config.model).to(device)
    loss_fn = instantiate(config.loss).to(device).eval()
    optimizer = instantiate(config.optimizer, params=model.parameters())
    scheduler = instantiate(config.scheduler, optimizer=optimizer)
    mixed_precision = device.type == "cuda" and config.precision != "fp32"
    use_bf16 = config.precision == "bf16" or (config.precision == "auto" and torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision and amp_dtype == torch.float16)
    print(f"Training on {device}, precision: {str(amp_dtype) if mixed_precision else 'float32'}")

    best_psnr = float("-inf")
    losses = []
    metrics = []
    start_time = perf_counter()
    for step, (cut_images, masks, targets) in enumerate(tqdm(train_loader, total=config.steps), start=1):
        model.train()
        cut_images, masks, targets = (tensor.to(device) for tensor in (cut_images, masks, targets))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=mixed_precision):
            predictions = model(torch.cat([cut_images, masks], dim=1))
            loss = loss_fn(predictions, targets, masks)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        losses.append(loss.item())

        if step % config.validation_every == 0 or step == config.steps:
            scores = validate(model, validation_loader, device, loss_fn, output_dir / "examples.png")
            train_loss = sum(losses) / len(losses)
            metrics.append({
                "step": step, "elapsed_seconds": perf_counter() - start_time,
                "train_loss": train_loss, "validation_loss": scores["loss"],
                "lr": optimizer.param_groups[0]["lr"],
                "PSNR_hole": scores["PSNR_hole"], "MAE_hole": scores["MAE_hole"],
            })
            (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            tqdm.write(
                f"[{step}/{config.steps}] Train loss: {train_loss:.4f}, "
                f"Validation loss: {scores['loss']:.4f}, "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
            tqdm.write(
                f"[{step}/{config.steps}] Validation hole PSNR: {scores['PSNR_hole']:.2f}, "
                f"hole MAE: {scores['MAE_hole']:.4f}"
            )
            checkpoint = {
                "model": model.state_dict(), "step": step, "validation": scores,
                "config": OmegaConf.to_container(config, resolve=True),
            }
            if scores["PSNR_hole"] > best_psnr:
                best_psnr = scores["PSNR_hole"]
                torch.save(checkpoint, output_dir / "best.pt")
            if step == config.steps:
                torch.save(checkpoint, output_dir / "last.pt")
            losses.clear()

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    run_evaluation(
        data_dir=config.dataset.root, output_dir=output_dir / "evaluation",
        seed=config.seed, repeats=1, device=str(device), model=model,
    )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("p1/configs/unet_p0_1.yaml"))
    args = parser.parse_args()
    with initialize_config_dir(version_base=None, config_dir=str(args.config.resolve().parent)):
        config = compose(config_name=args.config.stem)
    train(config, config_name=args.config.stem)
