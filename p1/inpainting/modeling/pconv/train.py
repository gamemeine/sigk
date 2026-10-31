import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from p1.inpainting.modeling.trainer import InpaintingTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Hydra YAML config")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--resume", type=Path, help="Resume from an existing output directory")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate best.pt after training in the same output directory")
    args = parser.parse_args()

    config_path = args.resume / "config.yaml" if args.resume else args.config

    with initialize_config_dir(config_dir=str(config_path.parent.resolve()), version_base=None):
        config = compose(config_name=config_path.name)

    output_dir = (
        args.resume
        or args.output_dir
        or (Path("p1/output") / config.training.experiment_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, str(output_dir / "config.yaml"), resolve=True)
    run_config = {
        **OmegaConf.to_container(config, resolve=True),
        "config": config_path,
        "output_dir": output_dir,
        "resume": args.resume,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str))

    training = OmegaConf.to_container(config.training, resolve=True)
    device = training.pop("device")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    training.pop("experiment_name")
    torch.manual_seed(training["seed"])

    model = instantiate(config.model, _convert_="all")
    criterion = instantiate(config.loss, _convert_="all")
    dataset = instantiate(config.data, _convert_="all")

    trainer = InpaintingTrainer(model, criterion, device=device)
    trainer.train(dataset, output_dir, resume=args.resume is not None, **training)
    if args.evaluate:
        from p1.inpainting.evaluate import run_evaluation

        run_evaluation(
            data_dir=config.data.root,
            models=[output_dir],
            output_dir=output_dir,
            device=device,
            seed=config.training.seed,
            repeats=5,
        )
