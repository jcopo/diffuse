# Standard library imports
import argparse
from pathlib import Path

# Third-party imports
from envyaml import EnvYAML

# Local imports
from ..datasets import get_dataloader
from ..training import LatentModelTrainer, ScoreModelTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, default="autoencoder")
    args = parser.parse_args()

    config = EnvYAML(args.config)

    # Setup directories
    model_dir = Path(config["model_dir"])
    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        config_src = Path(args.config)
        config_dst = model_dir / f"config_{config['dataset']}.yaml"
        config_dst.write_bytes(config_src.read_bytes())

    # Get data loaders
    train_val_loaders = get_dataloader(config, model=args.model)

    # Initialize and run appropriate trainer
    if args.model == "autoencoder":
        trainer = LatentModelTrainer(config)
    else:
        trainer = ScoreModelTrainer(config)

    trainer.train(train_val_loaders)


if __name__ == "__main__":
    main()
