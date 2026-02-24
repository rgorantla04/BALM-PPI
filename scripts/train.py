import argparse
import os
import sys
from typing import Optional

sys.path.append(os.getcwd())
from dotenv import load_dotenv
load_dotenv(".env")

from balm import common_utils
from balm.configs import Configs
from balm.trainer import Trainer

def argument_parser() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Protein-Protein Interaction Training"
    )
    parser.add_argument(
        "--config_filepath",
        type=str,
        required=True,
        help="Path to the YAML configuration file"
    )
    return parser.parse_args()

def main() -> None:
    """
    Main training function.
    Handles configuration loading, experiment setup, and training execution.
    """
    # Parse arguments and load config
    args = argument_parser()
    configs = Configs(**common_utils.load_yaml(args.config_filepath))

    # Set random seed for reproducibility
    common_utils.setup_random_seed(configs.training_configs.random_seed)
    
    # Setup experiment directories
    outputs_dir = common_utils.setup_experiment_folder(
        os.path.join(os.getcwd(), configs.training_configs.outputs_dir)
    )
    
    # Save configurations
    common_utils.save_training_configs(configs, outputs_dir)

    # Get wandb credentials
    wandb_entity = os.getenv("WANDB_ENTITY", "")
    wandb_project = os.getenv("WANDB_PROJECT_NAME", "")

    # Initialize trainer
    trainer = Trainer(configs, wandb_entity, wandb_project, outputs_dir)
    
    # Setup dataset
    trainer.set_dataset(train_ratio=configs.dataset_configs.train_ratio)
    
    # Initialize training environment
    trainer.setup_training()
    
    # Execute training
    trainer.train()

if __name__ == "__main__":
    main()