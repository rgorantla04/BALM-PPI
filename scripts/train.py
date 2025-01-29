import argparse
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv(".env")

from balm import common_utils
from balm.configs import Configs
from balm.trainer import Trainer
import esm  # Add ESM import


def argument_parser():
    parser = argparse.ArgumentParser(description="Protein-Protein Interaction using ESM-2")
    parser.add_argument("--config_filepath", type=str, required=True)
    # Add ESM-specific arguments if needed
    parser.add_argument("--esm_model", type=str, default="esm2_t33_650M_UR50D",
                       help="ESM-2 model variant to use")
    parser.add_argument("--layer_num", type=int, default=33,
                       help="Layer number to extract embeddings from")
    args = parser.parse_args()
    return args


def main() -> None:
    args = argument_parser()
    configs = Configs(**common_utils.load_yaml(args.config_filepath))

    # Set the random seed for reproducibility
    common_utils.setup_random_seed(configs.training_configs.random_seed)
    
    # Setup the output directory for the experiment
    outputs_dir = common_utils.setup_experiment_folder(
        os.path.join(os.getcwd(), configs.training_configs.outputs_dir)
    )
    
    # Save the training configuration to the output directory
    common_utils.save_training_configs(configs, outputs_dir)

    # Login to Weights & Biases (WandB)
    wandb_entity = os.getenv("WANDB_ENTITY", "")
    wandb_project = os.getenv("WANDB_PROJECT_NAME", "")

    # Load ESM-2 model
    model, alphabet = esm.pretrained.load_model_and_alphabet(args.esm_model)
    model = model.eval().cuda()
    batch_converter = alphabet.get_batch_converter()
    
    # Add ESM model and converter to configs
    configs.model_configs.esm_model = model
    configs.model_configs.batch_converter = batch_converter
    configs.model_configs.layer_num = args.layer_num

    # Initialize the Trainer and start training
    trainer = Trainer(configs, wandb_entity, wandb_project, outputs_dir)
    
    # Set the dataset with an optional train ratio (specific to Mpro)
    trainer.set_dataset(train_ratio=configs.dataset_configs.train_ratio)
    
    # Setup the training environment
    trainer.setup_training()
    
    # Start the training loop
    trainer.train()


if __name__ == "__main__":
    main()