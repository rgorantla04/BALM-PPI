import os
from typing import Union
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.tracking import WandBTracker
from datasets import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import wandb
import esm

from balm.configs import Configs
from balm.datasets import create_scaffold_split_dti
from balm.datasets.utils import DataCollatorWithPadding, get_dataset_split
from balm.models import BaselineModel, BALM
from balm.models.utils import load_trained_model, load_pretrained_pkd_bounds
from balm.metrics import get_ci, get_pearson, get_rmse, get_spearman
from balm.tokenization import pre_tokenize_unique_entities, tokenize_with_lookup
from balm import factories
from balm.datasets.utils import ESMDataCollator
from transformers import AutoTokenizer  # Remove this
from balm.tokenization import pre_tokenize_unique_entities, tokenize_with_lookup  # Remove these
from peft import get_peft_model, PeftConfig

class Trainer:
    """
    The Trainer class handles the training, validation, and testing processes for the models.
    It supports setting up datasets, initializing models, and managing the training loop with
    early stopping and learning rate scheduling.

    Attributes:
        configs (Configs): Configuration object with all necessary hyperparameters and settings.
        wandb_entity (str): Weights & Biases entity name.
        wandb_project (str): Weights & Biases project name.
        outputs_dir (str): Directory where output files such as checkpoints and logs are saved.
    """

    def __init__(
        self, configs: Configs, wandb_entity: str, wandb_project: str, outputs_dir: str
    ):
        """
        Initialize the Trainer with the provided configurations, Weights & Biases settings, 
        and output directory.

        Args:
            configs (Configs): Configuration object.
            wandb_entity (str): Weights & Biases entity name.
            wandb_project (str): Weights & Biases project name.
            outputs_dir (str): Directory where outputs are saved.
        """
        self.configs = configs

        self.dataset_configs = self.configs.dataset_configs
        self.training_configs = self.configs.training_configs
        self.model_configs = self.configs.model_configs

        self.gradient_accumulation_steps = (
            self.model_configs.model_hyperparameters.gradient_accumulation_steps
        )
        self.protein_max_seq_len = (
            self.model_configs.model_hyperparameters.protein_max_seq_len
        )
        self.proteina_max_seq_len = (
            self.model_configs.model_hyperparameters.proteina_max_seq_len
        )

        self.outputs_dir = outputs_dir
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project

        # Load ESM-2 model and alphabet instead of tokenizers
        self.model = self._initialize_model(configs)
        self.batch_converter = self.alphabet.get_batch_converter()
        # Determine which model to use based on fine-tuning type
        if configs.model_configs.fine_tuning_method:
            self.model = BALM(configs.model_configs)  # BALM handles PEFT
        elif (configs.model_configs.protein_fine_tuning_type == "baseline"
              and configs.model_configs.proteina_fine_tuning_type == "baseline"):
            self.model = BaselineModel(configs.model_configs)
        else:
            self.model = BALM(configs.model_configs)

        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader = None

        self._setup_run_name()
    def _initialize_model(self, configs):
        if configs.model_configs.fine_tuning_method:
            model = BALM(configs.model_configs)  # BALM handles PEFT
        elif (configs.model_configs.protein_fine_tuning_type == "baseline"
              and configs.model_configs.proteina_fine_tuning_type == "baseline"):
            model = BaselineModel(configs.model_configs)
        else:
            model = BALM(configs.model_configs)
        return model
    
    def _load_esm_model(self):
        """Load the ESM-2 model and alphabet"""
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model = model.eval().cuda()
        return model, alphabet

    def set_pkd_bounds(self, dataset):
        """
        Set the pKd bounds for scaling the labels in the dataset. If a checkpoint is loaded 
        for a zero-shot experiment, the bounds are loaded from the checkpoint.

        Args:
            dataset (Dataset): The dataset containing the pKd labels.
        """
        self.pkd_lower_bound = min(dataset.y)
        self.pkd_upper_bound = max(dataset.y)

        # Load pKd bounds from a trained model if performing a zero-shot experiment
        if self.model_configs.checkpoint_path:
            if self.dataset_configs.train_ratio == 0.0:
                self.pkd_lower_bound, self.pkd_upper_bound = load_pretrained_pkd_bounds(self.model_configs.checkpoint_path)
        
        print(
            f"Scaling labels: from {self.pkd_lower_bound} - {self.pkd_upper_bound} to -1 to 1"
        )

    def set_dataset(self, *args, **kwargs) -> dict:
        """
        Prepare and set up the dataset for training, validation, and testing. This includes
        pre-tokenization, filtering based on sequence length, and setting up DataLoaders.

        Returns:
            dict: Dictionary containing the dataset splits (train, valid, test).
        """
        dataset = factories.get_dataset(self.dataset_configs.dataset_name, *args, **kwargs)

        print(
            f"Training with {self.model_configs.loss_function} loss function."
        )

        # Apply pKd scaling if using cosine MSE loss
        if self.model_configs.loss_function == "cosine_mse":
            self.set_pkd_bounds(dataset)

            if self.pkd_upper_bound == self.pkd_lower_bound:
                # Handle case where all labels are the same
                dataset.y = [0 for _ in dataset.y]
            else:
                dataset.y = [
                    (pkd - self.pkd_lower_bound)
                    / (self.pkd_upper_bound - self.pkd_lower_bound)
                    * 2
                    - 1
                    for pkd in dataset.y
                ]
            # Preprocess Y column for non-TDC datasets using the same pKd scaling
            if not self.dataset_configs.dataset_name.startswith("DTI_"):
                if self.pkd_upper_bound == self.pkd_lower_bound:
                    dataset.data["Y"] = dataset.data["Y"].apply(lambda x: 0)
                else:
                    dataset.data["Y"] = dataset.data["Y"].apply(
                        lambda x: (x - self.pkd_lower_bound)
                        / (self.pkd_upper_bound - self.pkd_lower_bound)
                        * 2
                        - 1
                    )
        elif self.model_configs.loss_function in ["baseline_mse"]:
            print("Using original pKd")

        # Filter the dataset by sequence length
        print("Filtering dataset by length")
        print(f"Protein max length: {self.protein_max_seq_len}")
        print(f"proteina max length: {self.proteina_max_seq_len}")

        dataset_splits = {}
        for split, data_df in get_dataset_split(self.dataset_configs, self.training_configs, dataset).items():
            if data_df is None:
                continue
  

             # Convert directly to Dataset without pre-tokenization
            dataset = Dataset.from_pandas(data_df)

            # Filter by sequence length
            num_original_dataset = len(dataset)
            dataset = dataset.filter(
                lambda example: len(example["protein"]) <= self.protein_max_seq_len
                and len(example["proteina"]) <= self.proteina_max_seq_len
            )
            num_filtered_dataset = len(dataset)
            print(
                f"Number of filtered pairs: "
                f"{num_filtered_dataset}/{num_original_dataset} "
                f"({float(num_filtered_dataset)/num_original_dataset*100:.2f}%)"
            )
            dataset_splits[split] = dataset
        
        # Create new data collator for ESM-2
        data_collator = ESMDataCollator(
            batch_converter=self.batch_converter,
            protein_max_length=self.protein_max_seq_len,
            proteina_max_length=self.proteina_max_seq_len,
        )

        # Setup DataLoaders for train, valid, and test splits
        if "train" in dataset_splits:
            print(f"Setup Train DataLoader")
            self.train_dataloader = DataLoader(
                dataset_splits["train"],
                shuffle=True,
                collate_fn=data_collator,
                batch_size=self.training_configs.batch_size,
                pin_memory=True,
            )
        if "valid" in dataset_splits:
            print(f"Setup Valid DataLoader")
            self.valid_dataloader = DataLoader(
                dataset_splits["valid"],
                shuffle=False,
                collate_fn=data_collator,
                batch_size=self.training_configs.batch_size,
                pin_memory=True,
            )
        print(f"Setup Test DataLoader")
        self.test_dataloader = DataLoader(
            dataset_splits["test"],
            shuffle=False,
            collate_fn=data_collator,
            batch_size=self.training_configs.batch_size,
            pin_memory=True,
        )

    def _setup_run_name(self):
        """
        Setup the run name and group name for the Weights & Biases tracker.
        Handles both traditional fine-tuning and PEFT methods.
        """
        # Method abbreviations for cleaner names
        METHOD_ABBREV = {
            "lora": "lr",
            "loha": "lh",
            "lokr": "lk",
            "ia3": "ia"
        }

        # Group name remains the same
        self.group_name = f"{self.dataset_configs.dataset_name}_{self.dataset_configs.split_method}"

        # Collect hyperparameters for run name
        hyperparams = []

        # Handle protein model configuration
        protein_config = []
        if hasattr(self.model_configs, "protein_peft_config") and self.model_configs.protein_peft_config:
            config = self.model_configs.protein_peft_config
            method_abbr = METHOD_ABBREV.get(config.method, config.method)
            protein_config.append(f"prot_{method_abbr}")

            # Add relevant PEFT parameters
            if hasattr(config, "rank"):
                protein_config.append(f"r{config.rank}")
            if hasattr(config, "alpha"):
                protein_config.append(f"a{config.alpha}")
            if hasattr(config, "dropout"):
                protein_config.append(f"d{config.dropout}")
        else:
            # Traditional fine-tuning type
            protein_config.append(f"prot_{self.model_configs.protein_fine_tuning_type}")

        hyperparams.extend(protein_config)

        # Handle proteina model configuration
        proteina_config = []
        if hasattr(self.model_configs, "proteina_peft_config") and self.model_configs.proteina_peft_config:
            config = self.model_configs.proteina_peft_config
            method_abbr = METHOD_ABBREV.get(config.method, config.method)
            proteina_config.append(f"prota_{method_abbr}")

            # Add relevant PEFT parameters
            if hasattr(config, "rank"):
                proteina_config.append(f"r{config.rank}")
            if hasattr(config, "alpha"):
                proteina_config.append(f"a{config.alpha}")
            if hasattr(config, "dropout"):
                proteina_config.append(f"d{config.dropout}")
        else:
            # Traditional fine-tuning type
            proteina_config.append(f"prota_{self.model_configs.proteina_fine_tuning_type}")

        hyperparams.extend(proteina_config)

        # Join all parameters with underscores
        self.run_name = "_".join(str(param) for param in hyperparams)

    def setup_training(self):
        """
        Setup the training environment, including initializing the Accelerator, WandB tracker, 
        optimizer, and learning rate scheduler. Prepares the model and dataloaders for training.
        """
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            log_with="wandb",
        )
        self.wandb_tracker = None
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=self.wandb_project,
                init_kwargs={
                    "wandb": {
                        "entity": self.wandb_entity,
                        "name": self.run_name,
                        "group": self.group_name,
                    }
                },
                config=self.configs.dict(),
            )
            self.wandb_tracker: WandBTracker = self.accelerator.get_tracker("wandb")
        self.accelerator.wait_for_everyone()

        if self.train_dataloader is not None:
            param_groups = []
            # PEFT parameters if present
            peft_params = [p for n, p in self.model.named_parameters() if 'peft' in n]
            if peft_params:
                param_groups.append({
                    'params': peft_params,
                    'lr': self.model_configs.model_hyperparameters.learning_rate * 10
                })
            # Projection parameters
            proj_params = [p for n, p in self.model.named_parameters() 
                          if 'projection' in n and p.requires_grad]
            if proj_params:
                param_groups.append({
                    'params': proj_params,
                    'lr': self.model_configs.model_hyperparameters.learning_rate
                })
            self.optimizer = AdamW(param_groups)

            
            # Setup learning rate scheduler
            num_training_steps = (
                len(self.train_dataloader) * self.training_configs.epochs
            )
            warmup_steps_ratio = (
                self.model_configs.model_hyperparameters.warmup_steps_ratio
            )
            self.lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=warmup_steps_ratio,
                num_training_steps=num_training_steps,
            )

            # Prepare model, dataloaders, optimizer, and scheduler for training
            (
                self.model,
                self.train_dataloader,
                self.valid_dataloader,
                self.test_dataloader,
                self.optimizer,
                self.lr_scheduler,
            ) = self.accelerator.prepare(
                self.model,
                self.train_dataloader,
                self.valid_dataloader,
                self.test_dataloader,
                self.optimizer,
                self.lr_scheduler,
            )
        else:
            # If only testing, prepare the model and test dataloader
            (
                self.model,
                self.test_dataloader,
            ) = self.accelerator.prepare(
                self.model,
                self.test_dataloader,
            )

        # Load a trained model from checkpoint if specified
        if self.model_configs.checkpoint_path:
            load_trained_model(self.model, self.model_configs, is_training=self.train_dataloader is not None)

    def compute_metrics(self, labels, predictions):
        """
        Compute evaluation metrics including RMSE, Pearson, Spearman, and CI.

        Args:
            labels (Tensor): True labels.
            predictions (Tensor): Predicted values.

        Returns:
            dict: Dictionary containing the computed metrics.
        """
        if self.model_configs.loss_function in [
            "cosine_mse"
        ]:
            # Rescale predictions and labels back to the original pKd range
            pkd_range = self.pkd_upper_bound - self.pkd_lower_bound
            labels = (labels + 1) / 2 * pkd_range + self.pkd_lower_bound
            predictions = (predictions + 1) / 2 * pkd_range + self.pkd_lower_bound

        rmse = get_rmse(labels, predictions)
        pearson = get_pearson(labels, predictions)
        spearman = get_spearman(labels, predictions)
        ci = get_ci(labels, predictions)

        return {
            "rmse": rmse,
            "pearson": pearson,
            "spearman": spearman,
            "ci": ci,
        }

    def train(self):
        """Execute the training loop with PEFT support"""
        if self.train_dataloader is None:
            epoch = 0
            best_checkpoint_dir = None
        else:
            best_loss = float('inf')
            patience = self.training_configs.patience
            eval_train_every_n_epochs = self.training_configs.epochs // 4
            epochs_no_improve = 0
            best_checkpoint_dir = ""

            # Enhanced parameter reporting
            print("Parameter groups:")
            param_counts = {
                'peft': 0,
                'projection': 0,
                'base': 0
            }
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if 'peft' in name:
                    param_counts['peft'] += param.numel()
                elif 'projection' in name:
                    param_counts['projection'] += param.numel()
                else:
                    param_counts['base'] += param.numel()
            for group, count in param_counts.items():
                print(f"{group}: {count:,} parameters")

            for epoch in range(self.training_configs.epochs):
                self.model.train()
                total_train_loss = 0
                total_samples = 0

                # Progress bar with enhanced stats
                progress_bar = tqdm(
                    total=int(len(self.train_dataloader) // self.gradient_accumulation_steps),
                    position=0,
                    leave=True,
                    disable=not self.accelerator.is_local_main_process,
                )

                for train_step, batch in enumerate(self.train_dataloader):
                    with self.accelerator.accumulate(self.model):
                        outputs = self.model(batch)
                        loss = outputs["loss"]
                        self.accelerator.backward(loss)

                        # Track gradient norms for debugging
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self.model.parameters(), 
                            max_norm=1.0
                    )

                    self.optimizer.step()
                    self.model.zero_grad()

                    # Update progress with enhanced info
                    progress_bar.set_description(
                        f"Epoch {epoch}; Loss: {loss:.4f}; Grad: {grad_norm:.4f}"
                    )
                    total_train_loss += loss.detach().float()
                    total_samples += 1

                    if self.accelerator.sync_gradients:
                        progress_bar.update(1)

                # Proper scheduler stepping (once per epoch)
                self.lr_scheduler.step()

                # Enhanced metric logging
                self.log_training_stats(
                    epoch=epoch,
                    loss=total_train_loss/total_samples,
                    grad_norm=grad_norm,
                    param_counts=param_counts
                )

                # Evaluation
                if (epoch + 1) % eval_train_every_n_epochs == 0:
                    train_metrics = self.test("train")
                else:
                    train_metrics = {
                        "train/loss": total_train_loss / len(self.train_dataloader)
                    }
                valid_metrics = self.test("valid")

                # Checkpoint handling with PEFT support
                if valid_metrics:
                    current_loss = valid_metrics["valid/loss"]
                else:
                    current_loss = best_loss

                if current_loss <= best_loss:
                    best_loss = current_loss
                    epochs_no_improve = 0
                    best_checkpoint_dir = f"step_{epoch}"
                    # Use enhanced checkpoint saving
                    self.save_checkpoint(
                        os.path.join(self.outputs_dir, "checkpoint", best_checkpoint_dir),
                        epoch
                    )
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

            # Load best checkpoint with PEFT support
            if best_checkpoint_dir:
                checkpoint_path = os.path.join(
                    self.outputs_dir, "checkpoint", best_checkpoint_dir
                )
                self.load_checkpoint(checkpoint_path)
                self.accelerator.wait_for_everyone()

        # Test evaluation and artifact creation
        test_metrics = self.test("test", save_prediction=True)
        self.accelerator.log(test_metrics, step=epoch)

        if self.dataset_configs.dataset_name == "BindingDB_filtered":
            for split in ["train", "valid"]:
                metrics = self.test(split, save_prediction=True)
                self.accelerator.log(metrics, step=epoch)

        # Enhanced artifact saving with PEFT config
        if best_checkpoint_dir:
            artifact = wandb.Artifact(best_checkpoint_dir, type="model")
            artifact.add_dir(
                os.path.join(self.outputs_dir, "checkpoint", best_checkpoint_dir)
            )
            if hasattr(self.model_configs, 'peft_config'):
                artifact.metadata = {'peft_config': self.model_configs.peft_config.dict()}
            wandb.log_artifact(artifact)

    def test(self, split: str, save_prediction=False):
        """
        Evaluate the model on the specified dataset split and optionally save predictions.

        Args:
            split (str): The dataset split to evaluate on ('train', 'valid', 'test').
            save_prediction (bool): Whether to save the predictions as a CSV file.

        Returns:
            dict: Dictionary containing the evaluation metrics for the specified split.
        """
        if split == "train":
            dataloader = self.train_dataloader
        elif split == "valid":
            dataloader = self.valid_dataloader
        elif split == "test":
            dataloader = self.test_dataloader

        if dataloader is None:
            return {}

        total_loss = 0
        all_proteins = []
        all_proteina = []
        all_labels = []
        all_predictions = []

        self.model.eval()

        num_steps = len(dataloader)
        progress_bar = tqdm(
            total=num_steps,
            position=0,
            leave=True,
            disable=not self.accelerator.is_local_main_process,
        )

    
        for step, batch in enumerate(tqdm(dataloader)):
            with torch.no_grad():
                # Process sequences with ESM-2
                outputs = self.model(batch)
                loss = outputs["loss"]
                total_loss += loss.detach().float()
                all_proteins += batch["protein_sequences"]
                all_proteina += batch["proteina_sequences"]
                all_labels += [batch["labels"]]
                all_predictions += [outputs["logits" if self.model_configs.loss_function == "baseline_mse" else "cosine_similarity"]]
            progress_bar.set_description(f"Eval: {split} split")
            progress_bar.update(1)

        # Concatenate all predictions and labels across batches
        all_labels = torch.cat(all_labels, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        performance_metrics = self.compute_metrics(all_labels, all_predictions)
        metrics = {
            f"{split}/loss": total_loss / len(dataloader),
        }
        for metric_name, metric_value in performance_metrics.items():
            metrics[f"{split}/{metric_name}"] = metric_value

        if save_prediction:
            # Save predictions and labels to a CSV file
            df = pd.DataFrame(columns=["protein", "proteina", "label", "prediction"])
            df["protein"] = all_proteins
            df["proteina"] = all_proteina

            if self.model_configs.loss_function in [
                "cosine_mse"
            ]:
                pkd_range = self.pkd_upper_bound - self.pkd_lower_bound
                all_labels = (all_labels + 1) / 2 * pkd_range + self.pkd_lower_bound
                all_predictions = (
                    all_predictions + 1
                ) / 2 * pkd_range + self.pkd_lower_bound

            df["label"] = all_labels.cpu().numpy().tolist()
            df["prediction"] = all_predictions.cpu().numpy().tolist()
            df.to_csv(os.path.join(self.outputs_dir, f"{split}_prediction.csv"))

            # Log the predictions as a WandB artifact
            artifact = wandb.Artifact(f"{split}_prediction", type="prediction")
            artifact.add_file(os.path.join(self.outputs_dir, f"{split}_prediction.csv"))
            wandb.log_artifact(artifact)

        return metrics
    def save_checkpoint(self, dir_path, epoch):
        """Save model checkpoint with PEFT support"""
        # Save PEFT state if applicable
        if hasattr(self.model, 'save_peft_state'):
            peft_path = os.path.join(dir_path, 'peft_state')
            self.model.save_peft_state(peft_path)

        # Save complete state
        state_path = os.path.join(dir_path, f'checkpoint_{epoch}')
        self.accelerator.save_state(state_path)
        return state_path
    
    def log_training_stats(self, epoch, loss):
        stats = {
            'train/loss': loss,
            'train/lr': self.optimizer.param_groups[0]['lr'],
        }
        
        # Add PEFT-specific stats
        if hasattr(self.model, 'get_peft_state_size'):
            peft_stats = self.model.get_peft_state_size()
            stats.update(peft_stats)
        
        self.accelerator.log(stats, step=epoch)
