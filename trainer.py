import torch
from torch import nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import List
import comet_ml
import time
import os

from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer,
        train_dataloader: torch.utils.data.dataloader,
        num_classes: int,
        lr: float,
        validation_dataloader: torch.utils.data.dataloader = None,
        enable_checkpoint=False,
        checkpoint_dir="./checkpoint",
        limit_on_train_batch=None,
        enable_comet=False,
        comet_project=None,
        comet_experiment_name=None,
        cfg: dict = None,
        validation_data: List[str] = None,
        tokenizer=None,
    ):
        """
        Initialize the Trainer object.

        Parameters
        ----------
        model (torch.nn.Module): The neural network model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model parameters.
        train_dataloader (torch.utils.data.DataLoader): The PyTorch data loader for training data.
        num_classes (int): The number of classes in the final layer.
        lr (float): The learning rate for the optimizer (for logging purpose)
        validation_dataloader (Optional[torch.utils.data.DataLoader], optional): The PyTorch data loader for validation data.
        enable_checkpoint (bool, optional): Whether to enable checkpointing for model saving. Defaults to False.
        checkpoint_dir (str, optional): The directory to save checkpoints to. Defaults to "./checkpoint".
        limit_train_batch (Optional[int], optional): The maximum batch size allowed for training. If set, the training dataloader will be truncated if the batch size exceeds this limit. Defaults to None.

        """

        self.model = model
        self.model.to(device)
        self.optimizer = optimizer
        self.checkpoint_dir = Path(checkpoint_dir)
        self.num_classes = num_classes
        self.enable_checkpoint = enable_checkpoint
        self.lr = lr
        self.bs = train_dataloader.batch_size
        self.limit_on_train_batch = limit_on_train_batch
        self.enable_comet = enable_comet
        self.comet_project = comet_project
        self.comet_experiment_name = comet_experiment_name
        self.cfg = cfg
        self.comet_experiment_key = None
        self.tokenizer = tokenizer

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.validation_data = validation_data

        self.optimizer = optimizer
        self.history = {
            "training_loss": [],
            "validation_loss": [],
            "validation_metric": [],
        }
        # last successful epoch, if training is interrupted
        self.epoch = 0
        self.best_metric = float("-inf")
        self.should_checkpoint = False

    def _init_comet(self):
        common_parameters = {
            "auto_metric_logging": False,
            "log_git_metadata": False,
            "log_git_patch": False,
        }

        if not self.comet_experiment_key:
            self.comet_experiment = comet_ml.Experiment(
                "SmWWy3e7PsFONwITl0Snimwvg",
                project_name=self.comet_project,
                **common_parameters,
            )

            self.comet_experiment.set_name(self.comet_experiment_name)
            self.comet_experiment_key = self.comet_experiment.get_key()

        else:
            self.comet_experiment = comet_ml.ExistingExperiment(
                experiment_key=self.comet_experiment_key, **common_parameters
            )

        if self.cfg:
            self.comet_experiment.log_parameters(self.cfg)

    def _step(self, batch_idx, batch):
        src, tgt, _, tgt_mask = batch

        # tgt <eos> should not be a prompt
        logits, scores = self.model(src, tgt[:, :-1])

        logits = torch.cat(logits, dim=0).view(logits[0].size(0), -1, logits[0].size(1))

        # tgt <sos> should not be considered for loss because we have never predicted that
        loss = F.cross_entropy(logits.transpose(1, 2), tgt[:, 1:], reduction="none")
        loss = loss.masked_select(tgt_mask[:, 1:]).mean()

        return loss, logits

    def _bleu(self, token_ids, references):
        hypotheses = self.tokenizer.decode(token_ids, keep_specials=False)
        return corpus_bleu(references, hypotheses)

    def _validation(self):
        if not self.validation_dataloader:
            return

        self.model.eval()
        validation_loss_across_batches = []
        validation_metric_across_batches = []
        valid_batch_size = self.validation_dataloader.batch_size

        with torch.no_grad():
            i = 0
            bar = tqdm(self.validation_dataloader, leave=False)
            for batch in bar:
                # calculate loss on valid
                loss, logits = self._step(i, batch)
                validation_loss_across_batches.append(loss.item())

                # generate tokens from valid for bleu
                token_ids, _ = self.model.generate(
                    batch[0].to(device), max_seq=self.cfg["tgt_max_seq"]
                )
                validation_metric_across_batches.append(
                    self._bleu(
                        token_ids, self.validation_data[i : i + valid_batch_size]
                    )
                )
                i += valid_batch_size
                del batch

            bar.close()

        # this line is executed updated for each epoch
        self.history["validation_loss"].append(mean(validation_loss_across_batches))
        self.history["validation_metric"].append(mean(validation_metric_across_batches))

        if self.history["validation_metric"][-1] > self.best_metric:
            self.best_metric = self.history["validation_metric"][-1]
            self.should_checkpoint = True

    def fit(self, num_epoch: int, print_interval=1):
        """
        Trains the model for the specified number of epochs.

        Parameters:
        ----------
        num_epoch: The number of epochs to train for. This is continuous from the last epoch.
            Meaning, if the last epoch was 5, and `num_epoch` is 10, the model will be trained
            for epochs 6 to 15.
        print_interval: The interval for printing the training loss (default: 1).
        """
        self._init_comet()

        # main for-loop
        # each time `Trainer.fit()` is called, log will print
        # accumulated epoch counter
        for epoch in range(self.epoch + 1, self.epoch + num_epoch + 1):
            self.model.train()

            # print epoch duration
            if epoch % print_interval == 0:
                epoch_starts_at = time.time()

            loss_across_batches = []

            bar = tqdm(self.train_dataloader, unit="batch")

            # forward and backward
            for batch_no, batch in enumerate(bar):
                self.optimizer.zero_grad()

                loss, _ = self._step(-1, batch)

                loss.backward()
                self.optimizer.step()

                loss_across_batches.append(loss.item())

                # show each batch loss in tqdm bar
                bar.set_postfix(**{"loss": loss.item()})

                # skip training on the entire training dataset
                if batch_no == self.limit_on_train_batch:
                    break

                # free memory
                del batch

            # average batch losses
            self.history["training_loss"].append(mean(loss_across_batches))

            # calculate validation loss
            self._validation()

            self.epoch = epoch

            # log print
            if epoch % print_interval == 0:
                self._log(epoch_starts_at)

            if self.enable_checkpoint and self.should_checkpoint:
                self._save_checkpoint()

        self._close_comet()

    def _close_comet(self):
        if self.comet_experiment:
            self.comet_experiment.end()

    def _log(self, start_time):
        # generate console log statement
        time_elapsed = time.time() - start_time
        logstr = f"Epoch: {self.epoch},\t"
        logstr += f"Training loss: {self.history['training_loss'][-1]},\t"
        if self.validation_dataloader:
            logstr += f"Validation metric: {self.history['validation_metric'][-1]}\t"

        logstr += "Time elapsed: {:.2f} s".format(time_elapsed)
        print(logstr)

        # online logging
        if self.enable_comet:
            self.comet_experiment.log_metrics(
                {
                    "train/loss": self.history["training_loss"][-1],
                    "val/loss": self.history["validation_loss"][-1],
                    "val/bleu": self.history["validation_metric"][-1],
                },
                epoch=self.epoch,
            )

    def test(self, dataloader):
        """
        Tests the model on the specified dataloader.

        Parameters:
        ----------
        dataloader: The dataloader to test on.
        """

        bar = tqdm(dataloader)
        loss_across_batches = []
        metric_across_batches = []

        self.model.eval()
        with torch.no_grad():
            for batch in bar:
                loss, logits = self._step(-1, batch)

                loss_across_batches.append(loss.item())
                metric_across_batches.append(self._f1(logits, batch[1]).item())

                # show each batch loss in tqdm bar
                bar.set_postfix(
                    **{"loss": loss.item(), "metric": metric_across_batches[-1]}
                )
                del batch

        metric = mean(metric_across_batches)
        test_loss = mean(loss_across_batches)

        if self.enable_comet:
            self._init_comet()
            self.comet_experiment.log_others(
                {"test_f1_micro": metric, "test_loss": test_loss}
            )

        print(f"Loss: {test_loss},\tMetric: {metric}")

    def _get_model_path(self):
        return self.checkpoint_dir / f"model_lr{self.lr}_best.pt"

    def _save_checkpoint(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        path = self._get_model_path()

        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
                "lr": self.lr,
            },
            path,
        )

    def log_model(self, name=None):
        self._init_comet()
        self._save_checkpoint()
        if not name:
            name = f"model_epoch{self.epoch}"
        self.comet_experiment.log_model(name, self._get_model_path())
        self._close_comet()

    def load_from_checkpoint(self, checkpoint_path):
        """
        Loads a model from a checkpoint.

        Parameters:
        ----------
        checkpoint_path: The path to the checkpoint.

        Raises:
        ------
        Exception: If no checkpoint is found in the provided path.
        """
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epoch = checkpoint["epoch"]
            self.history = checkpoint["history"]
            self.lr = checkpoint["lr"]

            print(f"loaded existing model. epoch: {self.epoch}, lr: {self.lr}")
        else:
            raise Exception("No checkpoint found in the provided path")

    def plot_loss(self, slc=None):
        """
        Plots the training and validation loss.

        Parameters
        ----------
        slc (slice): Optional, if provided, uses this to slice the data
        """
        _x = list(range(1, self.epoch + 1))
        _trloss = self.history["training_loss"]
        _valloss = self.history["validation_loss"]

        # plot only a portion of the data if slc is provided
        # helps with visualization when the data has outliers
        if slc is not None:
            _x = _x[slc]
            _trloss = _trloss[slc]
            _valloss = _valloss[slc]

        plt.plot(_x, _trloss, label="Training loss")
        if self.validation_dataloader:
            plt.plot(_x, _valloss, label="Validation loss")

        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training metrics")
        plt.show()

    def plot_metric(self, slc):
        _x = list(range(1, self.epoch + 1))
        _val_metric = self.history["validation_metric"]

        # plot only a portion of the data if slc is provided
        # helps with visualization when the data has outliers
        if slc is not None:
            _x = _x[slc]
            _val_metric = _val_metric[slc]

        plt.plot(_x, _val_metric, label="Training loss")

        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Accuracy Metrics")
        plt.show()
