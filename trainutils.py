import json
import os
from datetime import datetime
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt  # drawing heat map of attention weights
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    import neptune
except ImportError:
    print("neptune not found")

device = "cuda" if torch.cuda.is_available() else "cpu"


class History:
    def __init__(
        self,
        enable_neptune,
        project_name=None,
        experiment_name=None,
        log_dir="./logs",
        api_token=None,
    ):
        self.parameters = {}
        if experiment_name is None:
            now = datetime.now()
            self.experiment_name = now.strftime("%Y-%m-%d_%H_%M_%S")
        else:
            self.experiment_name = experiment_name

        self.enable_neptune = enable_neptune

        if self.enable_neptune:
            api_token = (
                os.environ["NEPTUNE_API_TOKEN"] if api_token is None else api_token
            )

            self.run = neptune.init_run(
                project=project_name, api_token=api_token, name=experiment_name
            )

        self.data = {"train/loss": [], "val/loss": []}

        self.exp_dir = Path(log_dir) / self.experiment_name
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir, exist_ok=True)

    def log(self, key, val):
        if key in self.data:
            self.data[key].append(val)
        else:
            self.data[key] = [
                val,
            ]

        if self.enable_neptune:
            self.run[key].append(val)

    def hyperparameters(self, params: dict):
        self.parameters = params
        if self.enable_neptune:
            self.run["parameters"] = params

    def upload_checkpoint(self, path, name="model"):
        if self.enable_neptune:
            self.run[f"checkpoints/{name}"].upload(path)
        else:
            print("checkpoint wasn't uploaded to neptune")

    def stop(self):
        for key in self.data.keys():
            if isinstance(self.data[key], list):
                self.save_plot(key)

        with open(self.exp_dir / "parameters.txt", "w") as f:
            # Convert dictionary to JSON and write to file
            json.dump(self.parameters, f)

        if self.enable_neptune:
            self.run.stop()

    def save_plot(self, key):
        fig, ax = plt.subplots()
        y = self.data[key]
        ax.plot(y)
        ax.set_xlabel("step")
        ax.set_ylabel(key)
        fname = key.replace("/", "-")
        fig.savefig(self.exp_dir / f"{fname}.png")

    def __getitem__(self, key):
        return self.data[key]


def masked_cross_entropy(hypotheses, reference, mask):
    crossEntropy = torch.nn.functional.cross_entropy(
        hypotheses.transpose(1, 2), reference, reduction="none"
    )
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss


def save_checkpoint(model, optimizer, epoch, lr_scheduler, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    path = Path(checkpoint_dir) / "model_best.pt"

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        },
        path,
    )


def pplx(logits, tgt, mask):
    loss = masked_loss(logits, tgt, mask)
    return torch.exp(loss)


def validation(model, valid_dl, max_step):
    model.eval()
    loss_across_batches = []
    pplx_across_batches = []

    with torch.no_grad():
        bar = tqdm(valid_dl, leave=False)
        for step_no, batch in enumerate(bar):
            batch = [data.to(device) for data in batch]

            # calculate loss on valid
            loss, logits = step(model, batch)
            loss_across_batches.append(loss.item())

            pplx_across_batches.append(pplx(logits, batch[1], batch[3]).item())

            del batch

            if step_no == max_step:
                break

        bar.close()

    return {"loss": mean(loss_across_batches), "pplx": mean(pplx_across_batches)}


def masked_loss(logits, tgt, mask):
    return masked_cross_entropy(logits, tgt, mask)


def step(model, batch):
    src, tgt, src_mask, tgt_mask = batch
    logits = model(src, tgt, src_mask)
    loss = masked_loss(logits, tgt, tgt_mask)
    return loss, logits


def fit(
    model: nn.Module,
    optimizer,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    params: dict,
    max_epoch,
    lr_scheduler=None,
    checkpoint_dir="./checkpoint",
    max_step=-1,
    experiment_name=None,
    enable_neptune=False,
):

    history = History(enable_neptune, "learning-to-ask", experiment_name)
    history.hyperparameters(params)

    best_pplx = float("-inf")

    for epoch in range(1, max_epoch + 1):
        model.train()
        loss_across_batches = []
        bar = tqdm(train_dl, unit="batch")

        for step_no, batch in enumerate(bar):
            batch = [data.to(device) for data in batch]
            optimizer.zero_grad()
            loss, logits = step(model, batch)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            loss_across_batches.append(loss.item())

            # show each batch loss in tqdm bar
            bar.set_postfix(**{"loss": loss.item()})

            # skip training on the entire training dataset
            if step_no == max_step:
                break

        lr_scheduler.step()
        validation_metrics = validation(model, valid_dl, max_step)

        history.log("train/loss", mean(loss_across_batches))
        history.log("valid/loss", validation_metrics["loss"])
        history.log("valid/pplx", validation_metrics["pplx"])
        history.log("train/lr", lr_scheduler.get_last_lr()[0])

        if validation_metrics["pplx"] > best_pplx:
            best_pplx = validation_metrics["pplx"]
            save_checkpoint(model, optimizer, epoch, lr_scheduler, checkpoint_dir)
            print("🎉 best pplx reached, saved a checkpoint.")

        log_print(epoch, history)

    # upload the best model if neptune is enabled
    history.upload_checkpoint(checkpoint_dir + "/" + "model_best.pt", "model.pt")
    return history


def log_print(epoch, history):
    print(
        f"Epoch: {epoch},\tTrain Loss: {history['train/loss'][-1]},\tVal Loss: {history['valid/loss'][-1]}\tVal Perplexity: {history['valid/pplx'][-1]}\tLR: {history['train/lr'][-1]}"
    )
