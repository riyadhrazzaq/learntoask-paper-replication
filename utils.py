import os
from pathlib import Path
from statistics import mean
from typing import List

import neptune
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

import logging

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"


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
    src, tgt, _, tgt_mask = batch

    logits = model(src, tgt)
    loss = masked_loss(logits, tgt, tgt_mask)
    return loss, logits


def fit(
    model: nn.Module,
    optimizer,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    config: dict,
    args,
    lr_scheduler=None,
    checkpoint_dir="./checkpoint",
    max_step=-1,
    experiment_name=None,
    epoch=0,
):
    if args.enable_neptune:
        api_token = os.environ["NEPTUNE_API_TOKEN"]
        run = neptune.init_run(
            project="riyadhrazzaq/learning-to-ask",
            name=experiment_name,
            api_token=api_token,
        )
        run["parameters"] = config

    best_pplx = float("inf")

    history = {
        "loss/train": [],
        "pplx/valid": [],
        "loss/valid": [],
        "train/epoch/lr": [],
    }

    for epoch in range(epoch + 1, epoch + config["max_epoch"] + 1):
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
            if args.enable_neptune:
                run["train/batch/loss"].append(loss.item())

            # skip training on the entire training dataset
            if step_no == max_step:
                break

        lr_scheduler.step()
        validation_metrics = validation(model, valid_dl, max_step)

        history["loss/train"].append(mean(loss_across_batches))
        history["loss/valid"].append(validation_metrics["loss"])
        history["pplx/valid"].append(validation_metrics["pplx"])
        history["train/epoch/lr"].append(lr_scheduler.get_last_lr()[0])

        if validation_metrics["pplx"] < best_pplx:
            best_pplx = validation_metrics["pplx"]
            save_checkpoint(model, optimizer, epoch, lr_scheduler, checkpoint_dir)
            print("🎉 best pplx reached, saved a checkpoint.")

        log(epoch, history, run if args.enable_neptune else None, args.enable_neptune)

    if args.enable_neptune:
        run.stop()


def log(epoch, history, run, enable_neptune):
    if enable_neptune:
        run["train/epoch/loss"].append(history["loss/train"][-1])
        run["valid/epoch/pplx"].append(history["pplx/valid"][-1])
        run["valid/epoch/loss"].append(history["loss/valid"][-1])
        run["train/epoch/lr"].append(history["train/epoch/lr"][-1])

    print(
        f"Epoch: {epoch},\tTrain Loss: {history['loss/train'][-1]},\tVal Loss: {history['loss/valid'][-1]}\tVal Perplexity: {history['pplx/valid'][-1]}\tLR: {history['train/epoch/lr'][-1]}"
    )


def load_checkpoint(model, checkpoint_path, optimizer=None, lr_scheduler=None):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if lr_scheduler:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        epoch = checkpoint["epoch"]

        logger.info(f"🎉 Loaded existing model. Epoch: {checkpoint['epoch']}")
        return model, optimizer, lr_scheduler, epoch

    else:
        raise Exception("No checkpoint found in the provided path")
