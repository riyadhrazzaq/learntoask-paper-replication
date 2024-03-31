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
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def save_checkpoint(model, optimizer, epoch, lr_scheduler, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    path = Path(checkpoint_dir) / "model_best.pt"

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
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
    # tgt <sos> should not be considered for loss because we have never predicted that
    loss = F.cross_entropy(logits.transpose(1, 2), tgt[:, 1:], reduction="none")
    loss = loss.masked_select(mask[:, 1:]).mean()
    return loss


def step(model, batch):
    src, tgt, _, tgt_mask = batch

    # tgt <eos> should not be a prompt
    logits, scores = model(src, tgt[:, :-1])
    logits = torch.cat(logits, dim=0).view(logits[0].size(0), -1, logits[0].size(1))
    loss = masked_loss(logits, tgt, tgt_mask)
    return loss, logits


def fit(
    model: nn.Module,
    optimizer,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    tokenizer,
    config: dict,
    max_epoch=10,
    lr_scheduler=None,
    checkpoint_dir="./checkpoint",
    max_step=None,
    validation_data: List[str] = None,
):

    best_pplx = float("-inf")

    history = {"loss/train": [], "pplx/valid": [], "loss/valid": []}

    for epoch in range(1, max_epoch + 1):
        model.train()
        loss_across_batches = []
        bar = tqdm(train_dl, unit="batch")

        for step_no, batch in enumerate(bar):
            batch = [data.to(device) for data in batch]
            optimizer.zero_grad()
            loss, logits = step(model, batch)

            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            loss_across_batches.append(loss.item())

            # show each batch loss in tqdm bar
            bar.set_postfix(**{"loss": loss.item()})

            # skip training on the entire training dataset
            if step_no == max_step:
                break

            # free memory
            del batch

        validation_metrics = validation(model, valid_dl, max_step)

        history["loss/train"] = mean(loss_across_batches)
        history["loss/valid"] = validation_metrics["loss"]
        history["pplx/valid"] = validation_metrics["pplx"]

        if validation_metrics["pplx"] > best_pplx:
            best_pplx = validation_metrics["pplx"]
            save_checkpoint(model, optimizer, epoch, lr_scheduler, checkpoint_dir)

        log(epoch, history)


def log(epoch, history):
    print(
        f"Epoch: {epoch},\tVal Loss: {history['loss/valid']},\tVal Loss: {history['loss/valid']}\tVal Perplexity: {history['pplx/valid']}"
    )
