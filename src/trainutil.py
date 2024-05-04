import json
import logging
import os
from pathlib import Path
from statistics import mean
import yaml

import numpy as np
import torch
import torchmetrics
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using device: {}".format(device))


def validation(model, valid_dl, max_step, valid_step_interval, pad_index):
    print('Validating...')
    model.eval()
    loss_across_batches = []
    metric_across_batches = []

    with torch.no_grad():
        for step_no, batch in enumerate(valid_dl):
            batch = [tensor.to(device) for tensor in batch]

            # calculate loss on valid
            loss, logits = step(model, batch)
            loss_across_batches.append(loss.item())

            metric = torchmetrics.functional.text.perplexity(logits, batch[1], ignore_index=pad_index)

            # sum up
            metric_across_batches.append(metric.item())

            if step_no == max_step:
                break

            if step_no % valid_step_interval == 0:
                print(f"\tStep: {step_no}/{len(valid_dl)}, Loss: {loss.item()}")

    return {"loss": mean(loss_across_batches), "pplx": mean(metric_across_batches)}


def masked_cross_entropy(logits, reference, mask):
    loss_unreduced = torch.nn.functional.cross_entropy(
        logits.transpose(1, 2), reference, reduction="none"
    )
    loss = loss_unreduced.masked_select(mask).mean()
    loss = loss.to(device)
    return loss


def step(model, batch):
    src, tgt, src_mask, tgt_mask = batch
    logits = model(src, tgt, src_mask)
    loss = masked_cross_entropy(logits, tgt, tgt_mask)
    return loss, logits


def fit(
        model: nn.Module,
        optimizer,
        train_dl: torch.utils.data.DataLoader,
        valid_dl: torch.utils.data.DataLoader,
        cfg: dict,
        checkpoint_dir="./checkpoint",
        max_step=-1,
        epoch=0,
        lr_scheduler=None,
        ignore_index=-100
):
    logger.info("checkpoint_dir: {}".format(checkpoint_dir))
    np.random.seed(cfg['random_seed'])
    torch.manual_seed(cfg['random_seed'])
    best_pplx = float("inf")
    history = {"train/loss": [], "valid/loss": [], "valid/pplx": []}

    model.to(device)

    for epoch in range(epoch + 1, epoch + cfg["max_epoch"] + 1):
        print("Training...")
        model.train()
        loss_across_batches = []

        for step_no, batch in enumerate(train_dl):
            # move to gpu
            batch = [tensor.to(device) for tensor in batch]

            # reset grads
            optimizer.zero_grad()

            # step forward
            loss, logits = step(model, batch)

            # step backward
            loss.backward()

            optimizer.step()
            loss_across_batches.append(loss.item())

            # skip training on the entire training dataset
            # useful during debugging
            if step_no == max_step:
                break

            if step_no % cfg['train_step_interval'] == 0:
                print(f"\tStep: {step_no}/{len(train_dl)}, Loss: {loss.item()}")
        
        if lr_scheduler:
                lr_scheduler.step()

        validation_metrics = validation(model, valid_dl, max_step, cfg['valid_step_interval'], ignore_index)

        history["train/loss"].append(mean(loss_across_batches))
        history['valid/loss'].append(validation_metrics['loss'])
        history['valid/pplx'].append(validation_metrics['pplx'])

        if validation_metrics["pplx"] < best_pplx:
            best_pplx = validation_metrics["pplx"]
            save_checkpoint(model, optimizer, epoch, checkpoint_dir, "model_best.pt")
            print("\n🎉 best pplx reached, saved a checkpoint.")

        log(epoch, history)

    # save last checkpoint
    save_checkpoint(model, optimizer, epoch, checkpoint_dir, "model_last.pt")
    return history


def log(epoch, history):
    print(
        f"Epoch: {epoch},\tTrain Loss: {history['train/loss'][-1]},\tVal Loss: {history['valid/loss'][-1]}\tval pplx: {history['valid/pplx'][-1]}"
    )
    
    
def save_checkpoint(model, optimizer, src_tokenizer, tgt_tokenizer, epoch, checkpoint_dir, suffix="model_best.pt"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_dir = Path(checkpoint_dir)
    model_path = checkpoint_dir / suffix

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        model_path,
    )
    torch.save(src_tokenizer, checkpoint_dir / "src_tokenizer.pt")
    torch.save(tgt_tokenizer, checkpoint_dir / "tgt_tokenizer.pt")


def load_tokenizers(checkpoint_dir):
    src_tokenizer = torch.load(checkpoint_dir / "src_tokenizer.pt")
    tgt_tokenizer = torch.load(checkpoint_dir / "tgt_tokenizer.pt")
    return src_tokenizer, tgt_tokenizer


def load_checkpoint(model, checkpoint_path, optimizer=None, lr_scheduler=None):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
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
        
        


def save_history(history, config, history_dir, save_graph=True):
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)

    with open(f"{history_dir}/history.json", "w") as f:
        json.dump(history, f)

    with open(f"{history_dir}/config.yaml", "w") as f:
        yaml.dump(config, f)

    if save_graph:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(history["train/loss"], label="train/loss")
        ax.plot(history["valid/loss"], label="valid/loss")
        ax.legend()
        plt.savefig(f"{history_dir}/history_loss.png")

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(history["valid/pplx"], label="valid/pplx")
        ax.legend()
        plt.savefig(f"{history_dir}/history_pplx.png")

        

def init_optimizer_scheduler(model, cfg):
    if cfg["optim"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
    
    elif cfg["optim"] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
    lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer,
                                                             lr_lambda=lambda epoch: cfg['lr_decay'] if epoch > cfg['lr_decay_from'] else 1.0)
    return optimizer, lr_scheduler
        


# def train(parameters, trial=None):
#     logger.info(f"Training Parameters: {parameters}")
#
#     tokenizer = BertTokenizerFast.from_pretrained(cfg.model_name)
#     train_ds = DatasetFromJson(parameters["training_file"], tokenizer, cfg.max_length)
#     val_ds = DatasetFromJson(parameters["validation_file"], tokenizer, cfg.max_length)
#     train_dl = DataLoader(
#         train_ds,
#         batch_size=parameters["batch_size"],
#         collate_fn=CollateFn(tokenizer, return_raw=False),
#     )
#     val_dl = DataLoader(
#         val_ds,
#         batch_size=parameters["batch_size"],
#         collate_fn=CollateFn(tokenizer=tokenizer, return_raw=True),
#     )
#
#     model = model_init(parameters["model_name"],
#                        not parameters["no_pretrain"],
#                        parameters['nth_hidden_layer'])
#     optimizer = torch.optim.AdamW(
#         model.parameters(), lr=parameters["lr"], weight_decay=parameters["weight_decay"]
#     )
#     num_training_steps = parameters['max_epoch'] * len(train_dl)
#     lr_scheduler = get_scheduler(
#         name="linear", optimizer=optimizer, num_warmup_steps=parameters['warmup_steps'],
#         num_training_steps=num_training_steps
#     )
#     checkpoint_dir = f"{cfg.checkpoint_dir}/{parameters['experiment_name']}"
#     history_dir = f"{checkpoint_dir}/history"
#     history = fit(
#         model,
#         optimizer,
#         train_dl,
#         val_dl,
#         parameters,
#         checkpoint_dir,
#         max_step=parameters["max_step"],
#         epoch=0,
#         lr_scheduler=lr_scheduler,
#         trial=trial,
#         disable_tqdm=disable_tqdm
#     )
#     save_history(history, history_dir, save_graph=True)
#     return history
