import argparse

import jsonlines

from trainutil import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# define arguments, override the defaults from config.py with arguments
args = argparse.ArgumentParser()
args.add_argument('evaluation-file', type=str)
args.add_argument('outfile', type=str)
args.add_argument('checkpoint', type=str)

args.add_argument('--batch-size', type=int, default=cfg.batch_size)

args = args.parse_args()

# build param dictionary from args
params = vars(args)
params = {k.replace('-', '_'): v for k, v in params.items()}
params['model_name'] = cfg.model_name

logger.info(f"Params: {params}")


def generate(model, tokenizer, text, max_length):
    model.eval()

    with torch.no_grad():
        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

        # calculate loss on valid
        _, logits = step(model, encoding)
        logits = logits[0].transpose(0, 1)

        # returns a list of span
        hypothesis = parse_label_encoding(text, encoding, logits, LABELS)
    return hypothesis


def evaluate_from_dataloader(dataloader: DataLoader, outfile, model):
    model.to(device)
    all_logits = []
    with torch.no_grad():
        bar = tqdm(dataloader, leave=False)
        for step_no, batch in enumerate(bar):
            for key in batch["tensors"].keys():
                batch["tensors"][key] = batch["tensors"][key].to(device)

            # calculate loss on valid
            _, logits = step(model, batch["tensors"])

            logits = logits.transpose(1, 2)
            assert logits.size(1) == len(LABELS), "expects the label in dim=1"
            all_logits.append(logits)

    logger.info("🎉 Inference done! Saving predictions...")
    for batch_num, batch in tqdm(enumerate(dataloader)):
        logits = all_logits[batch_num]
        for i in range(logits.size(0)):
            hypothesis = parse_label_encoding(
                None, batch["encodings"][i], logits[i], LABELS
            )

            outfile.write(
                json.dumps({'id': batch["raws"][i]["id"], 'labels': hypothesis}, ensure_ascii=False) + "\n")

    logger.info("🎉 Saving done!")


def evaluate_from_file(filepath: str, model, tokenizer, max_length):
    """
    takes a filepath and saves output following the shared task's format and metrics if labels are available
    """
    infile = jsonlines.open(filepath)
    outfile = f"{args.work_dir}_{filepath}.hyp"
    outfile = open(outfile, 'w', encoding="utf-8")
    logger.info("file will be saved to %s", outfile)

    for sample in tqdm(infile):
        hypothesis = generate(model, tokenizer, sample['text'], max_length)
        outfile.write(json.dumps({'id': sample['id'], 'labels': hypothesis}, ensure_ascii=False) + "\n")
    infile.close()
    outfile.close()

    print('🎉 output saved to', outfile)


def main():
    tokenizer = BertTokenizerFast.from_pretrained(cfg.model_name)
    val_ds = DatasetFromJson(params['evaluation_file'], tokenizer, cfg.max_length)
    val_dl = DataLoader(val_ds, batch_size=params['batch_size'],
                        collate_fn=CollateFn(tokenizer=tokenizer, return_raw=True))

    model = model_init(params["model_name"],
                       not params["no_pretrain"],
                       params['nth_hidden_layer'])

    model, _, _, _ = load_checkpoint(model, params['checkpoint'])
    logger.info("🎉 Model loaded successfully!")

    logger.info("Generating predictions to %s", params['outfile'])

    outfile = open(params['outfile'], 'w', encoding="utf-8")
    evaluate_from_dataloader(val_dl, outfile, model)
    outfile.close()


if __name__ == '__main__':
    main()
