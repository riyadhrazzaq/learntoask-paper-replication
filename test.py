import argparse
import logging
import unittest
from copy import deepcopy

import torchtext.vocab

from tokenization import Tokenizer
import datahandler as dh
import train

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class TestTokenization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = ["this is a test", "this is test"]
        cls.tokenized_samples = [sample.split() for sample in cls.samples]
        cls.vocab = torchtext.vocab.build_vocab_from_iterator(
            cls.tokenized_samples,
            specials=["<SOS>", "<EOS>", "<PAD>", "<UNK>"],
            special_first=True,
        )
        cls.vocab.set_default_index(cls.vocab["<UNK>"])
        cls.tokenizer = Tokenizer(
            cls.vocab, cls.vocab["<PAD>"], cls.vocab["<SOS>"], cls.vocab["<EOS>"]
        )

        logger.info("vocab %s", cls.vocab.get_stoi())

    def test_tokenizer_cutshort(self):
        text_tensor, mask = self.tokenizer.encode(
            self.samples, add_sos=True, add_eos=True, max_seq=3
        )
        logger.info("text_tensor %s", text_tensor)
        logger.info("mask %s", mask)

        decoded_text = self.tokenizer.decode(text_tensor)
        logger.info("decoded_text %s", decoded_text)

        self.assertEqual(decoded_text, ["<SOS> this <EOS>", "<SOS> this <EOS>"])

    def test_tokenizer_padded(self):
        text_tensor, mask = self.tokenizer.encode(
            self.samples, add_sos=True, add_eos=True, max_seq=10
        )
        logger.info("text_tensor %s", text_tensor)
        logger.info("mask %s", mask)

        decoded_text = self.tokenizer.decode(text_tensor)
        logger.info("decoded_text %s", decoded_text)

        self.assertEqual(
            decoded_text,
            [
                "<SOS> this is a test <EOS> <PAD> <PAD> <PAD> <PAD>",
                "<SOS> this is test <EOS> <PAD> <PAD> <PAD> <PAD> <PAD>",
            ],
        )


class TestModelForwardPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocabulary = [["palestine"], ["mexico"], ["i"], ["where"]]
        cls.vocab = torchtext.vocab.build_vocab_from_iterator(
            cls.vocabulary,
            specials=["<SOS>", "<EOS>", "<PAD>", "<UNK>"],
            special_first=True,
        )
        cls.vocab.set_default_index(cls.vocab["<UNK>"])
        cls.tokenizer = Tokenizer(
            cls.vocab, cls.vocab["<PAD>"], cls.vocab["<SOS>"], cls.vocab["<EOS>"]
        )
        cls.config = {
            "hidden_dim": 4,
            "embedding_dim": 300,
            "num_layers": 2,
            "dropout": 0.3,
            "bidirectional": True,
            "lr": 1.0,
            "lr_decay": 0.5,
            "lr_decay_from": 3,
            "clip_norm": 5,
            "max_epoch": 10,
            "max_step": float("inf"),
            "src_max_seq": 10,
            "tgt_max_seq": 5,
            "train_glove": False,
            "batch_size": 2,
        }

        logger.info("vocab %s", cls.vocab.get_stoi())

    def test_model_forward_pass(self):
        src_path = "resources/test/src.txt"
        tgt_path = "resources/test/tgt.txt"
        train_dl = dh.get_data_loader(
            src_path, tgt_path, self.tokenizer, self.config, shuffle=False
        )
        valid_dl = deepcopy(train_dl)

        model, optimizer, lr_scheduler, epoch = dh.load_or_build_models(
            None, "data/glove840B300d.pt", "data/", self.config, self.vocab
        )
        print(model)


if __name__ == "__main__":
    unittest.main()
