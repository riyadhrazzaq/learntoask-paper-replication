from torchtext.vocab import build_vocab_from_iterator, GloVe


def yield_token(text_file_path):
    with io.open(text_file_path, encoding="utf-8") as f:
        for line in f:
            yield line.strip().split()
            
def load_and_build_vocab(sentence_path, question_path):
    sentence_vocab = build_vocab_from_iterator(
        yield_token(sentence_path),
        max_tokens=45000,
        specials=["<SOS>", "<EOS>", "<PAD>", "<UNK>"],
        special_first=True,
    )
    
    question_vocab = build_vocab_from_iterator(
        yield_token(question_path), max_tokens=28000
    )

    # merge two vocabs once collected from separate corpus
    vocab = torchtext.vocab.Vocab(sentence_vocab)
    vocab.set_default_index(vocab["<UNK>"])
    
    for token in question_vocab.get_itos():
        if token not in vocab:
            vocab.append_token(token)
    
    return vocab


def load_pretrained_glove(vocab, cache=None):
    embedding_vector = torch.zeros(size=(len(vocab), 300))
    glove = GloVe(cache="data/")
    for index in range(len(vocab)):
        embedding_vector[index] = glove[vocab.lookup_token(index)]