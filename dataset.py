import torch


class SentenceQuestionDataset(Dataset):
    def __init__(
        self,
        sentences: torch.Tensor,
        questions: torch.Tensor,
        sentences_mask=None,
        questions_mask=None,
    ):
        """
        Args:
            sentences (torch.Tensor): (N, Ls) A tensor containing the sentences.
            questions (torch.Tensor): (N, Lq) seq_len A tensor containing the questions.
            sentences_mask (torch.Tensor): (N, Ls) A tensor containing the sentences mask.
            questions_mask (torch.Tensor): (N, Lq) A tensor containing the questions mask.
        """
        self.sentences = sentences
        self.questions = questions
        self.sentences_mask = sentences_mask
        self.questions_mask = questions_mask

    def __len__(self):
        return self.sentences.size(0)

    def __getitem__(self, index):
        return (
            self.sentences[index],
            self.questions[index],
            self.sentences_mask[index],
            self.questions_mask[index],
        )
