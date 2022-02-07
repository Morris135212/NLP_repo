from transformers import AutoTokenizer


class BaseTokenizer(object):
    def __init__(self, pretrain="bert-base-uncased"):
        self.pretrain = pretrain

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.pretrain)
