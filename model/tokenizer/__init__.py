from transformers import AutoTokenizer


def get_tokenizer(pretrained="bert-base-uncased"):
    return AutoTokenizer.from_pretrained(pretrained)
