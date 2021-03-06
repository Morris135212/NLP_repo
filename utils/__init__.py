from enum import Enum


class model_map(Enum):
    """
    You can find models from this website
    https://huggingface.co/models
    """
    bert = "bert-base-uncased"
    bert_chinese = "bert-base-chinese"
    bigbird = "google/bigbird-roberta-base"


