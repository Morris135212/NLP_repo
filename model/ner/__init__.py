from transformers import BertForTokenClassification, BertConfig, \
    AutoModelForTokenClassification, AutoConfig

from model.tokenizer import BaseTokenizer


class BaseTokenClassificiation(BaseTokenizer):
    def __init__(self,
                 num_labels,
                 vocab_size=50000,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 max_position_embeddings=512):
        super(BaseTokenClassificiation, self).__init__()
        self.config = BertConfig(vocab_size=vocab_size,
                                 hidden_size=hidden_size,
                                 num_hidden_layers=num_hidden_layers,
                                 num_attention_heads=num_attention_heads,
                                 max_position_embeddings=max_position_embeddings,
                                 num_labels=num_labels)

    def __call__(self, *args, **kwargs):
        return BertForTokenClassification(self.config)


class AutoTokenClassification(BaseTokenizer):
    def __init__(self, num_labels, model_name="bert-base-uncased"):
        super(AutoTokenClassification, self).__init__(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.model_name = model_name
        self.config.num_labels = num_labels

    def __call__(self, *args, **kwargs):
        return AutoModelForTokenClassification.from_pretrained(self.model_name,
                                                               config=self.config)
