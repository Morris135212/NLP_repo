from transformers import BertConfig, BertForSequenceClassification

from model.tokenizer import BaseTokenizer


class BaseCls(BaseTokenizer):
    def __init__(self, num_labels, vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                 max_position_embeddings=512, pretrain="bert-base-uncased"):
        super(BaseTokenizer, self).__init__()
        self.config = BertConfig(vocab_size=vocab_size,
                                 hidden_size=hidden_size,
                                 num_hidden_layers=num_hidden_layers,
                                 num_attention_heads=num_attention_heads,
                                 max_position_embeddings=max_position_embeddings)
        self.pretrain = pretrain
        self.num_labels = num_labels

    def __call__(self, *args, **kwargs):
        return BertForSequenceClassification(self.config, num_labels=self.num_labels).from_pretrained(self.pretrain)
