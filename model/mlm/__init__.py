from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, LineByLineTextDataset


class BaseMLM(object):
    def __init__(self,
                 vocab_size=50000,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 max_position_embeddings=512):
        self.config = BertConfig(vocab_size=vocab_size,
                                 hidden_size=hidden_size,
                                 num_hidden_layers=num_hidden_layers,
                                 num_attention_heads=num_attention_heads,
                                 max_position_embeddings=max_position_embeddings)

    def __call__(self, *args, **kwargs):
        return BertForMaskedLM(self.config)
