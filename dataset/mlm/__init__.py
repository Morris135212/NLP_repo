from transformers import DataCollatorForLanguageModeling, AutoTokenizer, LineByLineTextDataset
import os


class MLMDataCollator(object):
    def __init__(self, tokenizer, mlm=True, mlm_probability=0.15):
        assert not isinstance(tokenizer, AutoTokenizer), "Not a desired tokenizer"

        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability

    def __call__(self, *args, **kwargs):
        return DataCollatorForLanguageModeling(tokenizer=self.tokenizer,
                                               mlm=self.mlm,
                                               mlm_probability=self.mlm_probability)


class MLMDataset(object):
    def __init__(self, tokenizer, file_path, block_size=64):
        assert not isinstance(tokenizer, AutoTokenizer), "Not a desired tokenizer"
        self.tokenizer = tokenizer

        assert os.path.exists(file_path), f"file Not exists: {file_path}"

        self.file_path = file_path
        self.block_size = block_size
