from transformers import Trainer, TrainingArguments
import os
from pathlib import Path
from utils.general import increment_path


class MLMTrainer(object):
    """
    The most common way to pretrain the model
    """
    def __init__(self,
                 model,
                 data_collator,
                 train_dataset,
                 eval_dataset,
                 output_dir="./",
                 name="runs",
                 overwrite_output_dir=True,
                 num_train_epochs=100,
                 per_device_train_batch_size=64,
                 save_steps=10000):
        self.model = model
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = increment_path(Path(output_dir)/name, exist_ok=False)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = f"{output_dir}/{name}"
        print(f"Saving output into {self.output_dir}")
        self.overwrite_output_dir = overwrite_output_dir
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.save_steps = save_steps

    def __call__(self, *args, **kwargs):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=self.overwrite_output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            save_steps=self.save_steps
        )

        return Trainer(model=self.model,
                       args=training_args,
                       data_collator=self.data_collator,
                       train_dataset=self.train_dataset,
                       eval_dataset=self.eval_dataset
                       )
