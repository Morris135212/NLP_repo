from transformers import Trainer, TrainingArguments


class MLMTrainer(object):
    def __init__(self,
                 model,
                 data_collator,
                 dataset,
                 output_dir="./",
                 overwrite_output_dir=True,
                 num_train_epochs=100,
                 per_device_train_batch_size=64,
                 save_steps=10000):
        self.model = model
        self.data_collator = data_collator
        self.dataset = dataset
        self.output_dir = output_dir
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
                       dataset=self.dataset)
