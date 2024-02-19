# model_definition.py
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from tokenizers import Tokenizer

def create_model(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model

def train_model(model, train_dataset, test_dataset, tokenizer, model_path):
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=model_path,
        overwrite_output_dir=True,
        num_train_epochs=24,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_steps=500,
        save_steps=1000,
        warmup_steps=300,
        prediction_loss_only=True,
        evaluation_strategy="steps",
        save_total_limit=3
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
