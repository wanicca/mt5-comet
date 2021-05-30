
import sys
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#from transformers import MT5Model, T5Tokenizer
from transformers import MT5ForConditionalGeneration, T5Tokenizer

#tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
raw_datasets = load_dataset("atomic")

def tokenize_function(examples):
    return tokenizer(examples["event"], padding="max_length", truncation=True)

def tokenize_labels(examples):
    with tokenizer.as_target_tokenizer():
         return tokenizer(examples["oReact"], return_tensors="pt")

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
#labels = raw_datasets.map(tokenize_labels, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["test"]


from transformers import TrainingArguments

training_args = TrainingArguments("test_trainer")
#model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
from transformers import Trainer

trainer = Trainer(
    model=model, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_eval_dataset
)

trainer.train()
import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.evaluate()


