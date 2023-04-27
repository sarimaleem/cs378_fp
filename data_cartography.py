# in this file I am gonna recreate the data cartography repository.

import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json

NUM_PREPROCESSING_WORKERS = 2

# Loading the Data
default_datasets = {'qa': ('squad',), 'nli': ('snli',)}
dataset_id = ('snli',)
eval_split = 'validation_matched' if dataset_id == ('glue', 'mnli') else 'validation'
dataset = datasets.load_dataset(*dataset_id)
model_path = './trained_model/'

# TODO uncomment this
# dataset = dataset.filter(lambda ex: ex['label'] != -1)

train_dataset = dataset["train"]
eval_dataset = dataset["validation"]
test_dataset = dataset["test"]

# example_indicies = range(10)
# reduced_eval_dataset = train_dataset[example_indicies]

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
prepare_train_dataset = prepare_eval_dataset = \
    lambda exs: prepare_dataset_nli(exs, tokenizer, 128)



# Loading the model


train_dataset_featurized = None
eval_dataset_featurized = None

eval_dataset_featurized = eval_dataset.map(
    prepare_eval_dataset,
    batched=True,
    num_proc=NUM_PREPROCESSING_WORKERS,
    remove_columns=eval_dataset.column_names
)

# eval_dataset_featureized = eval_dataset_featurized[0:10]


model_classes = {'qa': AutoModelForQuestionAnswering,
                    'nli': AutoModelForSequenceClassification}
model_class = model_classes['nli']
model = model_class.from_pretrained(model_path)



# This function wraps the compute_metrics function, storing the model's predictions
# so that they can be dumped along with the computed metrics
compute_metrics = compute_accuracy
# This function wraps the compute_metrics function, storing the model's predictions
# so that they can be dumped along with the computed metrics
eval_predictions = None
def compute_metrics_and_store_predictions(eval_preds):
    global eval_predictions
    eval_predictions = eval_preds
    return compute_metrics(eval_preds)

trainer_class = Trainer
trainer = trainer_class(
    model=model,
    train_dataset=train_dataset_featurized,
    eval_dataset=eval_dataset_featurized,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_and_store_predictions
)

results = trainer.evaluate()

print("hi")