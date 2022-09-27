from pathlib import Path
import re
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, load_metric
import datasets
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, pipeline
from transformers import DataCollatorForTokenClassification
import numpy as np
from collections import OrderedDict
# from unidecode import unidecode
import torch

check = 0

torch.cuda.empty_cache()

model_checkpoint = "aicryptogroup/videberta-v3-base"
batch_size = 16
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def read_wnut(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n ?\n', raw_text)
    token_docs = []
    tag_docs = []
    count = 0

    for doc in raw_docs:
        tokens = []
        tags = []
        count = count + 1
        for line in doc.split('\n'):
          #  print(line)
            token, tag = line.split(' ')
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs


train_docs, train_tags = read_wnut("data_train.txt")
dev_docs, dev_tags = read_wnut("data_train.txt")
test_docs, test_tags = read_wnut("data_train.txt")

label_set = []
for i in train_tags:
    label_set.extend(i)
label_set = list(OrderedDict.fromkeys(label_set))
print(label_set)

label_to_ids = {lb: i for i, lb in enumerate(label_set)}

train_tag_ids = [[label_to_ids[w] for w in label] for label in train_tags]
dev_tag_ids = [[label_to_ids[w] for w in label] for label in dev_tags]
test_tag_ids = [[label_to_ids[w] for w in label] for label in test_tags]

df_train = pd.DataFrame(list(zip(train_docs, train_tag_ids)), columns=['tokens', 'tags'])
df_dev = pd.DataFrame(list(zip(dev_docs, dev_tag_ids)), columns=['tokens', 'tags'])
df_test = pd.DataFrame(list(zip(test_docs, test_tag_ids)), columns=['tokens', 'tags'])

train_dataset = Dataset.from_pandas(df_train)

dev_dataset = Dataset.from_pandas(df_dev)
test_dataset = Dataset.from_pandas(df_test)
datasets = DatasetDict({
    "train": train_dataset,
    "validation": dev_dataset,
    "test": test_dataset
})


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    label_all_tokens = True

    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    # print(labels)

    tokenized_inputs["labels"] = labels
    # print(labels)
    return tokenized_inputs


tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)
# print(tokenized_datasets)

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_to_ids))
data_collator = DataCollatorForTokenClassification(tokenizer)

metric = load_metric("seqeval")
label_list = label_set

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


model = model.to(device)

model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(
    f"{model_name}-finetuned",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=15,
    weight_decay=0.01,
    push_to_hub=False,
    save_strategy="epoch",
    metric_for_best_model='f1',
    load_best_model_at_end=True,
)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("model-thanhhoa")

# model_checkpoint = "aicryptogroup/videberta-v3-base"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# model = AutoModelForTokenClassification.from_pretrained("model-thanhhoa")
# ner_model = pipeline('ner', model=model, tokenizer=tokenizer)

print("Xong")
# print(ner_model(u = unicode(sequence, "utf-8")))