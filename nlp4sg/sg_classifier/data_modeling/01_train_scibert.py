from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split
import pandas as pd
import numpy as np
from datasets import load_metric
from datasets import Dataset
import torch
import transformers
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoModelForSequenceClassification,AutoTokenizer
import os
os.environ["WANDB_DISABLED"] = "true"

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    recall = recall_score(y_true=labels, y_pred=predictions)
    precision = precision_score(y_true=labels, y_pred=predictions)
    recall_w = recall_score(y_true=labels, y_pred=predictions,average='weighted')
    precision_w = precision_score(y_true=labels, y_pred=predictions,average='weighted')
    f1 = f1_score(y_true=labels, y_pred=predictions)
    f1_pos = f1_score(y_true=labels, y_pred=predictions,average='binary',pos_label=1)
    f1_micro = f1_score(y_true=labels, y_pred=predictions,average='micro')
    f1_weighted = f1_score(y_true=labels, y_pred=predictions,average='weighted')
    return {"accuracy": accuracy, "precision": precision, "recall": recall,
             "precision_w": precision_w, "recall_w": recall_w,
             "f1": f1,"f1_pos": f1_pos,
            "f1_micro": f1_micro,"f1_weighted": f1_weighted} 


def freeze_weights(m):
    for name, param in m.named_parameters():
        param.requires_grad = False
def model_init():
    transformers.set_seed(42)
    m=AutoModelForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased", num_labels=2)
    m.bert.apply(freeze_weights)
    for name, param in m.bert.pooler.named_parameters():
        param.requires_grad = True
    for name, param in m.bert.encoder.layer[-1].named_parameters():
        param.requires_grad = True
    return m


def train(tokenizer,dataset):
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True,max_length=512)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # Create a 85-15 train-validation split.
    # Calculate the number of samples to include in each set.
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    torch.manual_seed(42)
    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(tokenized_datasets, [train_size, val_size])


    training_args = TrainingArguments(output_dir="./model_scibert_final", evaluation_strategy="epoch",
                                 per_device_train_batch_size=32,per_device_eval_batch_size=32,
                                 seed=42,num_train_epochs=15,auto_find_batch_size=True)
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer

def main():
    data_path="../../data/"
    outputs_path="../../outputs/"
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

    df=pd.read_csv(outputs_path+"train_set_labeled_bronze_ds_final.csv")
    df['text']=df.title_abstract
    df=df.loc[:,['ID','text','label']]
    df['label']=df.label.apply(int)
    dataset = Dataset.from_pandas(df)
    
    trainer=train(tokenizer,dataset)
    trainer.save_model("./model_scibert_final")
    
if __name__ == '__main__':
    main()
