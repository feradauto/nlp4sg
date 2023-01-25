from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split
from datasets import load_metric
from datasets import Dataset
import torch
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from sklearn.metrics import classification_report
import os
import sys
import pandas as pd
import numpy as np
import argparse
import re
from collections import Counter
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
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
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True,max_length=512)

def get_report(df_results):
    """Get classification report

    Parameters:
    df_results: should have column label and prediction
    """
    cr=classification_report(df_results.label,df_results.prediction,digits=4,output_dict=True)

    cr=pd.DataFrame(cr).reset_index().rename(columns={'index':'metric'})

    cr_df=pd.melt(cr,id_vars=['metric'],value_vars=['0','1','accuracy','macro avg','weighted avg'])

    cr_df=cr_df.loc[cr_df.metric!="support"]
    cr_df=cr_df.loc[~((cr_df.variable=="accuracy") & (cr_df.metric.isin(['precision','recall'])))]

    cr_df=cr_df.assign(variable=np.where(cr_df.variable=='0','negative',
                                        np.where(cr_df.variable=='1','positive',cr_df.variable)))

    cr_df=cr_df.assign(value=cr_df.value.apply(lambda x:round(x,4)*100))
    return cr_df

def get_all_results(df_predictions,df_labeled):
    df_labeled=df_labeled.assign(prediction=df_labeled.label)
    df_predictions=pd.concat([df_predictions,df_labeled])
    total_positives=df_predictions.loc[df_predictions.prediction==1]
    total_negatives=df_predictions.loc[df_predictions.prediction==0]
    total_negatives=total_negatives.loc[:,['ID','title','abstract','url','year','title_abstract']]
    total_negatives=total_negatives.assign(label=0)
    total_positives=total_positives.loc[:,['ID','title','abstract','url','year','title_abstract']]
    total_positives=total_positives.assign(label=1)
    return total_positives,total_negatives

def evaluate(df_test_final,trainer,threshold=0.5):
    """Get predictions

    Parameters:
    df_test_final (df): dataframe with text for predictions
    trainer: Trainer with all the configurations
    Returns:
    dataset_test_final_pd
    """
    dataset_test_final = Dataset.from_pandas(df_test_final)
    tokenized_datasets_test_final = dataset_test_final.map(tokenize_function, batched=True)

    test_results_final = trainer.predict(tokenized_datasets_test_final)
    preds_final=[]
    for e in test_results_final.predictions:
        preds_final.append(np.array(torch.softmax(torch.Tensor(e), dim=0)))

    preds_final=np.vstack(preds_final)
    dataset_test_final_pd=tokenized_datasets_test_final.data.to_pandas()

    dataset_test_final_pd=dataset_test_final_pd.assign(proba0=preds_final[:,0])
    dataset_test_final_pd=dataset_test_final_pd.assign(proba1=preds_final[:,1])
    dataset_test_final_pd=dataset_test_final_pd.assign(prediction=np.where(dataset_test_final_pd.proba1>threshold,1,0))
    return dataset_test_final_pd


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-o','--option', nargs='?', help='0 evaluate only test set, 1 for full results',default='1')
    args = parser.parse_args()
    data_path="../../data/"
    outputs_path="../../outputs/"
    ## READ DATA

    ## annotated test dataset
    df_test_final=pd.read_csv(outputs_path+"general/test_set_final.csv")
    df_dev_final=pd.read_csv(outputs_path+"general/dev_set_final.csv")
    df_train_final=pd.read_csv(outputs_path+"general/train_set_final.csv")

    df_unused=pd.read_csv(outputs_path+"sg_classifier/weakly_labeled_unused_bronze_ds_final.csv")
    df_unlabeled=pd.read_csv(outputs_path+"sg_classifier/unlabeled_set_bronze_ds_final.csv")
    df_unused=df_unused.assign(text=df_unused.title_abstract)
    df_unlabeled=df_unlabeled.assign(text=df_unlabeled.title_abstract)

    df_labeled=pd.read_csv(outputs_path+"sg_classifier/train_set_labeled_bronze_ds_final.csv")

    model = AutoModelForSequenceClassification.from_pretrained("../../../../../models/model_scibert_final", num_labels=2)


    ## Predict test dataset

    training_args = TrainingArguments(output_dir="../../../../../models/model_scibert_final", evaluation_strategy="epoch",
                                     per_device_train_batch_size=16,per_device_eval_batch_size=16,
                                     seed=42,num_train_epochs=15,auto_find_batch_size=True,
                                         do_train = False,do_predict = True)
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
    )
    df_test_final=df_test_final.iloc[:2000,:]
    if args.option=='0':
        print("Test set evaluation")
        df_results_dev=evaluate(df_dev_final,trainer,0.5)
        ## select best threshold
        best_threshold=0.5
        best_f1=0
        for p in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            df_analyze=df_results_dev.assign(prediction2=np.where(df_results_dev.proba1>p,1,0))
            f1=classification_report(df_analyze.label,df_analyze.prediction2,digits=4,output_dict=True)['1']['f1-score']
            if f1>best_f1:
                best_f1=f1
                best_threshold=p
        print("best_f1",best_f1)
        print("best_threshold",best_threshold)        
        df_results=evaluate(df_test_final,trainer,best_threshold)
        cr_final=get_report(df_results)
        cr_final=cr_final.rename(columns={'value':'Fine tuned SciBERT'})
        cr_final.to_csv(outputs_path+"sg_classifier/scores_scibert.csv",index=False)
        print("results: ",outputs_path+"sg_classifier/scores_scibert.csv")
    elif args.option=='1':
        print("All set evaluation")
        df_results_dev=evaluate(df_dev_final,trainer,0.5)
        ## select best threshold
        best_threshold=0.5
        best_f1=0
        for p in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            df_analyze=df_results_dev.assign(prediction2=np.where(df_results_dev.proba1>p,1,0))
            f1=classification_report(df_analyze.label,df_analyze.prediction2,digits=4,output_dict=True)['1']['f1-score']
            if f1>best_f1:
                best_f1=f1
                best_threshold=p
        print("best_f1",best_f1)
        print("best_threshold",best_threshold)        
        df_results=evaluate(df_test_final,trainer,best_threshold)
        df_general=pd.concat([df_unused,df_unlabeled]).reset_index(drop=True)
        df_general.label=df_general.label.fillna(0)
        df_general.label=df_general.label.apply(int)
        df_predictions=evaluate(df_general,trainer,best_threshold)
        total_positives,total_negatives=get_all_results(df_predictions,df_labeled)
        total_positives.to_csv(outputs_path+"sg_classifier/all_positive_examples_scibert_ds.csv",index=False)
        total_negatives.to_csv(outputs_path+"sg_classifier/all_negative_examples_scibert_ds.csv",index=False)

        cr_model=get_report(df_results)
        cr_model=cr_model.rename(columns={'value':'Fine tuned SciBERT'})
        cr_model.to_csv(outputs_path+"sg_classifier/scores_scibert.csv",index=False)
        print("results: ",outputs_path+"sg_classifier/scores_scibert.csv")


if __name__ == '__main__':
    main()

