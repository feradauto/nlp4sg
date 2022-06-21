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
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

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
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def get_rules_classifier(df_test_final,match_unique,keywords,workshops):
    """Rule classifier

    Parameters:
    train_set (df): Dataframe with train setpapers information
    match_unique (df): Dataframe with all papers information
    keywords (df):
    workshops (df):
    Returns:
    dataframe with predictions
    """
    keywords=keywords.assign(Keywords=np.where(keywords.Keywords=='asl',' asl ',keywords.Keywords))

    df_test_final=df_test_final.merge(match_unique,on=['ID'])
    ## percentile 99 of the cosine similarity with social needs
    perc_99=0.222915

    df_test_final=df_test_final.assign(abstract=df_test_final.abstract.fillna(''))

    df_test_final=df_test_final.assign(title_abstract=df_test_final.title+". "+df_test_final.abstract)

    df_test_final.title_abstract=df_test_final.title_abstract.replace("{","",regex=True).replace("}","",regex=True)

    df_test_final_positive=df_test_final.loc[(df_test_final.url.str.lower().str.contains('|'.join(list(workshops.Event.values))))|
               (df_test_final.title_abstract.str.lower().str.contains('|'.join(list(keywords.Keywords.values)))) |
               (df_test_final.cosine_similarity>=perc_99),:]

    df_test_final_negative=df_test_final.loc[~((df_test_final.url.str.lower().str.contains('|'.join(list(workshops.Event.values))))|
               (df_test_final.title_abstract.str.lower().str.contains('|'.join(list(keywords.Keywords.values)))) |
               (df_test_final.cosine_similarity>=perc_99)),:]

    df_test_final_positive=df_test_final_positive.assign(prediction=1)
    df_test_final_negative=df_test_final_negative.assign(prediction=0)


    df_rules=pd.concat([df_test_final_positive,df_test_final_negative])
    return df_rules

def process_test_dataset(df_test_final,anthology):
    """Format test dataset

    Parameters:
    df_test_final (df): Test dataset 
    anthology (df): Dataframe with all papers information
    Returns:
    df_test_final
    """
    df_test_final=df_test_final.loc[~df_test_final["SG_or_not (Jad's Annotation)"].isna(),:]
    df_test_final=df_test_final.assign(SG_or_not=np.where(df_test_final["Zhijing's annotation of SG_or_not"]+
                                       df_test_final["SG_or_not (Jad's Annotation)"]>0,1,0
                                      ))
    anthology=anthology.assign(abstract=anthology.abstract.fillna(''))
    anthology=anthology.assign(title_abstract=anthology.title+". "+anthology.abstract)
    anthology=anthology.loc[:,['ID','title_abstract']]
    anthology.title_abstract=anthology.title_abstract.replace("{","",regex=True).replace("}","",regex=True)
    df_test_final=df_test_final.merge(anthology,on=['ID'])
    df_test_final=df_test_final.loc[:,['ID','SG_or_not','title_abstract','url']].rename(columns={'SG_or_not':'label','title_abstract':'text'})
    df_test_final=df_test_final.loc[:,['ID','text','label','url']]
    df_test_final=df_test_final.loc[~df_test_final.label.isna()]
    df_test_final.label=df_test_final.label.apply(int)
    return df_test_final

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

def get_all_results(df_unused_positive,df_labeled,df_unlabeled):

    total_positives=pd.concat([df_unused_positive,df_labeled.loc[df_labeled.label==1],df_unlabeled.loc[df_unlabeled.prediction==1]])

    total_negatives=pd.concat([df_labeled.loc[df_labeled.label==0],df_unlabeled.loc[df_unlabeled.prediction==0]])

    total_negatives=total_negatives.loc[:,['ID','title','abstract','url','year','title_abstract']]

    total_negatives=total_negatives.assign(label=0)

    total_positives=total_positives.loc[:,['ID','title','abstract','url','year','title_abstract']]

    total_positives=total_positives.assign(label=1)
    return total_positives,total_negatives

def evaluate(df_test_final,trainer):
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
    dataset_test_final_pd=dataset_test_final_pd.assign(prediction=np.where(dataset_test_final_pd.proba1>0.5,1,0))
    return dataset_test_final_pd



def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-o','--option', nargs='?', help='0 evaluate only test set, 1 for full results',default='1')
    args = parser.parse_args()
    data_path="../../data/"
    outputs_path="../../outputs/"
    ## READ DATA
    workshops=pd.read_csv(data_path+"others/sg_workshops.csv")
    keywords=pd.read_csv(data_path+"others/sg_keywords.csv")
    ## text info of the dataset (it is more complete since it was extracted directly from the pdfs)
    anthology_test=pd.read_csv(data_path+"test_data/papers_ack.csv")
    ## annotated test dataset
    df_test_final=pd.read_csv(data_path+"test_data/test_set_SG_annotate_500.csv")

    match_unique=pd.read_csv(outputs_path+"general/papers_uniques.csv")

    df_unused_positive=pd.read_csv(outputs_path+"sg_classifier/unused_positive.csv")

    df_unlabeled=pd.read_csv(outputs_path+"sg_classifier/unlabeled_set.csv")

    df_labeled=pd.read_csv(outputs_path+"sg_classifier/train_set_labeled.csv")

    
    model = AutoModelForSequenceClassification.from_pretrained("./model/", num_labels=2)

    df_unlabeled=df_unlabeled.assign(text=df_unlabeled.title_abstract)

    ## Predict test dataset
    df_test_final=process_test_dataset(df_test_final,anthology_test)

    training_args = TrainingArguments(output_dir="model_finetuned", evaluation_strategy="epoch",
                                     per_device_train_batch_size=16,per_device_eval_batch_size=16,
                                     seed=42,num_train_epochs=5,auto_find_batch_size=True,
                                         do_train = False,do_predict = True)
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
    )


    if args.option=='0':
        print("Test set evaluation")
        df_results=evaluate(df_test_final,trainer)
        cr_final=get_report(df_results)
        cr_final=cr_final.rename(columns={'value':'Fine tuned BERT'})
        cr_final.to_csv(outputs_path+"sg_classifier/scores_bert.csv",index=False)
        print("results: ",outputs_path+"sg_classifier/scores_bert.csv")
    elif args.option=='1':
        print("Test set evaluation")
        df_results=evaluate(df_test_final,trainer)
        df_rules=get_rules_classifier(df_test_final,match_unique,keywords,workshops)
        print("Unlabeled set evaluation evaluation")
        df_unlabeled=evaluate(df_unlabeled,trainer)

        total_positives,total_negatives=get_all_results(df_unused_positive,df_labeled,df_unlabeled)
        total_positives.to_csv(outputs_path+"sg_classifier/all_positive_examples.csv",index=False)
        total_negatives.to_csv(outputs_path+"sg_classifier/all_negative_examples.csv",index=False)

        cr_model=get_report(df_results)
        cr_df_rules=get_report(df_rules)
        cr_df_rules=cr_df_rules.rename(columns={'value':'Rules classifier'})
        cr_model=cr_model.rename(columns={'value':'Fine tuned BERT'})
        cr_final=cr_model.merge(cr_df_rules,on=['metric','variable'])
        cr_final.to_csv(outputs_path+"sg_classifier/scores.csv",index=False)
        print("results: ",outputs_path+"sg_classifier/scores.csv")

    print(cr_final)

if __name__ == '__main__':
    main()

