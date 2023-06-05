## Example file evaluating model for task 1 using the test set of our NLP4SGPapers dataset
import argparse
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification,AutoTokenizer,pipeline
import pandas as pd
import csv
# function that takes a parameter and yields the dataset loaded from huggingface or from json file
# if the dataset is not available in huggingface
def load_data(dataset):
    print(dataset)
    if dataset != "feradauto/NLP4SGPapers":
        dataset = load_dataset("parquet", data_files={"test": dataset})['test']
        return dataset
    else:
        return load_dataset(dataset)['test']
  

def main(args):

    tokenizer = AutoTokenizer.from_pretrained("feradauto/scibert_nlp4sg",truncation=True)
    model = AutoModelForSequenceClassification.from_pretrained("feradauto/scibert_nlp4sg")

    tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,**tokenizer_kwargs
    )

    data_all=load_data(args['dataset'])
    ## This is to get the id of the paper, as the id is not always called the same
    if "parquet" in args["dataset"]:
        id_str='acl_id'
    else:
        id_str='ID'
    with open("results_task_1.csv", 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['ID', 'title', 'abstract','text','year','nlp4sg_score'])
        for d in data_all:
            text=""
            if d['title']:
                text+=d["title"]
            if d['abstract']:
                text+=". "+d["abstract"]
            output=classifier(text)
            if output[0]['label']=='NLP4SG':
                csv_writer.writerow([d[id_str],d['title'],d['abstract'],text,d['year'],output[0]['score']])

if __name__ == '__main__':
# parse from arguments the dataset to be used
    args=argparse.ArgumentParser()
    args.add_argument("--dataset",type=str,default="feradauto/NLP4SGPapers")
    args=vars(args.parse_args())
    main(args)