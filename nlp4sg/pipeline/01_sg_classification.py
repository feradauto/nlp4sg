## Example file evaluating model for task 1 using the test set of our NLP4SGPapers dataset
import argparse
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification,AutoTokenizer,pipeline
import pandas as pd

# function that takes a parameter and yields the dataset loaded from huggingface or from json file
# if the dataset is not available in huggingface
def load_data(dataset):
    print(dataset)
    if dataset != "feradauto/NLP4SGPapers":
        return load_dataset("json", data_files={"test": dataset})['test']
    else:
        return load_dataset(dataset)['test']
  

def main(args):

    dataset = load_dataset("feradauto/NLP4SGPapers")
    tokenizer = AutoTokenizer.from_pretrained("feradauto/scibert_nlp4sg",truncation=True)
    model = AutoModelForSequenceClassification.from_pretrained("feradauto/scibert_nlp4sg")

    tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,**tokenizer_kwargs
    )

    data=[]
    data_all=load_data(args['dataset'])
    for d in data_all:
        text=""
        if d['title']:
            text+=d["title"]
        if d['abstract']:
            text+=". "+d["abstract"]
        output=classifier(text)
        if output[0]['label']=='NLP4SG':
            data.append((d['ID'],d['title'],d['abstract'],text,d['year'],output[0]['score']))

    df = pd.DataFrame(data, columns =['ID', 'title', 'abstract','text','year','nlp4sg_score'])
    df.to_csv("results_task_1.csv",index=False)

if __name__ == '__main__':
# parse from arguments the dataset to be used
    args=argparse.ArgumentParser()
    args.add_argument("--dataset",type=str,default="feradauto/NLP4SGPapers")
    args=vars(args.parse_args())
    main(args)