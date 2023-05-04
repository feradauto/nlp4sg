## Example file evaluating model for task 1 using the test set of our NLP4SGPapers dataset
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification,AutoTokenizer,pipeline
import pandas as pd


def main():

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
    for d in dataset['test']:
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
    main()