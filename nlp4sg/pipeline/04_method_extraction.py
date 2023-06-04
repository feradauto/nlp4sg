import argparse
import time
import os
import pandas as pd
import numpy as np
import openai
import time
openai.api_key = os.getenv("OPENAI_API_KEY")
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


def predict_gpt3():
    data_path="./"
    df=pd.read_csv(data_path+"results_task_1.csv")


    ack_preprompt="""Identify the NLP method(s) used in this paper. Select a text span that is an appropriate answer, or if no span serves as a good answer, just come up with a phrase. Separate the methods with commas and don't include NLP tasks. Examples of methods are: BERT, SVM, CNN, etc."""

    ack_postprompt="""The primary NLP method used in this paper is:"""

    df=df.assign(task_prompt_text=ack_preprompt+"\n"+df.text+"\n"+ack_postprompt)
    for i,d in df.iterrows():
        input_prompt=d['task_prompt_text']
        completion = openai.Completion.create(engine="text-davinci-002", prompt=input_prompt,temperature=0,max_tokens=60,logprobs=1)
        df.loc[i,'GPT3_response']=completion.choices[0].text
        time.sleep(3)

    df=df.assign(clean_response=df.GPT3_response.replace("\n","",regex=True))
    df['clean_response']=df['clean_response'].str.replace(r'\.$', '')

    return df
def get_zero_shot_prediction(model_name):
    data_path="./"
    df=pd.read_csv(data_path+"results_task_1.csv")
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

    for i,d in df.iterrows():
        QA_input = {
        'question': "Which NLP method(s) does this paper use?",
        'context': d['text']
        }
        res = nlp(QA_input)
        df.at[i,'answer']=res['answer']

    df=df.assign(clean_response=df.answer.str.replace(r'\.$', ''))
    return df

def clean_response(df):
    df=df.rename(columns={'clean_response':'method'})
    df.method=df.method.str.lstrip(' ')
    df.method=df.method.apply(lambda x:[x])
    return df


def main(args):
    outputs_path="./"

    if args['model']=='openai':
        df=predict_gpt3()
    else:
        df=get_zero_shot_prediction(args['model'])

    df=clean_response(df)
    df.to_csv(outputs_path+"method_extr_task3.csv",index=False)


if __name__ == '__main__':
    args=argparse.ArgumentParser()
    ## openai if you want to use GPT 3
    args.add_argument("--model",type=str,default="bert-large-uncased-whole-word-masking-finetuned-squad")
    args=vars(args.parse_args())
    main(args)
