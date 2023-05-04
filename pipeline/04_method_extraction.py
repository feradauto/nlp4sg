import time
import os
import pandas as pd
import numpy as np
import openai
import time
openai.api_key = os.getenv("OPENAI_API_KEY")

def main():
    data_path="./"
    output_path="./"
    df=pd.read_csv(output_path+"results_task_1.csv")


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

    df.to_csv(output_path+"method_extr_task3.csv",index=False)


if __name__ == '__main__':
    main()
