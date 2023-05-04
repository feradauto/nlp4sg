import os
import sys
import openai
import pandas as pd
import numpy as np
import argparse

def main():
    data_path="./"
    outputs_path="./"
    test_set=pd.read_csv(outputs_path+"results_task_1.csv")

    openai.api_key = os.getenv("OPENAI_API_KEY")

    preprompt="There is an NLP paper with the title and abstract:\n"
    question="Which of the UN goals does this paper directly contribute to? Provide the goal number and name."
    df=df.assign(statement=preprompt+df.text+"\n"+question)

    for i,d in df.iterrows():
        input_prompt=d['statement']
        completion = openai.Completion.create(engine="text-davinci-002", prompt=input_prompt,temperature=0,max_tokens=100,logprobs=1)
        dict_norm={}
        dict_uniques={}

        df.loc[i,'full_prompt']=input_prompt
        df.loc[i,'GPT3_response']=completion.choices[0].text


    df.to_csv(outputs_path+"results_task_2.csv",index=False)


if __name__ == '__main__':
    main()
