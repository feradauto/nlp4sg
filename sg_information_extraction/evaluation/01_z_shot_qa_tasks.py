import pandas as pd
import numpy as np
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

def main():
    data_path="../../data/"
    outputs_path="../../outputs/"
    df=pd.read_csv(outputs_path+"general/test_set_final.csv")

    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)


    df=df.assign(abstract_for_prompt=np.where(df.abstract_clean!="",
                                                       ""+df.abstract_clean,""))
    df=df.assign(abstract_for_prompt=df.abstract_for_prompt.fillna(""))
    df=df.assign(paper_text=""+df.title_clean+"\n"+df.abstract_for_prompt)

    df=df.loc[df.label==1].reset_index(drop=True)

    for i,d in df.iterrows():
        QA_input = {
        'question': "Which NLP task does this paper address?",
        'context': d['paper_text']
        }
        res = nlp(QA_input)
        df.at[i,'answer']=res['answer']

    df=df.assign(clean_response=df.answer.str.replace(r'\.$', ''))

    df.to_csv(outputs_path+"sg_ie/bert_tasks_simple.csv",index=False)
    
if __name__ == '__main__':
    main()
