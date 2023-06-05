# Format inputs for SCIREX model

### Libraries and variables

import pandas as pd
import numpy as np

### Process data

def format_scirex(df):
    df=df.assign(abstract_clean=df.abstract_clean.fillna(""))
    df=df.loc[:,['ID','title_clean','abstract_clean']]

    df=df.assign(text=df.title_clean+". "+df.abstract_clean)

    df.loc[:,['text']]=df.loc[:,['text']].replace(",", " , ",regex=True).replace("\.", " . ",regex=True).replace(":", " : ",regex=True).replace("-", " - ",regex=True).replace("  ", " ",regex=True)

    df=df.assign(words=df.text.str.split())

    df_words=df.loc[:,['ID','words']].rename(columns={'ID':'doc_id'})

    par_indexes=[]
    all_sentences=[]
    for i,d in df_words.iterrows():
        sent_indexes=[]
        sentence_index=[0]
        for j in range(1,len(d['words'])):
            if d['words'][j] == '.':
                sentence_index.append(j)
                sent_indexes.append(sentence_index)
                sentence_index=[j]
        ##last
        sentence_index.append(j)
        sent_indexes.append(sentence_index)
        ##
        all_sentences.append(sent_indexes)
        par_indexes.append([[0,j]])

    df_words=df_words.assign(sections=par_indexes)

    df_words=df_words.assign(sentences=all_sentences)
    return df_words

def main():
    data_path="../../data/"
    output_path="../../outputs/"

    df=pd.read_csv(output_path+"sg_ie/positives_ready.csv")
    test_set_complete=pd.read_csv(output_path+"general/test_set_final.csv")
    train_set=pd.read_csv(output_path+"general/train_set_final.csv")
    dev_set=pd.read_csv(output_path+"general/dev_set_final.csv")
    df_test=pd.concat([test_set_complete,dev_set,train_set]).reset_index(drop=True)
    
    df_words=format_scirex(df)
    df_test_words=format_scirex(df_test)

    #df_words.to_json(output_path+"sg_ie/sg_papers_scirex_final_2.jsonl",orient="records",lines=True)

    df_test_words.to_json(output_path+"sg_ie/sg_papers_scirex_test_final_3.jsonl",orient="records",lines=True)
if __name__ == '__main__':
    main()

