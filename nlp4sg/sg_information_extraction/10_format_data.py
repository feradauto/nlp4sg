## Clean text in datasets

import pandas as pd
import numpy as np

def format_data(data):
    data=data.assign(title_clean=data.title.replace("{","",regex=True).replace("}","",regex=True))
    data=data.assign(abstract_clean=data.abstract.replace("{","",regex=True).replace("}","",regex=True).fillna(""))
    data=data.assign(acknowledgments_clean=data.acknowledgments.replace("{","",regex=True).replace("}","",regex=True).fillna(""))

    data=data.assign(abstract_for_prompt=np.where(data.abstract_clean!="",
                                                       "Abstract: "+data.abstract_clean,""))

    data=data.assign(acknowledgments_for_prompt=np.where(data.acknowledgments_clean!="",
                                                       "\nAcknowledgments: "+data.acknowledgments_clean,""))

    data=data.assign(title_abstract_clean=data.title_clean+". "+data.abstract_clean)
    return data

def main():
    data_path="../data/"
    outputs_path="../outputs/"
    df_positive=pd.read_csv(outputs_path+"sg_classifier/all_positive_examples_final.csv")
    acks=pd.read_csv(data_path+"pdfs_data/positives_ack_final_8k_new.csv")
    df_test_ack=pd.read_csv(data_path+"test_data/papers_test_set_ack.csv")
    
    df_positive=df_positive.loc[:,['ID','title','abstract','url']].rename(columns={'abstract':'past_abstract'}).drop_duplicates()
    acks=acks.loc[:,['title','abstract','year','acknowledgments']].drop_duplicates(subset=['title'])
    positives=df_positive.merge(acks,on=['title'],how='left')
    positives=positives.assign(abstract=np.where(positives.abstract.isna(),positives.past_abstract,positives.abstract))

    positives=format_data(positives)
    df_test_ack=format_data(df_test_ack)
    
    df_test_ack.to_csv(outputs_path+"sg_ie/test_ready.csv",index=False)
    positives.to_csv(outputs_path+"sg_ie/positives_ready.csv",index=False)
    
if __name__ == '__main__':
    main()