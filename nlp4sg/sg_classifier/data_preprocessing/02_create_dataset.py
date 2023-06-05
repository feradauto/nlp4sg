## create random sample from all ACL anthology
import numpy as np
import pandas as pd
import spacy
from spacy_langdetect import LanguageDetector

def generate_random_sample(match_unique,df_url,df_lan):
    """Get random sample (test set) for annotation and training set

    Parameters:
    match_unique (df): Dataframe with papers information
    social_needs (df): Dataframe with papers information
    Returns:
    dataframe for annotation
    """
    def detect_language(text):
        doc = nlp(text)
        return doc._.language['language']
        
    nlp = spacy.load("en")
    nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)
    
    test_set=match_unique.sample(n=1000,random_state=42)
    test_set=test_set.merge(df_url.loc[:,['url','ID']],how='left')
    test_set=test_set.loc[:,['ID','url','title']]
    ## just shuffle
    test_set=test_set.sample(frac=1,random_state=42)
    ## training set
    df_url=df_url.loc[:,['url','ID']]
    train_set=match_unique.loc[~match_unique.ID.isin(test_set.ID.unique())]
    train_set=train_set.loc[~train_set.title.isin(test_set.title.unique())]
    
    ## test set second part
    #train_set=train_set.assign(lan=train_set.title_abstract.apply(lambda x:detect_language(x)))
    
    train_set=train_set.merge(df_lan,on=['ID'],how='left')
    train_set=train_set.loc[train_set.lan=='en',:]
    
    test_set_2000=train_set.sample(n=2030,random_state=42)
    test_set_2000=test_set_2000.sample(frac=1,random_state=42)
    test_set_2000=test_set_2000.merge(df_url.loc[:,['url','ID']],how='left')
    test_set_2000=test_set_2000.loc[:,['ID','url','title']]
    
    train_set=train_set.loc[~train_set.ID.isin(test_set_2000.ID.unique())]
    train_set=train_set.loc[~train_set.title.isin(test_set_2000.title.unique())]
    train_set=train_set.merge(df_url,on=['ID'],how='left')
    
    test_set_2000_plus=train_set.sample(n=2000,random_state=42)
    test_set_2000_plus=test_set_2000_plus.sample(frac=1,random_state=42)
    test_set_2000_plus=test_set_2000_plus.merge(df_url.loc[:,['url','ID']],how='left')
    test_set_2000_plus=test_set_2000_plus.loc[:,['ID','url','title']]
    
    train_set=train_set.loc[~train_set.ID.isin(test_set_2000_plus.ID.unique())]
    train_set=train_set.loc[~train_set.title.isin(test_set_2000_plus.title.unique())]
    
    test_set_200_final=train_set.sample(n=200,random_state=42)
    test_set_200_final=test_set_200_final.sample(frac=1,random_state=42)
    test_set_200_final=test_set_200_final.merge(df_url.loc[:,['url','ID']],how='left')
    test_set_200_final=test_set_200_final.loc[:,['ID','url','title']]
    
    train_set=train_set.loc[~train_set.ID.isin(test_set_200_final.ID.unique())]
    train_set=train_set.loc[~train_set.title.isin(test_set_200_final.title.unique())]
    
    
    test_set=pd.concat([test_set,test_set_2000,test_set_2000_plus,test_set_200_final])
    
    return test_set,train_set

def main():
    data_path="../../data/"
    outputs_path="../../outputs/"
    match_unique=pd.read_csv(outputs_path+"general/papers_uniques.csv")
    df_url=pd.read_csv(data_path+"papers/anthology.csv")
    df_lan=pd.read_csv(outputs_path+"general/language.csv")
    test_set,train_set=generate_random_sample(match_unique,df_url,df_lan)

    test_set.to_csv(outputs_path+"general/dataset_SG.csv",index=False)
    train_set.to_csv(outputs_path+"general/others_SG.csv",index=False)

if __name__ == '__main__':
    main()