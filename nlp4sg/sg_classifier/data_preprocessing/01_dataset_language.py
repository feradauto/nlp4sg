import numpy as np
import pandas as pd
import spacy
from spacy_langdetect import LanguageDetector
from langdetect import DetectorFactory

def get_language(df):
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
    DetectorFactory.seed = 42 
    nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)
    
    ## test set second part
    df=df.assign(lan=df.title_abstract_clean.apply(lambda x:detect_language(x)))
    
    df=df.loc[:,['ID','lan']]
    
    return df



def main():
    data_path="../../data/"
    outputs_path="../../outputs/"
    anthology=pd.read_csv(data_path+"test_data/papers_test_set_ack.csv")
    anthology=anthology.assign(abstract=np.where(anthology.processed_abstract==1,"",anthology.abstract))
    anthology=anthology.assign(abstract=anthology.abstract.fillna(''))
    anthology=anthology.assign(title_clean=anthology.title.replace("{","",regex=True).replace("}","",regex=True))
    anthology=anthology.assign(abstract_clean=anthology.abstract.replace("{","",regex=True).replace("}","",regex=True))
    anthology=anthology.assign(title_abstract_clean=anthology.title_clean+". "+anthology.abstract_clean)

    df_lan=get_language(anthology)

    df_lan=df_lan.assign(lan=np.where(df_lan.ID.isin(['leinonen-etal-2018-new','blanchon-2002-pattern']),'en',df_lan.lan))

    df_lan.to_csv(outputs_path+"general/test_set_SG_annotate_5k2_gold_final.csv",index=False)
    
if __name__ == '__main__':
    main()