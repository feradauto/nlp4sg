import numpy as np
import pandas as pd
import re
POS_PROPORTION=15
TRAIN_SET_SIZE=30000

def add_labeled_data(df_test_final,train_set,test_set):


    train_set=train_set.assign(abstract=train_set.abstract.fillna(''))
    train_set=train_set.assign(title_abstract=train_set.title+". "+train_set.abstract)
    train_set.title_abstract=train_set.title_abstract.replace("{","",regex=True).replace("}","",regex=True)

    ## website labeled positive examples

    df_test_final=df_test_final.assign(title_abstract=df_test_final.title_abstract_clean)

    ## concat website
    train_set=pd.concat([train_set,df_test_final])
    return train_set

def get_keyword_types(keywords):
    ## keywords
    keywords=keywords.loc[(~keywords.Keywords.isna())&(keywords['Concern w.r.t. precision'].isna()),['Keywords','python_checker (default = string_match, other_options=nltk.word_tokenize + match; lower_case+remove_non_alphabet+string_match']]
    keywords.columns=['keywords','method']
    keywords.keywords=keywords.keywords.str.lower()
    keywords=keywords.assign(extraction_method=np.where(keywords.method.isna(),'contains',
                                            np.where(keywords.method.str.lower().str.contains('exclude'),'start_special',
                                            np.where(keywords.method.str.lower().str.contains('not'),'not_in',
                                            np.where(keywords.method.str.lower().str.contains('starts'),'start','in')))))
    keywords=keywords.assign(keywords=keywords.keywords.replace("-"," ",regex=True))
    keywords=keywords.assign(keywords=keywords.keywords.apply(lambda x:re.sub('[^a-zA-Z0-9 ]+', '',x)))
    key_start=keywords.loc[keywords.extraction_method=='start']
    key_contains=keywords.loc[keywords.extraction_method=='contains']
    key_in=keywords.loc[keywords.extraction_method=='in']
    key_not_in=keywords.loc[keywords.extraction_method=='not_in']
    key_special=keywords.loc[keywords.extraction_method=='start_special']
    return (key_start,key_contains,key_in,key_not_in,key_special)


def keyword_search(train_set,keywords):

    key_start,key_contains,key_in,key_not_in,key_special=get_keyword_types(keywords)
    train_set=train_set.assign(title_abstract_search=train_set.title_clean.replace("-"," ",regex=True))
    train_set=train_set.assign(title_abstract_search=train_set.title_abstract_search.apply(lambda x:re.sub('[^a-zA-Z0-9 ]+', '',x)))

    ## keywords
    train_set=train_set.assign(silver_pos=np.where(
        (train_set.title_abstract_search.apply(lambda x:any(word.startswith(tuple(key_special.keywords)) for word in x.lower().split()))) 
        ,1,0))

    train_set=train_set.assign(silver_pos=np.where(
        (train_set.title_abstract_search.apply(lambda x: any(word.startswith(tuple(key_not_in.keywords)) for word in x.lower().split())))
        ,0,train_set.silver_pos))

    train_set=train_set.assign(silver_pos=np.where(
        (train_set.title_abstract_search.str.lower().str.contains('|'.join(list(key_contains.keywords.values)))) |
        (train_set.title_abstract_search.apply(lambda x:any(word.startswith(tuple(key_start.keywords)) for word in x.lower().split()))) |
        (train_set.title_abstract_search.apply(lambda x:any(word in (tuple(key_in.keywords)) for word in x.lower().split()))) 
        ,1,train_set.silver_pos))

    return train_set

def create_augmented_set(train_set):
    train_set_positive=train_set.loc[
        (train_set.label==1)
        ,:]

    train_set_negative=train_set.loc[
        (train_set.label==0)
        ,:]

    train_set_positive=train_set_positive.loc[:,['ID','title','abstract','title_abstract','label','year','url']]
    train_set_negative=train_set_negative.loc[:,['ID','title','abstract','title_abstract','label','year','url']]


    ## fix the proportion of positive and negative examples
    obs_total=TRAIN_SET_SIZE
    pct=POS_PROPORTION
    pos_obs=int(obs_total*pct/100)
    neg_obs=int(obs_total*(100-pct)/(100))

    pos_obs=pos_obs-train_set.loc[(train_set.label==1)&(train_set.gold==1)].shape[0]
    neg_obs=neg_obs-train_set.loc[(train_set.label==0)&(train_set.gold==1)].shape[0]

    ## always select the gold data and sample from silver data
    positive_gold=train_set.loc[(train_set.label==1)&(train_set.gold==1)]
    positive_silver=train_set.loc[(train_set.label==1)&(train_set.gold==0)]
    negative_gold=train_set.loc[(train_set.label==0)&(train_set.gold==1)]
    negative_silver=train_set.loc[(train_set.label==0)&(train_set.gold==0)]

    positive_silver_sample=positive_silver.sample(n=pos_obs,random_state=42)
    negative_silver_sample=negative_silver.sample(n=neg_obs,random_state=42)
    train_set_final=pd.concat([positive_gold,negative_gold,positive_silver_sample,negative_silver_sample]).sample(frac=1,random_state=42)

    train_set_final=train_set_final.reset_index(drop=True)
    train_set_final.label=train_set_final.label.apply(int)
    return train_set_final

def main():
    data_path="../../data/"
    outputs_path="../../outputs/"
    train_set=pd.read_csv(outputs_path+"general/others_SG.csv")
    test_set=pd.read_csv(data_path+"test_data/test_set_SG_annotate_5k2_gold_final.csv")
    df_test_final=pd.read_csv(outputs_path+"general/train_set_final.csv")
    ## help for filtering positive examples
    workshops=pd.read_csv(data_path+"others/sg_workshops_v3.csv")
    keywords=pd.read_csv(data_path+"others/sg_keywords_v7.csv")
    ## labeled positive examples

    test_set=test_set.assign(title_clean=test_set.title.replace("{","",regex=True).replace("}","",regex=True))

    train_set=add_labeled_data(df_test_final,train_set,test_set,)

    train_set['title_clean']=train_set.title.replace("{","",regex=True).replace("}","",regex=True)

    train_set=train_set.assign(gold=np.where(~(train_set.label.isna()),1,0))

    train_set=keyword_search(train_set,keywords)

    #workshops comment to omit workshops
    #train_set=train_set.assign(silver_pos=np.where((train_set.label.isna())&(train_set.url.str.lower().str.contains('|'.join(list(workshops.event.values)))),1,train_set.silver_pos))
    #train_set.label.value_counts()

    train_set=train_set.assign(label=np.where((train_set.gold==0),train_set.silver_pos,train_set.label))

    train_set_final=create_augmented_set(train_set)

    train_set_final.to_csv(outputs_path+"sg_classifier/train_set_labeled_silver_keyword_ds.csv",index=False)

if __name__ == '__main__':
    main()