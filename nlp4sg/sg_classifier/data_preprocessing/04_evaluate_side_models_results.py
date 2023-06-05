## Evaluate data augmentation techniques on train set

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import re
PERC99=0.22449437871575342
MEDIAN=0.076746

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


def keyword_search(df_merged,keywords):
    key_start,key_contains,key_in,key_not_in,key_special=get_keyword_types(keywords)
    ## title abstract
    df_merged=df_merged.assign(title_abstract_search=df_merged.title_abstract_clean.replace("-"," ",regex=True))
    df_merged=df_merged.assign(title_abstract_search=df_merged.title_abstract_search.apply(lambda x:re.sub('[^a-zA-Z0-9 ]+', '',x)))
    # title
    df_test_final.title=df_test_final.title.replace("{","",regex=True).replace("}","",regex=True)
    df_merged=df_merged.assign(title_search=df_merged.title.replace("-"," ",regex=True))
    df_merged=df_merged.assign(title_search=df_merged.title_search.apply(lambda x:re.sub('[^a-zA-Z0-9 ]+', '',x)))

    ## keywords
    df_merged=df_merged.assign(keyword_pred=np.where(
        (df_merged.title_abstract_search.apply(lambda x:any(word.startswith(tuple(key_special.keywords)) for word in x.lower().split()))) 
        ,1,0))

    df_merged=df_merged.assign(keyword_pred=np.where(
        (df_merged.title_abstract_search.apply(lambda x: any(word.startswith(tuple(key_not_in.keywords)) for word in x.lower().split())))
        ,0,df_merged.keyword_pred))

    df_merged=df_merged.assign(keyword_pred=np.where(
        (df_merged.title_abstract_search.str.lower().str.contains('|'.join(list(key_contains.keywords.values)))) |
        (df_merged.title_abstract_search.apply(lambda x:any(word.startswith(tuple(key_start.keywords)) for word in x.lower().split()))) |
        (df_merged.title_abstract_search.apply(lambda x:any(word in (tuple(key_in.keywords)) for word in x.lower().split()))) 
        ,1,df_merged.keyword_pred))

    ## keywords
    df_merged=df_merged.assign(keyword_title_pred=np.where(
        (df_merged.title.apply(lambda x:any(word.startswith(tuple(key_special.keywords)) for word in x.lower().split()))) 
        ,1,0))

    df_merged=df_merged.assign(keyword_title_pred=np.where(
        (df_merged.title.apply(lambda x: any(word.startswith(tuple(key_not_in.keywords)) for word in x.lower().split())))
        ,0,df_merged.keyword_title_pred))

    df_merged=df_merged.assign(keyword_title_pred=np.where(
        (df_merged.title.str.lower().str.contains('|'.join(list(key_contains.keywords.values)))) |
        (df_merged.title.apply(lambda x:any(word.startswith(tuple(key_start.keywords)) for word in x.lower().split()))) |
        (df_merged.title.apply(lambda x:any(word in (tuple(key_in.keywords)) for word in x.lower().split()))) 
        ,1,df_merged.keyword_title_pred))
    
    return df_merged

def combinations(df_merged):

    df_merged=df_merged.assign(pred_combined=np.where(df_merged.workshop_pred==1,1,
                                np.where(df_merged.keyword_pred==1,1,
                                np.where(df_merged.similarity_pos_pred==1,1,
                                np.where(df_merged.similarity_neg_pred,0,2)))))

    df_merged=df_merged.assign(pred_combined_title=np.where(df_merged.workshop_pred==1,1,
                                np.where(df_merged.keyword_title_pred==1,1,
                                np.where(df_merged.similarity_pos_pred==1,1,
                                np.where(df_merged.similarity_neg_pred,0,2)))))

    df_merged=df_merged.assign(pred_combined_title2=np.where(df_merged.keyword_title_pred==1,1,
                                np.where(df_merged.workshop_pred==1,1,
                                np.where(df_merged.similarity_pos_pred==1,1,
                                np.where(df_merged.similarity_neg_pred,0,2)))))

    df_merged=df_merged.assign(pred_similarities=np.where(df_merged.similarity_pos_pred==1,1,
                                np.where(df_merged.similarity_neg_pred,0,2)))

    df_merged=df_merged.assign(pred_keyword_t_workshop=np.where(df_merged.workshop_pred==1,1,
                                np.where(df_merged.keyword_title_pred==1,1,0)))

    df_merged=df_merged.assign(pred_keyword_workshop=np.where(df_merged.workshop_pred==1,1,
                                np.where(df_merged.keyword_pred==1,1,0)))

    df_merged=df_merged.assign(pred_combined_title_final=np.where(df_merged.keyword_title_pred==1,1,
                                np.where(df_merged.similarity_pos_pred==1,1,
                                np.where(df_merged.similarity_neg_pred,0,2))))
    return df_merged

def main():
    data_path="../../data/"
    outputs_path="../../outputs/"
    df_test_final=pd.read_csv(outputs_path+"general/train_set_final.csv")
    workshops=pd.read_csv(data_path+"others/sg_workshops_v3.csv")
    keywords=pd.read_csv(data_path+"others/sg_keywords_v7.csv")
    match_unique=pd.read_csv(outputs_path+"general/papers_uniques.csv")

    workshops=workshops.loc[(workshops.SG_or_not==1)].reset_index(drop=True)
    match_unique=match_unique.loc[:,['ID','cosine_similarity']]
    df_merged=df_test_final.merge(match_unique,how='left',on='ID')

    # similarity
    df_merged=df_merged.assign(similarity_neg_pred=np.where(df_merged.cosine_similarity<=MEDIAN,0,1))
    df_merged=df_merged.assign(similarity_pos_pred=np.where(df_merged.cosine_similarity>=PERC99,1,0))

    ## workshop
    df_merged=df_merged.assign(workshop_pred=np.where((df_merged.url.str.lower().str.contains('|'.join(list(workshops.event.values)))),1,0))

    ## keywords
    df_merged=keyword_search(df_merged,keywords)

    ## combine techniques
    df_merged=combinations(df_merged)

    #workshops
    print(classification_report(df_merged.label,df_merged.workshop_pred,digits=4))


    #keyword title
    print(classification_report(df_merged.label,df_merged.keyword_title_pred,digits=4))

    #keyword
    print(classification_report(df_merged.label,df_merged.keyword_pred,digits=4))

    #cos_sim pos
    print(classification_report(df_merged.label,df_merged.similarity_pos_pred,digits=4))

    #cos_sim neg
    print(classification_report(df_merged.label,df_merged.similarity_neg_pred,digits=4))

    #similarities
    print(classification_report(df_merged.loc[df_merged.pred_similarities.isin([0,1])].label,
                                df_merged.loc[df_merged.pred_similarities.isin([0,1])].pred_similarities,digits=4))

    ## combinations

    #combined no workshops
    print(classification_report(df_merged.loc[df_merged.pred_combined_title_final.isin([0,1])].label,
                                df_merged.loc[df_merged.pred_combined_title_final.isin([0,1])].pred_combined_title_final,digits=4))

    #combined title
    print(classification_report(df_merged.loc[df_merged.pred_combined_title.isin([0,1])].label,
                                df_merged.loc[df_merged.pred_combined_title.isin([0,1])].pred_combined_title,digits=4))

    #combined keyword abstract
    print(classification_report(df_merged.loc[df_merged.pred_combined.isin([0,1])].label,
                                df_merged.loc[df_merged.pred_combined.isin([0,1])].pred_combined,digits=4))
if __name__ == '__main__':
    main()

