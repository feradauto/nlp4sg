## Generate extra observations from the SG papers. Only generate for the low occurrence goals

import json
import pandas as pd
import re
import numpy as np


def main():
    outputs="../../outputs/"
    data_path="../../data/"

    positives=pd.read_csv(outputs+"sg_ie/positives_ready.csv")

    keywords=pd.read_csv(data_path+"others/sg_keywords_v7.csv")

    keywords=keywords.loc[(~keywords.Keywords.isna())&(keywords['Concern w.r.t. precision'].isna()),['which_UN_goal','Keywords','python_checker (default = string_match, other_options=nltk.word_tokenize + match; lower_case+remove_non_alphabet+string_match']]
    keywords.columns=['un_goal','keywords','method']
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

    keywords.un_goal=keywords.un_goal.replace("health; gender_equality","health")

    positives=positives.assign(title_search=positives.title_clean.replace("-"," ",regex=True))

    positives=positives.assign(title_search=positives.title_search.apply(lambda x:re.sub('[^a-zA-Z0-9 ]+', '',x)))

    positives=positives.loc[:,['ID','title_clean','abstract_clean','title_search']]


    for k in keywords.un_goal.unique():
        positives[k]=np.where(
        (positives.title_search.apply(lambda x:any(word.startswith(tuple(key_special.loc[key_special.un_goal==k].keywords)) for word in x.lower().split()))) 
        ,1,0)

        positives[k]=positives[k]+np.where(
            (positives.title_search.apply(lambda x: any(word.startswith(tuple(key_not_in.loc[key_not_in.un_goal==k].keywords)) for word in x.lower().split())))
            ,-1,0)

        positives[k]=positives[k]+np.where(
            ((positives.title_search.str.lower().str.contains('|'.join(list(key_contains.loc[key_contains.un_goal==k].keywords.values))))& (key_contains.loc[key_contains.un_goal==k].shape[0]>0)) |
            (positives.title_search.apply(lambda x:any(word.startswith(tuple(key_start.loc[key_start.un_goal==k].keywords)) for word in x.lower().split()))) |
            (positives.title_search.apply(lambda x:any(word in (tuple(key_in.loc[key_in.un_goal==k].keywords)) for word in x.lower().split()))) 
            ,1,0)

    positives_final=positives.copy()

    positives_final=positives_final.assign(Goal=np.where(positives_final.poverty>0,'poverty',
                        np.where(positives_final.health>0,'health',
                        np.where(positives_final.education>0,'education',
                        np.where(positives_final.hate_speech>0,'hate_speech',
                        np.where(positives_final.peace_justice_and_strong_institutions>0,'peace_justice_and_strong_institutions',
                        np.where(positives_final.privacy_protection>0,'privacy_protection',
                        np.where(positives_final.disinformation_and_fake_news>0,'disinformation_and_fake_news',
                        np.where(positives_final.social_equality>0,'social_equality', 
                        np.where(positives_final.hunger>0,'hunger',
                        np.where(positives_final.marine_life>0,'marine_life',
                        np.where(positives_final.clean_water>0,'clean_water',
                        np.where(positives_final.energy>0,'energy',
                        np.where(positives_final.responsible_consumption_and_production>0,'responsible_consumption_and_production',
                        np.where(positives_final.sustainable_cities>0,'sustainable_cities',
                        np.where(positives_final.partnership>0,'partnership',
                        np.where(positives_final.climate>0,'climate',      
                        np.where(positives_final.gender_equality>0,'gender_equality',      
                        np.where(positives_final.industry_innovation_infrastructure>0,'industry_innovation_infrastructure',
                        np.where(positives_final.decent_work_and_economy>0,'decent_work_and_economy',''

                        ))))))))))))))))))))

    positives_final=positives_final.loc[~positives_final.title_clean.str.lower().str.startswith('book review')]
    positives_final=positives_final.loc[~positives_final.title_clean.str.lower().str.startswith('copyright information')]
    positives_final=positives_final.loc[~positives_final.title_clean.str.lower().str.startswith('session')]
    positives_final=positives_final.loc[~positives_final.title_clean.str.lower().str.startswith('transcript')]

    positives_final=positives_final.loc[~positives_final.title_clean.str.lower().str.startswith('(invited presentation)')]
    positives_final=positives_final.loc[~positives_final.title_clean.str.lower().str.startswith('program of the')]

    positives_final=positives_final.loc[~positives_final.title_clean.str.contains('\[In Chinese\].')]
    positives_final=positives_final.loc[~positives_final.title_clean.str.contains('\[In French\].')]
    positives_final=positives_final.loc[~positives_final.title_clean.str.contains('\[In Portuguese\].')]
    positives_final=positives_final.loc[~positives_final.title_clean.str.contains('\[In Spanish\].')]
    positives_final=positives_final.loc[~positives_final.title_clean.str.contains('Author Index: Volumes')]
    positives_final=positives_final.loc[~positives_final.title_clean.str.contains('Author Index: Volume')]

    positives_final=positives_final.loc[~positives_final.title_clean.isin([
        "語料庫為本的語義訊息抽取與辨析以近義詞研究為例 (Synonym Discrimination Based on Corpus) [In Chinese]. ",
        "台灣共通語言 (Taiwan Common Language) [In Chinese]. ",
        "American Journal of Computational Linguistics (February 1976). ",
        "Author Index: Volumes 6-19. ",
        "基於訊息配對相似度估計的聊天記錄解構(Chat Log Disentanglement based on Message Pair Similarity Estimation). ",
        "基於BERT模型之多國語言機器閱讀理解研究(Multilingual Machine Reading Comprehension based on BERT Model). ",
        "An Introduction to MT. ",
        "Author and Keyword Index. ",
        "CITAC Computer, Inc.. ",
        "American Journal of Computational Linguistics (September 1975). ",
        "ACL in 1977. ",
        "Summary of Session 7 -- Natural Language (Part 2). ",
        "25th Annual Meeting of the Association for Computational Linguistics. "
    ])]

    df_100_n_all=positives_final.loc[positives_final.Goal.isin(['hunger','poverty','gender_equality','clean_water','energy','decent_work_and_economy','sustainable_cities','responsible_consumption_and_production','climate','life_on_land','marine_life','partnership'])]

    df_100_n=pd.concat([df_100_n_all])

    goal_keywords=positives_final.Goal.value_counts().reset_index().rename(columns={'index':'Goal','Goal':'ocurrences'})

    df_100_n=df_100_n.merge(goal_keywords,on=['Goal'],how='left').sort_values(['ocurrences','Goal'])

    df_100_n=df_100_n.loc[:,['ID','title_clean','abstract_clean','Goal']]

    positives_r=pd.read_csv(outputs+"sg_ie/positives_ready.csv")

    positives_r=positives_r.loc[:,['ID','url','acknowledgments_clean']]

    df_100_n=df_100_n.merge(positives_r,on=['ID'],how='left')

    df_100_n=df_100_n.loc[:,['ID',  'url','title_clean', 'abstract_clean','acknowledgments_clean', 'Goal']]

    df_100_n.to_csv(outputs+"sg_classifier/low_occurrence.csv",index=False)
if __name__ == '__main__':
    main()