import pandas as pd
import numpy as np

no_abstract=['korhonen-2002-assigning', 'pusateri-glass-2004-modeling',
       'castillo-etal-2004-talp', 'claveau-sebillot-2004-efficiency','vauquois-etal-1965-syntaxe',
       'webster-1994-building', 'patrick-etal-2002-slinerc','kipper-etal-2004-using',
       'korhonen-preiss-2003-improving', 'hui-2002-measuring','engel-etal-2002-parsing',
       'blache-2003-meta', 'bechet-etal-2000-tagging','purver-etal-2001-means','elenko-etal-2004-coreference',
       'kim-etal-2000-phrase','pirrelli-battista-1996-monotonic','russell-1976-computer',
       'purver-2002-processing','simmons-bennett-novak-1975-semantically',
       'mori-2002-information', 'habash-dorr-2001-large','bilgram-keson-1998-construction',
       'seneff-polifroni-2001-hypothesis', 'johnson-1997-personal','ferragne-etal-2012-rocme'
       'hovy-2002-building', 'schiehlen-2004-annotation','nn-1977-finite-string-volume-14-number-7',
       'zelenko-etal-2002-kernel', 'suzuki-etal-2002-topic','von-glasersfeld-1974-yerkish',
       'murata-etal-2001-using', 'bilac-tanaka-2004-hybrid','tseng-etal-2021-aspect',
       'nightingale-tanaka-2003-comparing','shapiro-1975-generation',
       'yangarber-etal-2002-unsupervised', 'lee-bryant-2002-contextual','lin-etal-2006-information',
       'dong-1990-transtar', 'lager-1998-logic','ipper-etal-2004-using','yamabana-etal-2000-lexicalized',
       'takeuchi-etal-2004-construction', 'freitag-2004-toward','shudo-etal-2000-collocations',
       'ueffing-etal-2002-generation', 'munteanu-etal-2004-improved','hajicova-kucerova-2002-argument',
       'forbes-webber-2002-semantic', 'foret-nir-2002-rigid','moschitti-2010-kernel','von-glasersfeld-1974-yerkish',
            'lin-etal-2006-information',
        'navigli-velardi-2002-automatic']

def process_dataset(df_test_final,anthology):
    """Format test dataset

    Parameters:
    df_test_final (df): Test dataset 
    anthology (df): Dataframe with all papers information
    Returns:
    df_test_final
    """
    df_test_final=df_test_final.rename(columns={"u2 annotation of SG_or_not":"label"})
    df_test_final["label"]=df_test_final["label"].fillna(0)
    df_test_final=df_test_final.loc[:,['ID','url','label','task_annotation','method_annotation','org_annotation','Most Related SG goal',
       '(if exists) 2nd Related SG Goal', '(if exists) 3rd Related SG Goal']]
    anthology=anthology.assign(acknowledgments=anthology.acknowledgments.fillna(''))
    anthology=anthology.assign(abstract=anthology.abstract.fillna(''))
    anthology=anthology.assign(title_clean=anthology.title.replace("{","",regex=True).replace("}","",regex=True))
    anthology=anthology.assign(abstract_clean=anthology.abstract.replace("{","",regex=True).replace("}","",regex=True))
    anthology=anthology.assign(acknowledgments_clean=anthology.acknowledgments.replace("{","",regex=True).replace("}","",regex=True))
    anthology=anthology.assign(title_abstract_clean=anthology.title_clean+". "+anthology.abstract_clean)
    anthology=anthology.loc[:,['ID','title_abstract_clean','title','abstract','title_clean','abstract_clean','acknowledgments_clean']]
    df_test_final=df_test_final.merge(anthology,on=['ID'])
    df_test_final=df_test_final.assign(text=df_test_final.title_abstract_clean)
    df_test_final=df_test_final.loc[~df_test_final.label.isna()]
    df_test_final.label=df_test_final.label.apply(int)
    return df_test_final

def filter_papers(df_test_final,df_lan_all,df_lan):

    ## just english papers
    df_test_final=df_test_final.merge(df_lan,how='left')
    df_test_final=df_test_final.loc[(df_test_final.lan=='en') | (df_test_final.lan.isna()),:]

    df_lan_all=df_lan_all.rename(columns={'lan':'lang_all'})
    df_test_final=df_test_final.merge(df_lan_all,how='left')
    df_test_final=df_test_final.loc[(df_test_final.lang_all=='en') | (df_test_final.lang_all.isna()),:]

    df_test_final=df_test_final.loc[~df_test_final.text.str.lower().str.startswith('book review')]
    df_test_final=df_test_final.loc[~df_test_final.text.str.lower().str.startswith('copyright information')]
    df_test_final=df_test_final.loc[~df_test_final.text.str.lower().str.startswith('session')]
    df_test_final=df_test_final.loc[~df_test_final.text.str.lower().str.startswith('transcript')]

    df_test_final=df_test_final.loc[~df_test_final.text.str.lower().str.startswith('(invited presentation)')]
    df_test_final=df_test_final.loc[~df_test_final.text.str.lower().str.startswith('program of the')]

    df_test_final=df_test_final.loc[~df_test_final.text.str.contains('\[In Chinese\].')]
    df_test_final=df_test_final.loc[~df_test_final.text.str.contains('\[In French\].')]
    df_test_final=df_test_final.loc[~df_test_final.text.str.contains('\[In Portuguese\].')]
    df_test_final=df_test_final.loc[~df_test_final.text.str.contains('\[In Spanish\].')]
    df_test_final=df_test_final.loc[~df_test_final.text.str.contains('Author Index: Volumes')]
    df_test_final=df_test_final.loc[~df_test_final.text.str.contains('Author Index: Volume')]

    df_test_final=df_test_final.loc[~df_test_final.text.isin([
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

    df_test_final=df_test_final.reset_index(drop=True)
    return df_test_final

def main():
    data_path="../../data/"
    outputs_path="../../outputs/"

    df=pd.read_csv(data_path+"test_data/test_set_SG_annotate_5k2_gold_final.csv")
    anthology_test=pd.read_csv(data_path+"test_data/papers_test_set_ack.csv")
    df_lan_all=pd.read_csv(outputs_path+"general/test_set_5k2_languages.csv")
    match_unique=pd.read_csv(outputs_path+"general/papers_uniques.csv")
    df_lan=pd.read_csv(outputs_path+"general/test_set_language.csv")
    
    anthology_test=anthology_test.assign(abstract=np.where(anthology_test.ID.isin(no_abstract),"",anthology_test.abstract))

    df_test_final=process_dataset(df,anthology_test)

    df_year=match_unique.loc[:,['ID','year']]

    df_test_final=df_test_final.merge(df_year,on=['ID'])

    df_test_final=filter_papers(df_test_final,df_lan_all,df_lan)

    df_test_final=df_test_final.loc[:,df_test_final.columns[:-2]]

    df_test_final=df_test_final.rename(columns={'Most Related SG goal':'goal1_raw',
           '(if exists) 2nd Related SG Goal':'goal2_raw', '(if exists) 3rd Related SG Goal':'goal3_raw'})

    test_set=df_test_final.loc[:2087].copy()
    dev_set=df_test_final.loc[2088:2587].copy()
    train_set=df_test_final.loc[2588:].copy()

    test_set.to_csv(outputs_path+"general/test_set_final.csv",index=False)
    train_set.to_csv(outputs_path+"general/train_set_final.csv",index=False)
    dev_set.to_csv(outputs_path+"general/dev_set_final.csv",index=False)    

if __name__ == '__main__':
    main()