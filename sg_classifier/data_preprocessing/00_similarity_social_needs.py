## Measures similarity papers with social needs

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def format_dfs(df,social_needs):
    """format input dataframes

    Parameters:
    df (df): Dataframe with papers information
    social_needs (df): Dataframes with social needs
    Returns:
    Dataframes 

   """
    social_needs_list=social_needs.loc[:,['Goal']]

    df=df.loc[:,['ID','title','abstract','year']]

    ## remove not relevant rows
    repeated=df.title.value_counts().reset_index().rename(columns={'title':'counts','index':'title'})

    repeated=repeated.loc[repeated.counts>2]
    df=df.loc[~df.title.isin(repeated.title.unique())]

    df=df.assign(abstract=df.abstract.fillna(''))

    df=df.assign(title_abstract=df.title+" "+df.abstract)

    df=df.reset_index(drop=True)

    social_needs_melted=pd.melt(social_needs,id_vars=['Goal','Goal_Desc'],value_vars=social_needs.columns[2:],var_name=['Target'],value_name='Target_Desc')

    social_needs_melted=social_needs_melted.loc[~social_needs_melted['Target_Desc'].isna()]

    social_needs=social_needs.loc[:,['Goal','Goal_Desc']]

    social_needs=social_needs.assign(social_need=social_needs.Goal+" "+social_needs.Goal_Desc)

    df=df.reset_index().rename(columns={'index':'row_id'})
    social_needs=social_needs.reset_index().rename(columns={'index':'social_id'})
    
    return (df,social_needs)

def get_similarities(df,social_needs):
    """Get cosine similarities between 2 dfs

    Parameters:
    df (df): Dataframe with papers information
    social_needs (df): Dataframes with social needs
    Returns:
    dataframe with cosine similarity 

   """

    model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')

    embeddings_abstract = model.encode(df.title_abstract)
    embeddings_social = model.encode(social_needs.social_need)

    df_merged=df.merge(social_needs,how="cross")

    for i,d in df_merged.iterrows():
        similarity=cosine_similarity([embeddings_abstract[d['row_id']]],[embeddings_social[d['social_id']]])[0][0]
        df_merged.loc[i,['cosine_similarity']]=similarity
    return df_merged

def get_best_match(df_merged):

    df_merged['words']=df_merged['title_abstract'].str.split().str.len()
    ## remove entries that are not normal papers
    df_merged_filtered=df_merged.loc[df_merged.words>1]
    df_merged_filtered=df_merged_filtered.loc[~df_merged_filtered.title_abstract.str.lower().str.contains('proceedings of|international workshop on|international conference on|international journal of computational|workshop on|summary of discussion|summary of the discussion|title index: volume|minutes of the')]
    df_merged_filtered=df_merged_filtered.loc[~df_merged_filtered.title_abstract.str.lower().str.contains('conference on applied natural language processing|program committee|computational linguistics, volume|call for papers|reports from session chairs|{i}nternational {c}onference on|{u}niversity of {w}ashington presentation')]

    match_unique=df_merged_filtered.sort_values('cosine_similarity',ascending=False).drop_duplicates(subset=['title_abstract'])
    return match_unique

def main():
    data_path="../../data/"
    outputs_path="../../outputs/"
    df=pd.read_csv(data_path+"papers/anthology.csv")
    social_needs=pd.read_csv(data_path+"others/social_needs.csv")
    df,social_needs=format_dfs(df,social_needs)
    df_merged=get_similarities(df,social_needs)
    df_merged.to_csv(outputs_path+"general/social_abstracts_cosine_clean.csv",index=False)
    df_merged=pd.read_csv(outputs_path+"general/social_abstracts_cosine_clean.csv")
    match_unique=get_best_match(df_merged)
    match_unique=match_unique.sample(frac=1,random_state=42)
    match_unique.to_csv(outputs_path+"general/papers_uniques.csv",index=False)

if __name__ == '__main__':
    main()

