import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

def assign_social_need(social_needs,df):

    social_needs_list=social_needs.loc[:,['Goal']]
    social_needs=social_needs.loc[:,['Goal','Goal_Desc']]

    social_needs=social_needs.assign(social_need=social_needs.Goal+" "+social_needs.Goal_Desc)

    model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')

    df=df.reset_index().rename(columns={'index':'row_id'})
    social_needs=social_needs.reset_index().rename(columns={'index':'social_id'})

    embeddings_abstract = model.encode(df.text)
    embeddings_social = model.encode(social_needs.social_need)

    df_merged=pd.merge(df.assign(A=1),social_needs.assign(A=1), on='A').drop('A', 1)

    df_merged['cosine_similarity']=None

    for i,d in df_merged.iterrows():
        similarity=cosine_similarity([embeddings_abstract[d['row_id']]],[embeddings_social[d['social_id']]])[0][0]
        df_merged.loc[i,['cosine_similarity']]=similarity

    match_unique=df_merged.sort_values('cosine_similarity',ascending=False).drop_duplicates(subset=['title_abstract_clean'])

    match_unique.cosine_similarity=pd.to_numeric(match_unique.cosine_similarity)
    
    match_unique=match_unique.loc[:,['ID','title_clean','title_abstract_clean','Goal','year','cosine_similarity']]
    match_unique=match_unique.assign(Goal=np.where(match_unique.cosine_similarity<=0,"Other",match_unique.Goal))
    return match_unique

def get_zero_shot_classification(df):
    candidate_labels = ['No Poverty',
     'No Hunger',
     'Good Health and Well-Being',
     'Quality Education',
     'Gender Equality',
     'Clean Water and Sanitation',
     'Affordable and Clean Energy',
     'Decent Work and Economic Growth',
     'Industry, Innovation and Infrastructure',
     'Reduced Inequalities',
     'Sustainable Cities and Communities',
     'Responsible Consumption and Production',
     'Climate Action',
     'Life Below Water',
     'Life on Land',
     'Peace, Justice and Strong Institutions',
     'Partnership for the Goals',
    'Disinformation and fake news', 'Privacy protection', 'Deception detection','Hate speech']
    descs=['End poverty in all its forms everywhere',
       'End hunger, achieve food security and improved nutrition and promote sustainable agriculture',
       'Ensure healthy lives and promote well-being for all at all ages',
       'Ensure inclusive and equitable quality education and promote lifelong learning opportunities for all',
       'Achieve gender equality and empower all women and girls',
       'Ensure availability and sustainable management of water and sanitation for all',
       'Ensure access to affordable, reliable, sustainable and modern energy for all',
       'Promote sustained, inclusive and sustainable economic growth, full and productive employment and decent work for all',
       'Build resilient infrastructure, promote inclusive and sustainable industrialization and foster innovation',
       'Reduce inequality within and among countries',
       'Make cities and human settlements inclusive, safe, resilient and sustainable',
       'Ensure sustainable consumption and production patterns',
       'Take urgent action to combat climate change and its impacts',
       'Conserve and sustainably use the oceans, seas and marine resources for sustainable development',
       'Protect, restore and promote sustainable use of terrestrial ecosystems, sustainably manage forests, combat desertification, and halt and reverse land degradation and halt biodiversity loss',
       'Promote peaceful and inclusive societies for sustainable development, provide access to justice for all and build effective, accountable and inclusive institutions at all levels',
       'Strengthen the means of implementation and revitalize the global partnership for sustainable development',
        'Fake news is false or misleading information presented as news', 
        'Privacy is the ability of an individual or group to seclude themselves or information about themselves, and thereby express themselves selectively', 
        'Deception is the act or statement that misleads or promotes a belief, concept, or idea that is not true',
        'Public speech that expresses hate or encourages violence towards a person or group based on something such as race, religion, sex, or sexual orientation' ]
    
    
    classifier = pipeline("zero-shot-classification",
                          model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", device=0)

    batch_size = 100 # see how big you can make this number before OOM
    sequences = df['text'].to_list()
    results = []
    for i in range(0, len(sequences), batch_size):
        results += classifier(sequences[i:i+batch_size], descs, multi_class=True)

    for i,d in df.iterrows():
        label_dict={}
        for l,s in zip(results[i]['labels'],results[i]['scores']):
            goal_index = descs.index(l)
            label_dict[candidate_labels[goal_index]]=s
        df.loc[i,'label_complete']=[label_dict]

    probas=df.label_complete.apply(pd.Series)

    df=df.merge(probas,left_index=True,right_index=True)
    return df


def get_classifications(df,social_needs):
    df_goals=df.loc[:,['ID','goal1_raw', 'goal2_raw', 'goal3_raw','goal1', 'goal2', 'goal3']].copy()
    df=df.loc[:,['ID','title_clean','abstract_clean','title_abstract_clean','year','text']]
    match_unique=assign_social_need(social_needs,df)
    df_zero_shot=get_zero_shot_classification(df)
    df_classification=df_goals.merge(df_zero_shot,on=['ID'],how='left').merge(match_unique.loc[:,['ID','Goal','cosine_similarity']],on=['ID'],how='left')
    return df_classification

def main():
    data_path="../data/"
    outputs_path="../outputs/"
    test_set=pd.read_csv(outputs_path+"general/test_set_final.csv")
    train_set=pd.read_csv(outputs_path+"general/train_set_final.csv")
    dev_set=pd.read_csv(outputs_path+"general/dev_set_final.csv")
    low_ocurrence=pd.read_csv(data_path+"test_data/low_occurrence_annotated.csv")


    social_needs=pd.read_csv(data_path+"others/social_needs.csv")
    papers=pd.read_csv(outputs_path+"general/papers_uniques.csv")
    papers=papers.loc[:,['ID','year']]

    low_ocurrence=low_ocurrence.rename(columns={'Most Related SG goal':'goal1_raw',
           '(if exists) 2nd Related SG Goal':'goal2_raw', '(if exists) 3rd Related SG Goal':'goal3_raw'})
    low_ocurrence=low_ocurrence.rename(columns={"SG_or_not":"label"})
    low_ocurrence["label"]=low_ocurrence["label"].fillna(0)
    low_ocurrence.abstract_clean=low_ocurrence.abstract_clean.fillna('')
    low_ocurrence=low_ocurrence.assign(text=low_ocurrence.title_clean+". "+low_ocurrence.abstract_clean)
    low_ocurrence=low_ocurrence.assign(title_abstract_clean=low_ocurrence.text)

    low_ocurrence=low_ocurrence.merge(papers,on=['ID'],how='left')
    df_all_goals=pd.concat([dev_set,train_set,test_set,low_ocurrence])
    df_all_goals.goal1_raw=df_all_goals.goal1_raw.fillna('')
    df_all_goals.goal2_raw=df_all_goals.goal2_raw.fillna('')
    df_all_goals.goal3_raw=df_all_goals.goal3_raw.fillna('')
    df_all_goals=df_all_goals.assign(goal1=np.where(df_all_goals['goal1_raw'].str.lower().str.contains("education"),'Quality Education',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("poverty"),'No Poverty',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("hunger"),'Zero Hunger',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("clean_water"),'Clean Water and Sanitation',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("clean_energy"),'Affordable and Clean Energy',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("life_land"),'Life on Land',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("marine_life"),'Life Below Water',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("health"),'Good Health and Well-Being',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("climate"),'Climate Action',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("peace|privacy|disinformation_and_fake_news|deception|hate"),'Peace, Justice and Strong Institutions',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("social biases|race & identity"),'Reduced Inequalities',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("industry|innovation|research"),'Industry, Innovation and Infrastructure',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("sustainable cities|sustainable_cities"),'Sustainable Cities and Communities',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("gender"),'Gender Equality',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("decent work|decent_work_and_economy"),'Decent Work and Economic Growth',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("partnership"),'Partnership for the goals',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("responsible_consumption_and_production"),'Responsible Consumption and Production',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("reduced|social_equality"),'Reduced Inequalities',''
                              )))))))))))))))))))

    df_all_goals=df_all_goals.assign(goal2=np.where(df_all_goals['goal2_raw'].str.lower().str.contains("education"),'Quality Education',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("poverty"),'No Poverty',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("hunger"),'Zero Hunger',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("clean_water"),'Clean Water and Sanitation',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("clean_energy"),'Affordable and Clean Energy',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("life_land"),'Life on Land',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("marine_life"),'Life Below Water',         
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("health"),'Good Health and Well-Being',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("climate"),'Climate Action',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("peace|privacy|disinformation_and_fake_news|deception|hate"),'Peace, Justice and Strong Institutions',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("social biases|race & identity"),'Reduced Inequalities',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("industry|innovation|research"),'Industry, Innovation and Infrastructure',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("sustainable cities|sustainable_cities"),'Sustainable Cities and Communities',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("gender"),'Gender Equality',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("decent work|decent_work_and_economy"),'Decent Work and Economic Growth',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("partnership"),'Partnership for the goals',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("responsible_consumption_and_production"),'Responsible Consumption and Production',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("reduced|social_equality"),'Reduced Inequalities',''
                              )))))))))))))))))))

    df_all_goals=df_all_goals.assign(goal3=np.where(df_all_goals['goal3_raw'].str.lower().str.contains("education"),'Quality Education',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("poverty"),'No Poverty',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("hunger"),'Zero Hunger',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("clean_water"),'Clean Water and Sanitation',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("clean_energy"),'Affordable and Clean Energy',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("life_land"),'Life on Land',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("marine_life"),'Life Below Water',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("health"),'Good Health and Well-Being',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("climate"),'Climate Action',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("peace|privacy|disinformation_and_fake_news|deception|hate"),'Peace, Justice and Strong Institutions',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("social biases|race & identity"),'Reduced Inequalities',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("industry|innovation|research"),'Industry, Innovation and Infrastructure',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("sustainable cities|sustainable_cities"),'Sustainable Cities and Communities',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("gender"),'Gender Equality',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("decent work|decent_work_and_economy"),'Decent Work and Economic Growth',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("partnership"),'Partnership for the goals',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("responsible_consumption_and_production"),'Responsible Consumption and Production',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("reduced|social_equality"),'Reduced Inequalities',''
                              )))))))))))))))))))

    df_all_goals_sg=df_all_goals.loc[df_all_goals.label==1].reset_index(drop=True)


    df_test_final=get_classifications(df_all_goals_sg,social_needs)

    df_test_final.to_csv(outputs_path+"sg_goals/goal_classifier_test_desc_debertav3.csv",index=False)
if __name__ == '__main__':
    main()
