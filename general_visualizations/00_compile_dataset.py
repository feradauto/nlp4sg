import pandas as pd
import numpy as np

def assign_goals(df_all_goals):

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
    return df_all_goals


def get_annotations_task3(gold_task3):
    obs=[]
    for i,d in gold_task3.iterrows():
        response={}
        response['task']=[]
        response['method']=[]
        for e in d['annotations'][0]['result']:
            if e['from_name']=='label':
                if e['value']['labels'][0] in ['task']:
                    response['task'].append(e['value']['text'])
                elif e['value']['labels'][0] in ['method']:
                    response['method'].append(e['value']['text'])
                elif e['value']['labels'][0]=='SG_element_in_task':
                    response['SG_element_in_task'].append(e['value']['text'])
            elif e['from_name']=='method_c':
                response['method'].append(e['value']['text'][0])
            elif e['from_name']=='task_c':
                response['task'].append(e['value']['text'][0])
        response['ID']=d['data']['ID']
        obs.append(response)

    df_task3=pd.DataFrame(obs)
    return df_task3

def get_dummy_sdg(df):
    df=df.assign(sdg1=np.where(df['goal1'].str.lower().str.contains("goal 1 |goal 1:|poverty"),1,0))
    df=df.assign(sdg2=np.where(df['goal1'].str.lower().str.contains("goal 2|hunger"),1,0))
    df=df.assign(sdg3=np.where(df['goal1'].str.lower().str.contains("goal 3|health"),1,0))
    df=df.assign(sdg4=np.where(df['goal1'].str.lower().str.contains("goal 4|education"),1,0))
    df=df.assign(sdg5=np.where(df['goal1'].str.lower().str.contains("goal 5|gender"),1,0))
    df=df.assign(sdg6=np.where(df['goal1'].str.lower().str.contains("goal 6|clean water"),1,0))
    df=df.assign(sdg7=np.where(df['goal1'].str.lower().str.contains("goal 7|clean energy"),1,0))
    df=df.assign(sdg8=np.where(df['goal1'].str.lower().str.contains("goal 8|decent work"),1,0))
    df=df.assign(sdg9=np.where(df['goal1'].str.lower().str.contains("goal 9|industry|innovation"),1,0))
    df=df.assign(sdg10=np.where(df['goal1'].str.lower().str.contains("goal 10|inequal"),1,0))
    df=df.assign(sdg11=np.where(df['goal1'].str.lower().str.contains("goal 11|sustainable cities"),1,0))
    df=df.assign(sdg12=np.where(df['goal1'].str.lower().str.contains("goal 12|responsible consumption"),1,0))
    df=df.assign(sdg13=np.where(df['goal1'].str.lower().str.contains("goal 13|climate"),1,0))
    df=df.assign(sdg14=np.where(df['goal1'].str.lower().str.contains("goal 14|life below water"),1,0))
    df=df.assign(sdg15=np.where(df['goal1'].str.lower().str.contains("goal 15|life on land"),1,0))
    df=df.assign(sdg16=np.where(df['goal1'].str.lower().str.contains("goal 16|peace|justice"),1,0))
    df=df.assign(sdg17=np.where(df['goal1'].str.lower().str.contains("goal 17|partnership"),1,0))

    df=df.assign(sdg1=np.where(df['goal2'].str.lower().str.contains("goal 1 |goal 1:|poverty"),1,df.sdg1))
    df=df.assign(sdg2=np.where(df['goal2'].str.lower().str.contains("goal 2|hunger"),1,df.sdg2))
    df=df.assign(sdg3=np.where(df['goal2'].str.lower().str.contains("goal 3|health"),1,df.sdg3))
    df=df.assign(sdg4=np.where(df['goal2'].str.lower().str.contains("goal 4|education"),1,df.sdg4))
    df=df.assign(sdg5=np.where(df['goal2'].str.lower().str.contains("goal 5|gender"),1,df.sdg5))
    df=df.assign(sdg6=np.where(df['goal2'].str.lower().str.contains("goal 6|clean water"),1,df.sdg6))
    df=df.assign(sdg7=np.where(df['goal2'].str.lower().str.contains("goal 7|clean energy"),1,df.sdg7))
    df=df.assign(sdg8=np.where(df['goal2'].str.lower().str.contains("goal 8|decent work"),1,df.sdg8))
    df=df.assign(sdg9=np.where(df['goal2'].str.lower().str.contains("goal 9|industry|innovation"),1,df.sdg9))
    df=df.assign(sdg10=np.where(df['goal2'].str.lower().str.contains("goal 10|inequal"),1,df.sdg10))
    df=df.assign(sdg11=np.where(df['goal2'].str.lower().str.contains("goal 11|sustainable cities"),1,df.sdg11))
    df=df.assign(sdg12=np.where(df['goal2'].str.lower().str.contains("goal 12|responsible consumption"),1,df.sdg12))
    df=df.assign(sdg13=np.where(df['goal2'].str.lower().str.contains("goal 13|climate"),1,df.sdg13))
    df=df.assign(sdg14=np.where(df['goal2'].str.lower().str.contains("goal 14|life below water"),1,df.sdg14))
    df=df.assign(sdg15=np.where(df['goal2'].str.lower().str.contains("goal 15|life on land"),1,df.sdg15))
    df=df.assign(sdg16=np.where(df['goal2'].str.lower().str.contains("goal 16|peace|justice"),1,df.sdg16))
    df=df.assign(sdg17=np.where(df['goal2'].str.lower().str.contains("goal 17|partnership"),1,df.sdg17))

    df=df.assign(sdg1=np.where(df['goal3'].str.lower().str.contains("goal 1 |goal 1:|poverty"),1,df.sdg1))
    df=df.assign(sdg2=np.where(df['goal3'].str.lower().str.contains("goal 2|hunger"),1,df.sdg2))
    df=df.assign(sdg3=np.where(df['goal3'].str.lower().str.contains("goal 3|health"),1,df.sdg3))
    df=df.assign(sdg4=np.where(df['goal3'].str.lower().str.contains("goal 4|education"),1,df.sdg4))
    df=df.assign(sdg5=np.where(df['goal3'].str.lower().str.contains("goal 5|gender"),1,df.sdg5))
    df=df.assign(sdg6=np.where(df['goal3'].str.lower().str.contains("goal 6|clean water"),1,df.sdg6))
    df=df.assign(sdg7=np.where(df['goal3'].str.lower().str.contains("goal 7|clean energy"),1,df.sdg7))
    df=df.assign(sdg8=np.where(df['goal3'].str.lower().str.contains("goal 8|decent work"),1,df.sdg8))
    df=df.assign(sdg9=np.where(df['goal3'].str.lower().str.contains("goal 9|industry|innovation"),1,df.sdg9))
    df=df.assign(sdg10=np.where(df['goal3'].str.lower().str.contains("goal 10|inequal"),1,df.sdg10))
    df=df.assign(sdg11=np.where(df['goal3'].str.lower().str.contains("goal 11|sustainable cities"),1,df.sdg11))
    df=df.assign(sdg12=np.where(df['goal3'].str.lower().str.contains("goal 12|responsible consumption"),1,df.sdg12))
    df=df.assign(sdg13=np.where(df['goal3'].str.lower().str.contains("goal 13|climate"),1,df.sdg13))
    df=df.assign(sdg14=np.where(df['goal3'].str.lower().str.contains("goal 14|life below water"),1,df.sdg14))
    df=df.assign(sdg15=np.where(df['goal3'].str.lower().str.contains("goal 15|life on land"),1,df.sdg15))
    df=df.assign(sdg16=np.where(df['goal3'].str.lower().str.contains("goal 16|peace|justice"),1,df.sdg16))
    df=df.assign(sdg17=np.where(df['goal3'].str.lower().str.contains("goal 17|partnership"),1,df.sdg17))
    return df


def main():
    data_path="../data/"
    outputs_path="../outputs/"
    test_set=pd.read_csv(outputs_path+"general/test_set_final.csv")
    dev_set=pd.read_csv(outputs_path+"general/dev_set_final.csv")
    train_set=pd.read_csv(outputs_path+"general/train_set_final.csv")
    low_ocurrence=pd.read_csv(data_path+"test_data/low_occurrence_annotated.csv")
    gold_task3=pd.read_json(outputs_path+"sg_ie/final_annotation_user2.json")
    test_set=test_set.assign(test=1)
    train_set=train_set.assign(test=0)
    dev_set=dev_set.assign(test=2)
    test_set=test_set.loc[:1999]
    df_all_goals=pd.concat([train_set,dev_set,test_set])
    df_all_goals=df_all_goals.assign(goal1_raw=np.where(df_all_goals.label==0,'',df_all_goals.goal1_raw))
    df_all_goals=df_all_goals.assign(goal2_raw=np.where(df_all_goals.label==0,'',df_all_goals.goal2_raw))
    df_all_goals=df_all_goals.assign(goal3_raw=np.where(df_all_goals.label==0,'',df_all_goals.goal3_raw))

    df_all_goals=assign_goals(df_all_goals)


    df_all_goals=df_all_goals.loc[:,['ID', 'url', 'title_clean',
       'abstract_clean','label', 'goal1','goal2', 'goal3', 'acknowledgments_clean', 'year','test']]
    df_task3=get_annotations_task3(gold_task3)
    df_all_goals=df_all_goals.merge(df_task3,on=['ID'],how='left')
    df_final=df_all_goals.loc[:,['ID', 'url', 'title_clean',
           'abstract_clean','label', 'task','method','goal1',
           'goal2', 'goal3', 'acknowledgments_clean', 'year','test']].rename(columns={'title_clean':'title','label':'label_nlp4sg',
                                                                              'abstract_clean':'abstract',
                                                                              'acknowledgments_clean':'acknowledgments'})
    df_final=get_dummy_sdg(df_final)

    low_ocurrence=low_ocurrence.rename(columns={'Most Related SG goal':'goal1_raw',
           '(if exists) 2nd Related SG Goal':'goal2_raw', '(if exists) 3rd Related SG Goal':'goal3_raw'})
    low_ocurrence=low_ocurrence.rename(columns={"SG_or_not":"label"})
    low_ocurrence["label"]=low_ocurrence["label"].fillna(0)
    low_ocurrence=low_ocurrence.assign(goal1_raw=np.where(low_ocurrence.label==0,'',low_ocurrence.goal1_raw))
    low_ocurrence=low_ocurrence.assign(goal2_raw=np.where(low_ocurrence.label==0,'',low_ocurrence.goal2_raw))
    low_ocurrence=low_ocurrence.assign(goal3_raw=np.where(low_ocurrence.label==0,'',low_ocurrence.goal3_raw))
    low_ocurrence=assign_goals(low_ocurrence)

    low_ocurrence.label=low_ocurrence.label.apply(int)
    low_ocurrence=low_ocurrence.rename(columns={'title_clean':'title', 'abstract_clean':'abstract',
                                  'acknowledgments_clean':'acknowledgments','label':'label_nlp4sg'})
    low_ocurrence=low_ocurrence.loc[~(low_ocurrence.ID.isin(df_all_goals.ID.unique()))]
    low_ocurrence=low_ocurrence.loc[:,['ID', 'url', 'title', 'abstract', 'label_nlp4sg', 'goal1', 'goal2', 'goal3',
           'acknowledgments']]
    low_ocurrence=get_dummy_sdg(low_ocurrence)

    low_ocurrence.to_csv("../dataset/nlp4sg_papers_low_occurrence.csv",index=False)
    
    df_final.to_csv("../dataset/nlp4sg_papers.csv",index=False)
if __name__ == '__main__':
    main()