import json
import pandas as pd
import re
import numpy as np


def extract_predictions_gpt3(df):
    df=df.assign(sdg1=np.where(df['GPT3_response'].str.lower().str.contains("goal 1 |goal 1:|poverty"),1,0))
    df=df.assign(sdg2=np.where(df['GPT3_response'].str.lower().str.contains("goal 2|hunger"),1,0))
    df=df.assign(sdg3=np.where(df['GPT3_response'].str.lower().str.contains("goal 3|health"),1,0))
    df=df.assign(sdg4=np.where(df['GPT3_response'].str.lower().str.contains("goal 4|education"),1,0))
    df=df.assign(sdg5=np.where(df['GPT3_response'].str.lower().str.contains("goal 5|gender"),1,0))
    df=df.assign(sdg6=np.where(df['GPT3_response'].str.lower().str.contains("goal 6|clean water"),1,0))
    df=df.assign(sdg7=np.where(df['GPT3_response'].str.lower().str.contains("goal 7|clean energy"),1,0))
    df=df.assign(sdg8=np.where(df['GPT3_response'].str.lower().str.contains("goal 8|decent work"),1,0))
    df=df.assign(sdg9=np.where(df['GPT3_response'].str.lower().str.contains("goal 9|industry|innovation"),1,0))
    df=df.assign(sdg10=np.where(df['GPT3_response'].str.lower().str.contains("goal 10|inequal"),1,0))
    df=df.assign(sdg11=np.where(df['GPT3_response'].str.lower().str.contains("goal 11|sustainable cities"),1,0))
    df=df.assign(sdg12=np.where(df['GPT3_response'].str.lower().str.contains("goal 12|responsible consumption"),1,0))
    df=df.assign(sdg13=np.where(df['GPT3_response'].str.lower().str.contains("goal 13|climate"),1,0))
    df=df.assign(sdg14=np.where(df['GPT3_response'].str.lower().str.contains("goal 14|life below water"),1,0))
    df=df.assign(sdg15=np.where(df['GPT3_response'].str.lower().str.contains("goal 15|life on land"),1,0))
    df=df.assign(sdg16=np.where(df['GPT3_response'].str.lower().str.contains("goal 16|peace|justice"),1,0))
    df=df.assign(sdg17=np.where(df['GPT3_response'].str.lower().str.contains("goal 17|partnership"),1,0))
    return df

def assign_goals(df_all_goals):
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
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("partnership"),'Partnership for the Goals',
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
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("partnership"),'Partnership for the Goals',
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
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("partnership"),'Partnership for the Goals',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("responsible_consumption_and_production"),'Responsible Consumption and Production',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("reduced|social_equality"),'Reduced Inequalities',''
                              )))))))))))))))))))
    return df_all_goals

def extract_tasks_methods(df_labels_task,df_goals):    
    df_labels_task['Method']=[list([]) for _ in range(df_labels_task.shape[0])]
    df_labels_task['Task']=[list() for _ in range(df_labels_task.shape[0])]


    for i,d in df_labels_task.iterrows():
        tasks=[]
        for e in d['task_scirex']:
            tasks.append(e['top_word'])
        df_labels_task.at[i,'Task']=tasks
        methods=[]
        for e in d['method_scirex']:
            methods.append(e['top_word'])
        df_labels_task.at[i,'Method']=methods

    df_labels_task=df_labels_task.merge(df_goals,on=['ID'],how='left')

    df_labels_task=df_labels_task.explode('Task').rename(columns={'Task':'tasks'})

    df_labels_task=df_labels_task.explode('Method').rename(columns={'Method':'methods'})

    df_labels_task=df_labels_task.loc[~df_labels_task.tasks.isin(['natural language processing','healthcare','downstream task','inference','predictions','nlp','nlp applications'])]

    df_labels_task=df_labels_task.loc[~df_labels_task.methods.isin(['natural language processing','nlp','nlp applications'])]

    df_labels_task['total_methods']=df_labels_task.groupby(['ID']).methods.transform('count')

    df_labels_task['weight']=1/df_labels_task.total_methods

    df_labels_task=df_labels_task.dropna()
    return df_labels_task

def map_names(df_mapping,df_labels_task):

    df_mapping.center=df_mapping.center.fillna(method='ffill')

    df_mapping=df_mapping.loc[df_mapping.word!='Cluster name: ']

    df_mapping.word=df_mapping.word.str.rstrip(' ').str.lstrip(' ')
    df_mapping.center=df_mapping.center.str.rstrip(' ').str.lstrip(' ')

    df_mapping_tasks=df_mapping.rename(columns={'word':'tasks','center':'center_task'})
    df_mapping_methods=df_mapping.rename(columns={'word':'methods','center':'center_method'})

    df_labels_task=df_labels_task.merge(df_mapping_tasks,on=['tasks'],how='left').merge(df_mapping_methods,on=['methods'],how='left')

    df_labels_task=df_labels_task.assign(tasks=np.where((~df_labels_task.center_task.isna()),df_labels_task.center_task,df_labels_task.tasks))

    df_labels_task=df_labels_task.assign(methods=np.where((~df_labels_task.center_method.isna()),df_labels_task.center_method,df_labels_task.methods))
    return df_labels_task

def common_tasks(df_processed_group,df_labels_task):

    top_tasks=df_processed_group.groupby(['tasks']).weight.sum().reset_index().sort_values(['weight'],ascending=False).head(25).loc[:,['tasks']]

    top_methods=df_processed_group.groupby(['methods']).weight.sum().reset_index().sort_values(['weight'],ascending=False).head(25).loc[:,['methods']]

    df_processed_group=df_processed_group.assign(tasks=np.where(
    df_processed_group.tasks.isin(top_tasks.tasks.values),df_processed_group.tasks,"Other tasks"))

    df_processed_group=df_processed_group.assign(methods=np.where(
    df_processed_group.methods.isin(top_methods.methods.values),df_processed_group.methods,"Other methods"))

    df_processed_group=df_processed_group.sort_values(['weight'],ascending=False)



    df_processed_group=df_processed_group.groupby(['Goal','tasks','methods']).weight.sum().reset_index()

    df_tops=df_processed_group.copy()

    valid_methods=df_tops.loc[df_tops.methods!='Other methods',['methods']].drop_duplicates()
    valid_methods=valid_methods.assign(valid_method=1)

    valid_tasks=df_tops.loc[df_tops.tasks!='Other tasks',['tasks']].drop_duplicates()
    valid_tasks=valid_tasks.assign(valid_task=1)

    df_labels_task=df_labels_task.merge(valid_methods,on=['methods'],how='left').merge(valid_tasks,on=['tasks'],how='left')

    df_labels_task=df_labels_task.assign(valid_method=df_labels_task.valid_method.fillna(0))
    df_labels_task=df_labels_task.assign(valid_task=df_labels_task.valid_task.fillna(0))

    w_valid_method=df_labels_task.groupby(['ID']).valid_method.sum().reset_index()
    w_valid_method=w_valid_method.loc[w_valid_method.valid_method>0]

    df_labels_task_wvm=df_labels_task.loc[df_labels_task.ID.isin(w_valid_method.ID.unique())]
    df_labels_task_nwvm=df_labels_task.loc[~df_labels_task.ID.isin(w_valid_method.ID.unique())]

    df_labels_task_wvm=df_labels_task_wvm.loc[df_labels_task_wvm.valid_method==1]

    df_labels_task=pd.concat([df_labels_task_wvm,df_labels_task_nwvm])

    w_valid_task=df_labels_task.groupby(['ID']).valid_task.sum().reset_index()
    w_valid_task=w_valid_task.loc[w_valid_task.valid_task>0]

    df_labels_task_wvm_t=df_labels_task.loc[df_labels_task.ID.isin(w_valid_task.ID.unique())]
    df_labels_task_nwvm_t=df_labels_task.loc[~df_labels_task.ID.isin(w_valid_task.ID.unique())]

    df_labels_task_wvm_t=df_labels_task_wvm_t.loc[df_labels_task_wvm_t.valid_task==1]

    df_labels_task=pd.concat([df_labels_task_wvm_t,df_labels_task_nwvm_t])

    df_labels_task['total_methods']=df_labels_task.groupby(['ID']).methods.transform('count')

    df_labels_task['weight']=1/df_labels_task.total_methods

    df_labels_task=df_labels_task.loc[~((df_labels_task.methods.str.contains('translation|smt|mt'))&(df_labels_task.total_methods==1))]

    df_labels_task=df_labels_task.reset_index(drop=True)

    df_labels_task['total_methods']=df_labels_task.groupby(['ID']).methods.transform('count')

    df_labels_task['weight']=1/df_labels_task.total_methods
    return df_labels_task

def keep_top_names(df_labels_task):

    df_processed_group=df_labels_task.groupby(['Goal','tasks','methods']).weight.sum().reset_index()

    top_tasks=df_processed_group.groupby(['tasks']).weight.sum().reset_index().sort_values(['weight'],ascending=False).head(28).loc[:,['tasks']]

    top_tasks=top_tasks.loc[~top_tasks.tasks.isin(['predictions','downstream task','healthcare','detection','language technology'])]

    top_methods=df_processed_group.groupby(['methods']).weight.sum().reset_index().sort_values(['weight'],ascending=False).head(25).loc[:,['methods']]

    df_processed_group=df_processed_group.assign(tasks=np.where(
    df_processed_group.tasks.isin(top_tasks.tasks.values),df_processed_group.tasks,"Other tasks"))

    df_processed_group=df_processed_group.assign(methods=np.where(
    df_processed_group.methods.isin(top_methods.methods.values),df_processed_group.methods,"Other methods"))

    df_processed_group=df_processed_group.sort_values(['weight'],ascending=False)



    df_processed_group=df_processed_group.groupby(['Goal','tasks','methods']).weight.sum().reset_index()
    return df_processed_group

def merge_social_need_names(social_needs,df_processed_group_filtered):

    social_needs.Goal=social_needs.Goal.replace({'No Hunger':'Zero Hunger','Industry, Innovation and Infrastrucure':'Industry, Innovation and Infrastructure'})

    social_needs=social_needs.loc[:,['Goal']].reset_index().rename(columns={'index':'order'})

    df_processed_group_filtered=df_processed_group_filtered.merge(social_needs,on=['Goal'],how='left')

    df_processed_group_filtered=df_processed_group_filtered.sort_values('order').reset_index(drop=True)
    return df_processed_group_filtered

def rename_and_filter(df_processed_group_filtered):

    ## sequential
    df_processed_group_filtered=df_processed_group_filtered.assign(others_count=np.where(
        df_processed_group_filtered.tasks=='Other tasks',1,0))
    df_processed_group_filtered=df_processed_group_filtered.assign(others_count=np.where(
        df_processed_group_filtered.methods=='Other methods',df_processed_group_filtered.others_count+1,df_processed_group_filtered.others_count))

    df_processed_group_filtered=df_processed_group_filtered.loc[(df_processed_group_filtered.others_count<=1),: ].reset_index(drop=True)

    df_processed_group_filtered=df_processed_group_filtered.assign(methods=np.where((df_processed_group_filtered.methods.isin(['annotation']))
                                                          ,'annotation schemes',df_processed_group_filtered.methods))

    df_processed_group_filtered=df_processed_group_filtered.assign(methods=np.where((df_processed_group_filtered.methods.isin(['automatic speech recognition']))
                                                          ,'automatic speech recognition models',df_processed_group_filtered.methods))

    df_processed_group_filtered=df_processed_group_filtered.assign(methods=np.where((df_processed_group_filtered.methods.isin(['classification']))
                                                          ,'classifiers',df_processed_group_filtered.methods))

    df_processed_group_filtered=df_processed_group_filtered.assign(methods=np.where((df_processed_group_filtered.methods.isin(['fact checking']))
                                                          ,'fact checking models',df_processed_group_filtered.methods))

    df_processed_group_filtered=df_processed_group_filtered.assign(methods=np.where((df_processed_group_filtered.methods.isin(['named entity recognition']))
                                                          ,'ner models',df_processed_group_filtered.methods))

    df_processed_group_filtered=df_processed_group_filtered.assign(methods=np.where((df_processed_group_filtered.methods.isin(['nlp applications']))
                                                          ,'nlp models',df_processed_group_filtered.methods))

    df_processed_group_filtered=df_processed_group_filtered.assign(methods=np.where((df_processed_group_filtered.methods.isin(['part of speech']))
                                                          ,'part of speech models',df_processed_group_filtered.methods))

    df_processed_group_filtered=df_processed_group_filtered.assign(methods=np.where((df_processed_group_filtered.methods.isin(['machine translation']))
                                                          ,'machine translation system',df_processed_group_filtered.methods))

    df_processed_group_filtered=df_processed_group_filtered.assign(tasks=np.where((df_processed_group_filtered.tasks.isin(['annotation']))
                                                          ,'data collection',df_processed_group_filtered.tasks))

    df_processed_group_filtered=df_processed_group_filtered.assign(tasks=np.where((df_processed_group_filtered.tasks.isin(['word embeddings']))
                                                          ,'embeddings',df_processed_group_filtered.tasks))
    return df_processed_group_filtered

def order_format(df_sankey):


    df_sankey=df_sankey.assign(id_source=np.where(df_sankey.source=='Other methods',9998,df_sankey.id_source))
    df_sankey=df_sankey.assign(id_source=np.where(df_sankey.source=='Other tasks',9997,df_sankey.id_source))


    df_sankey=df_sankey.assign(id_target=np.where(df_sankey.target=='Other methods',9998,df_sankey.id_target))
    df_sankey=df_sankey.assign(id_target=np.where(df_sankey.target=='Other tasks',9997,df_sankey.id_target))

    df_sankey=df_sankey.assign(id_order=np.where(df_sankey.target.isin(['No organization','Other methods','Other tasks']),
                                                 9999,
                                        np.where(df_sankey.link_type=='goal_tasks',df_sankey.id_source,999)))

    df_sankey=df_sankey.sort_values(['id_order','id_source','id_target'])

    df_sankey=df_sankey.assign(source=np.where(df_sankey.link_type=='goal_tasks',df_sankey.source,df_sankey.source.str.title()))


    df_sankey.target=df_sankey.target.str.title()

    capitalized=['Lstm','Bert','Covid 19','Computational Linguistics']

    df_sankey.source=df_sankey.source.replace("Roberta","RoBERTa")
    df_sankey.source=df_sankey.source.replace("Bert","BERT")
    df_sankey.source=df_sankey.source.replace("Lstm","LSTM")
    df_sankey.source=df_sankey.source.replace("Nlp","NLP",regex=True)
    df_sankey.source=df_sankey.source.replace("Covid 19","COVID-19 Analysis")
    df_sankey.source=df_sankey.source.replace("Computational Linguistics","Linguistic Analysis")
    df_sankey.source=df_sankey.source.replace("Multi Task Learning","Multi-Task Learning")


    df_sankey.target=df_sankey.target.replace("Roberta","RoBERTa")
    df_sankey.target=df_sankey.target.replace("Bert","BERT")
    df_sankey.target=df_sankey.target.replace("Lstm","LSTM")
    df_sankey.target=df_sankey.target.replace("Nlp","NLP",regex=True)
    df_sankey.target=df_sankey.target.replace("Covid 19","COVID-19 Analysis")
    df_sankey.target=df_sankey.target.replace("Computational Linguistics","Linguistic Analysis")
    df_sankey.target=df_sankey.target.replace("Multi Task Learning","Multi-Task Learning")
    df_sankey=df_sankey.assign(value=round(df_sankey['value'],1))

    df_sankey=df_sankey.assign(id_target=df_sankey.id_target.apply(str))
    df_sankey=df_sankey.assign(id_source=df_sankey.id_source.apply(str))
    return df_sankey

def get_names_and_order(df_sankey,tasks_list,goals_list,methods_list):

    sources=df_sankey.loc[:,['source','id_order']].rename(columns={'source':'name'})
    targets=df_sankey.loc[:,['target','id_order']].rename(columns={'target':'name'})

    order_df=pd.concat([sources,targets]).drop_duplicates(subset=['name'],keep='first').sort_values(['id_order','name'])

    order_df=order_df.assign(id_order=np.where(order_df.name.isin(['No Organization','Other Methods','Other Tasks']),
                                                 9999,order_df.id_order))

    order_df=order_df.sort_values(['id_order','name'])

    order_df=order_df.reset_index(drop=True).reset_index().rename(columns={'index':'order'})

    tasks_list.name=tasks_list.name.str.title()

    order_df=order_df.assign(node_type=np.where(order_df.name.isin(goals_list.name.unique()),"Goal",
                                    np.where(order_df.name.isin(tasks_list.name.unique()),"Task","Method")))

    order_dict = dict(zip(order_df.name, order_df.order))
    return order_dict,order_df

def get_names_merge(df_processed_group_filtered):

    names_list=list(df_processed_group_filtered.Goal.unique())
    entry_type=['Goal' for i in range(df_processed_group_filtered.Goal.nunique())]

    names_list.extend(list(df_processed_group_filtered.tasks.unique()))
    entry_type.extend(['task' for i in range(df_processed_group_filtered.tasks.nunique())])

    names_list.extend(list(df_processed_group_filtered.methods.unique()))
    entry_type.extend(['method' for i in range(df_processed_group_filtered.methods.nunique())])

    names=pd.DataFrame(list(zip(names_list,entry_type)),columns=['names','types']).reset_index()

    names=names.rename(columns={'index':'id'})

    names_goals=names.loc[names.types=='Goal'].rename(columns={'names':'Goal'})
    names_tasks=names.loc[names.types=='task'].rename(columns={'names':'tasks'})
    names_methods=names.loc[names.types=='method'].rename(columns={'names':'methods'})
    names_goals=names_goals.rename(columns={'id':'id_goal'})
    names_tasks=names_tasks.rename(columns={'id':'id_tasks'})
    names_methods=names_methods.rename(columns={'id':'id_methods'})

    df_processed_group_filtered=df_processed_group_filtered.rename(columns={'weight':'value'})

    df_processed_group_filtered=df_processed_group_filtered.merge(names_goals,on=['Goal']).merge(names_tasks,on=['tasks']).merge(names_methods,on=['methods'])
    return df_processed_group_filtered

def get_papers_features(positives,df_all_goals,df_labels_task):
    positives=positives.loc[:,['ID','title_clean','abstract_clean']]

    papers_dataset=df_all_goals.loc[:,['ID','title_clean','abstract_clean']]
    positives=positives.loc[~positives.ID.isin(papers_dataset.ID.unique())]
    positives=pd.concat([positives,papers_dataset])
    df_labels_task_save=df_labels_task.loc[:,['ID', 'methods', 'tasks', 'Goal',
          'center_task','center_method']]

    methods_grouped=df_labels_task_save.loc[:,['ID','methods','center_method']].drop_duplicates().groupby(['ID']).agg({'methods':lambda x: list(x),'center_method':lambda x: list(x)}).reset_index()

    tasks_grouped=df_labels_task_save.loc[:,['ID','tasks','center_task']].drop_duplicates().groupby(['ID']).agg({'tasks':lambda x: list(x),'center_task':lambda x: list(x)}).reset_index()

    goal_grouped=df_labels_task_save.loc[:,['ID','Goal']].drop_duplicates().groupby(['ID']).agg({'Goal':lambda x: list(x)}).reset_index()

    df_labels_task_save=methods_grouped.merge(tasks_grouped,on=['ID'],how='left').merge(goal_grouped,on=['ID'],how='left').merge(positives,on='ID',how='left')

    df_labels_task_save=df_labels_task_save.assign(title_clean=df_labels_task_save.title_clean.replace("-"," ",regex=True).replace("  "," ",regex=True))
    df_labels_task_save=df_labels_task_save.assign(abstract_clean=df_labels_task_save.abstract_clean.replace("-"," ",regex=True).replace("  "," ",regex=True))
    return df_labels_task_save

mapping_sdg={'sdg1': 'No Poverty',
 'sdg2': 'Zero Hunger',
 'sdg3': 'Good Health and Well-Being',
 'sdg4': 'Quality Education',
 'sdg5': 'Gender Equality',
 'sdg6': 'Clean Water and Sanitation',
 'sdg7': 'Affordable and Clean Energy',
 'sdg8': 'Decent Work and Economic Growth',
 'sdg9': 'Industry, Innovation and Infrastructure',
 'sdg10': 'Reduced Inequalities',
 'sdg11': 'Sustainable Cities and Communities',
 'sdg12': 'Responsible Consumption and Production',
 'sdg13': 'Climate Action',
 'sdg14': 'Life Below Water',
 'sdg15': 'Life on Land',
 'sdg16': 'Peace, Justice and Strong Institutions',
 'sdg17': 'Partnership for the Goals'
}

def main():
    outputs_path="../outputs/"
    data_path="../data/"

    df_labels_task=pd.read_json(outputs_path+"sg_ie/positives_tasks_methods_clusters_final_f.json")
    df_task_test=pd.read_json(outputs_path+"sg_ie/test_scirex_tasks_methods_clusters_final_f.json")
    df_g=pd.read_csv("../sg_match/progress_singular_gpt3.csv")
    social_needs=pd.read_csv(data_path+"others/social_needs.csv")
    test_set=pd.read_csv(outputs_path+"general/test_set_final.csv")
    train_set=pd.read_csv(outputs_path+"general/train_set_final.csv")
    dev_set=pd.read_csv(outputs_path+"general/dev_set_final.csv")
    positives=pd.read_csv(outputs_path+"sg_ie/positives_ready.csv")
    ###df_mapping=pd.read_csv(outputs+"words_mapping.csv")
    df_mapping=pd.read_csv("../sg_match/clusters.psv",sep='|')
    df_all_goals=pd.concat([dev_set,train_set,test_set])
    df_all_goals=df_all_goals.loc[df_all_goals.label==1]

    df_g=df_g.loc[~df_g.ID.isin(df_all_goals.ID.unique())]

    df_all_goals=assign_goals(df_all_goals)

    df_group=pd.concat([df_all_goals.loc[:,['ID','goal1']],df_all_goals.loc[:,['ID','goal2']].rename(columns={'goal2':'goal1'}),df_all_goals.loc[:,['ID','goal3']].rename(columns={'goal3':'goal1'})])

    df_group=df_group.loc[df_group.goal1!='']

    df_group=df_group.rename(columns={'goal1':'Goal'})

    df_g=extract_predictions_gpt3(df_g)

    df_g=pd.melt(df_g,id_vars=['ID'],value_vars=['sdg1', 'sdg2', 'sdg3', 'sdg4', 'sdg5',
           'sdg6', 'sdg7', 'sdg8', 'sdg9', 'sdg10', 'sdg11', 'sdg12', 'sdg13',
           'sdg14', 'sdg15', 'sdg16', 'sdg17'],var_name=['sdg'],value_name='sdg_val')

    df_g=df_g.loc[df_g.sdg_val==1]

    df_g['Goal']=df_g.sdg.replace(mapping_sdg)
    df_g=df_g.loc[:,['ID','Goal']]

    df_goals=pd.concat([df_g,df_group])

    df_goals=df_goals.drop_duplicates()

    df_task_test=df_task_test.loc[df_task_test.ID.isin(df_all_goals.ID.unique())]

    df_labels_task=pd.concat([df_labels_task,df_task_test]).reset_index(drop=True)

    df_labels_task=extract_tasks_methods(df_labels_task,df_goals)

    df_labels_task=map_names(df_mapping,df_labels_task)

    ## restart

    df_processed_group=df_labels_task.groupby(['Goal','tasks','methods']).weight.sum().reset_index()

    ## original papers

    df_labels_task=common_tasks(df_processed_group,df_labels_task)

    df_processed_group=keep_top_names(df_labels_task)

    df_processed_group_filtered=df_processed_group.copy()

    df_processed_group_filtered=merge_social_need_names(social_needs,df_processed_group_filtered)

    ## check this!!!!

    df_processed_group_filtered.weight.sum()

    df_processed_group_filtered=rename_and_filter(df_processed_group_filtered)

    df_processed_group_filtered=get_names_merge(df_processed_group_filtered)


    goal_tasks=df_processed_group_filtered.groupby(['Goal','id_goal','tasks','id_tasks']).value.sum().reset_index()
    tasks_methods=df_processed_group_filtered.groupby(['methods','id_methods','tasks','id_tasks']).value.sum().reset_index()

    goal_tasks=goal_tasks.rename(columns={'Goal':'source','id_goal':'id_source','tasks':'target','id_tasks':'id_target'})
    tasks_methods=tasks_methods.rename(columns={'tasks':'source','id_tasks':'id_source','methods':'target','id_methods':'id_target'})

    goal_tasks=goal_tasks.assign(link_type='goal_tasks')
    tasks_methods=tasks_methods.assign(link_type='tasks_methods')

    tasks_list=goal_tasks.loc[:,['target']].drop_duplicates().rename(columns={'target':'name'}).reset_index(drop=True)

    goals_list=goal_tasks.loc[:,['source']].drop_duplicates().rename(columns={'source':'name'}).reset_index(drop=True)

    methods_list=tasks_methods.loc[:,['target']].drop_duplicates().rename(columns={'target':'name'}).reset_index(drop=True)

    df_sankey=pd.concat([goal_tasks,tasks_methods])

    df_sankey=order_format(df_sankey)


    df_sankey.rename(columns={'value':'weight'}).to_json("sankey_no_org.json",orient="records")

    order_dict,order_df=get_names_and_order(df_sankey,tasks_list,goals_list,methods_list)

    with open('order_sankey.json', 'w') as fp:
        json.dump(order_dict, fp)

    order_df.loc[:,['name','node_type']].to_json("names.json",orient="records")

    df_labels_task_save=get_papers_features(positives,df_all_goals,df_labels_task)

    df_labels_task_save.to_json("papers_features.json",orient="records")
    
if __name__ == '__main__':
    main()