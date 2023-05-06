import os
import sys
import openai
import pandas as pd
import numpy as np
import argparse
from transformers import pipeline
def predict_gpt3():
    data_path="./"
    df=pd.read_csv(data_path+"results_task_1.csv")

    openai.api_key = os.getenv("OPENAI_API_KEY")

    preprompt="There is an NLP paper with the title and abstract:\n"
    question="Which of the UN goals does this paper directly contribute to? Provide the goal number and name."
    df=df.assign(statement=preprompt+df.text+"\n"+question)

    for i,d in df.iterrows():
        input_prompt=d['statement']
        completion = openai.Completion.create(engine="text-davinci-002", prompt=input_prompt,temperature=0,max_tokens=100,logprobs=1)

        df.loc[i,'full_prompt']=input_prompt
        df.loc[i,'GPT3_response']=completion.choices[0].text
    
    df=extract_predictions_gpt3(df)
    return df
def get_sn_rules_classification(df_classification):
    sd_pivot=pd.melt(df_classification,id_vars=['ID'],
            value_vars=['Reduced Inequalities', 'Sustainable Cities and Communities',
           'Responsible Consumption and Production', 'Good Health and Well-Being',
           'Life on Land', 'Life Below Water',
           'Peace, Justice and Strong Institutions',
           'Decent Work and Economic Growth', 'Partnership for the Goals',
           'Affordable and Clean Energy', 'Clean Water and Sanitation',
           'Industry, Innovation and Infrastructure', 'Quality Education',
           'Gender Equality', 'No Poverty', 'Climate Action', 'No Hunger'],value_name='proba',var_name="social_need")

    sd_pivot=sd_pivot.sort_values('proba',ascending=False)
    sd_pivot=sd_pivot.groupby(['ID']).head(2)
    sd_pivot=sd_pivot.assign(place=sd_pivot.groupby(['ID']).cumcount())
    sd_pivot=sd_pivot.fillna("")
    sn_top2=pd.pivot_table(sd_pivot,index=['ID'],columns=['place'],values=['social_need'],
                  aggfunc=np.sum)
    sn_top2=sn_top2.reset_index()
    sn_top2.columns=[col[0]+""+str(col[1]) for col in sn_top2.columns]
    sd_pivot_max=sd_pivot.groupby(['ID']).proba.max().reset_index().rename(columns={'proba':'proba_max'})
    df_classification=df_classification.merge(sn_top2,on=['ID'],how='left').merge(sd_pivot_max,on=['ID'])

    df_sn=df_classification.loc[:,['ID','social_need0','social_need1',"proba_max"]]
    return df_sn
def extract_predictions_mnli(df):
    mapping_mnli={
    'No Poverty':'sdg1',
    'Zero Hunger':'sdg2',
    'No Hunger':'sdg2',
    'Good Health and Well-Being':'sdg3',
    'Quality Education':'sdg4',
    'Gender Equality':'sdg5',
    'Clean Water and Sanitation':'sdg6',
    'Affordable and Clean Energy':'sdg7',
    'Decent Work and Economic Growth':'sdg8',
    'Industry, Innovation and Infrastructure':'sdg9',
    'Reduced Inequalities':'sdg10',
    'Sustainable Cities and Communities':'sdg11',
    'Responsible Consumption and Production':'sdg12',
    'Climate Action':'sdg13',
    'Life Below Water':'sdg14',
    'Life on Land':'sdg15',
    'Hate speech':'sdg16',
    'Disinformation and fake news':'sdg16',
    'Deception detection':'sdg16',
    'Privacy protection':'sdg16',
    'Peace, Justice and Strong Institutions':'sdg16',
    'Partnership for the Goals':'sdg17',
    }


    
    df=df.assign(sdg1=np.where(df['No Poverty']>=0.5,1,0))
    df=df.assign(sdg2=np.where(df['No Hunger']>=0.5,1,0))
    df=df.assign(sdg3=np.where(df['Good Health and Well-Being']>=0.5,1,0))
    df=df.assign(sdg4=np.where(df['Quality Education']>=0.5,1,0))
    df=df.assign(sdg5=np.where(df['Gender Equality']>=0.5,1,0))
    df=df.assign(sdg6=np.where(df['Clean Water and Sanitation']>=0.5,1,0))
    df=df.assign(sdg7=np.where(df['Affordable and Clean Energy']>=0.5,1,0))
    df=df.assign(sdg8=np.where(df['Decent Work and Economic Growth']>=0.5,1,0))
    df=df.assign(sdg9=np.where(df['Industry, Innovation and Infrastructure']>=0.5,1,0))
    df=df.assign(sdg10=np.where(df['Reduced Inequalities']>=0.5,1,0))
    df=df.assign(sdg11=np.where(df['Sustainable Cities and Communities']>=0.5,1,0))
    df=df.assign(sdg12=np.where(df['Responsible Consumption and Production']>=0.5,1,0))
    df=df.assign(sdg13=np.where(df['Climate Action']>=0.5,1,0))
    df=df.assign(sdg14=np.where(df['Life Below Water']>=0.5,1,0))
    df=df.assign(sdg15=np.where(df['Life on Land']>=0.5,1,0))
    df=df.assign(sdg16=np.where(df['Peace, Justice and Strong Institutions']>=0.5,1,0))
    #df=df.assign(sdg16=np.where(df['Privacy protection']>=0.5,1,df.sdg16))
    #df=df.assign(sdg16=np.where(df['Deception detection']>=0.5,1,df.sdg16))
    #df=df.assign(sdg16=np.where(df['Hate speech']>=0.5,1,df.sdg16))
    #df=df.assign(sdg16=np.where(df['Disinformation and fake news']>=0.5,1,df.sdg16))
    df=df.assign(sdg17=np.where(df['Partnership for the Goals']>=0.5,1,0))

    df_top=get_sn_rules_classification(df)
    df=df.merge(df_top,on=['ID'],how='left')
    for i,d in df.iterrows():
        df.at[i,mapping_mnli[d['social_need0']]]=1
    return df
def get_zero_shot_classification(model):
    data_path="./"
    df=pd.read_csv(data_path+"results_task_1.csv")
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
     'Partnership for the Goals']
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
       'Strengthen the means of implementation and revitalize the global partnership for sustainable development']
    
    
    classifier = pipeline("zero-shot-classification",
                          model=model, device='cpu')

    batch_size = 1 # see how big you can make this number before OOM
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
    df=extract_predictions_mnli(df)
    return df
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

def main(args):
    
    outputs_path="./"

    if args['model']=='openai':
        df=predict_gpt3()
    else:
        df=get_zero_shot_classification(args['model'])

    df.to_csv(outputs_path+"results_task_2.csv",index=False)


if __name__ == '__main__':
    args=argparse.ArgumentParser()
    ## openai if you want to use GPT 3
    ## facebook/bart-large-mnli
    args.add_argument("--model",type=str,default="facebook/bart-large-mnli")
    args=vars(args.parse_args())
    main(args)