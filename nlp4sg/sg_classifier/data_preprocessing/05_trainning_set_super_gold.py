## Create train set with annotated data from NLP4SGINSPECTOR

import numpy as np
import pandas as pd
import re

def main():
    data_path="../../data/"
    outputs_path="../../outputs/"
    train_set=pd.read_csv(outputs_path+"general/others_SG.csv")
    df_test_final=pd.read_csv(outputs_path+"general/train_set_final.csv")
    df_test_final=df_test_final.assign(title_abstract=df_test_final.title_abstract_clean)
    train_set=df_test_final.loc[:,['ID','title','abstract','title_abstract','label','url']]
    train_set_final=train_set.sample(frac=1,random_state=42)
    train_set_final=train_set_final.reset_index(drop=True)
    train_set_final.label=train_set_final.label.apply(int)
    train_set_final.to_csv(outputs_path+"sg_classifier/train_set_labeled_super_gold.csv",index=False)

if __name__ == '__main__':
    main()