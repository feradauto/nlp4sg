import pandas as pd
import numpy as np
import json
import requests
import time



def main():
    df_urls=pd.read_csv("./positives.csv")
    for i,d in df_urls.iterrows():
        response = requests.get(d['url'])
        save_path="./test_set/"+d['ID']+".pdf"
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(save_path," saved url:",d['url'])

if __name__ == '__main__':
    main()