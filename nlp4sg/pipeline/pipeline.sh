## Run all the pipeline with our NLP4SGPapers dataset
base_folder="./nlp4sg"
## First step of the pipeline SG classifier, you should specify the dataset
## Although we use our NLP4SGPapers dataset, you can use any other dataset. It seems that this repo (https://github.com/shauryr/ACL-anthology-corpus/tree/main) is going to collect periodically the ACL anthology dataset
python $base_folder/pipeline/01_sg_classification.py --dataset "feradauto/NLP4SGPapers"

## Second step, UN SDG Classification
## use "openai" to use GPT Instruct text-davinci-002 and get best results
## otherwise use facebook/bart-large-mnli
#python $base_folder/pipeline/02_unsdg_classification.py --model "openai"
python $base_folder/pipeline/02_unsdg_classification.py --model "facebook/bart-large-mnli"

## Third step, method and task extraction
## use "openai" to use GPT Instruct text-davinci-002 and get best results
## otherwise use bert-large-uncased-whole-word-masking-finetuned-squad
python $base_folder/pipeline/03_task_extraction.py #--model "openai"
python $base_folder/pipeline/04_method_extraction.py #--model "openai"

## Forth step, prepare files for visualization in the website
python $base_folder/pipeline/05_visualization.py