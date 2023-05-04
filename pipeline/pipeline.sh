## Run all the pipeline with our NLP4SGPapers dataset

## First step of the pipeline SG classifier
python $base_folder/pipeline/01_sg_classification.py

## Second step, UN SDG Classification
python $base_folder/pipeline/02_unsdg_classification.py

## Third step, method and task extraction
python $base_folder/pipeline/03_task_extraction.py
python $base_folder/pipeline/04_method_extraction.py