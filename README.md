# Beyond Good Intentions: Reporting the Research Landscape of NLP for Social Good


For reproduction of the models place the "data", "dataset" and "outputs" folders in this directory <br/>

"sg_classifier" folder corresponds to the code for the dataset creation and the task <br/>
"sg_match" folder" corresponds to the code for the models of task 2 <br/>
"sg_information_extraction" folder corresponds to the code for the models of task 3 <br/>


## General installation instructions

1. `conda create -n nlp4sg python=3.7`
2. `conda activate nlp4sg`
3. `pip install -r requirements.txt`
4. `export base_folder=path_to_the_project`
5. `export OPENAI_API_KEY=your_gpt3_key`  necessary for GPT-3 models
