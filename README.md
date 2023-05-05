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


### Use models

Here is an example that runs the models for the 3 NLP4SG tasks using the test set of our NLP4SGPapers dataset.
```bash
./pipeline/pipeline.sh
```





### Models

To reproduce the results presented in the paper follow the instructions for each task

[Task 1: NLP4SG Classification](https://github.com/feradauto/nlp4sg/tree/main/sg_classifier)  <br/>
[Task 2: UN SDG Classifier](https://github.com/feradauto/nlp4sg/tree/main/sg_match)  <br/>
[Task 3: Task and Method identification](https://github.com/feradauto/nlp4sg/tree/main/sg_information_extraction)




