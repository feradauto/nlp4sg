# Beyond Good Intentions: Reporting the Research Landscape of NLP for Social Good


## General instructions to execute the pipeline

1. `conda create -n nlp4sg python=3.9`
2. `conda activate nlp4sg`
3. `pip install -r requirements.txt`
4. `export OPENAI_API_KEY=your_gpt3_key`  necessary for GPT-3 models


Here is an example that runs the models for the 3 NLP4SG tasks using the test set of our NLP4SGPapers dataset.
```bash
./nlp4sg/pipeline/pipeline.sh
```

### More details

To install the project, execute the following command
```bash
python -m pip install .   
```

You can find more details about the models and evaluation for each task in the following files:

[Task 1: NLP4SG Classification](https://github.com/feradauto/nlp4sg/tree/main/nlp4sg/sg_classifier)  <br/>
[Task 2: UN SDG Classifier](https://github.com/feradauto/nlp4sg/tree/main/nlp4sg/sg_match)  <br/>
[Task 3: Task and Method identification](https://github.com/feradauto/nlp4sg/tree/main/nlp4sg/sg_information_extraction) <br/>

These folders include the preprocessing of the data, training and evaluation for task 1, and the evaluation of other models for tasks 2 and 3.

## Dataset

feradauto/NLP4SGPapers -- https://huggingface.co/datasets/feradauto/NLP4SGPapers

## Model in HugginFace hub

NLP4SG classification Task 1
feradauto/scibert_nlp4sg -- https://huggingface.co/feradauto/scibert_nlp4sg


## Reference
Beyond Good Intentions: Reporting the Research Landscape of NLP for Social Good -- https://arxiv.org/abs/2305.05471

```
@misc{gonzalez2023good,
      title={Beyond Good Intentions: Reporting the Research Landscape of NLP for Social Good}, 
      author={Fernando Gonzalez and Zhijing Jin and Bernhard Schölkopf and Tom Hope and Mrinmaya Sachan and Rada Mihalcea},
      year={2023},
      eprint={2305.05471},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```