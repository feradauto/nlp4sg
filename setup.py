from setuptools import setup
from setuptools import find_packages

install_requires = install_requires=[
        'numpy>=1.22',
        'openai==0.18.1',
        'pandas==1.3.5',
        'requests==2.27.1',
        'scikit-learn==1.2.0',
        'sentence-transformers==2.2.2',
        'sentencepiece==0.1.97',
        'transformers==4.17.0',
        'torch==2.0.1',
        'torchaudio==2.0.2',
        'torchvision==0.15.2',
        'notebook==6.0.1',
        'matplotlib==3.4.0',
        'jupyter==1.0.0',
        'accelerate==0.15.0',
        'datasets==2.8.0',
        'evaluate==0.4.0',
    ]

setup(
    name='nlp4sg',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/feradauto/nlp4sg',
    license='CC BY-SA 4.0',
    author='Fernando Gonzalez Adauto',
    author_email='fer.adauto@gmail.com',
    description='NLP4SG initiative',
    install_requires=install_requires,
)