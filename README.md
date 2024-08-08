# DeepSample 

This repository provides the artifacts for DeepSample execution. 
The experiments on 9 DNN for classification and 2 DNN for regression can be replicated by running the script 'run_DeepSample.sh'.
Once the experiments are completed, all the results will be generated in the 'Results' folder.

## DeepSample folder
The folder "DeepSample" contains all the ".jar" files and the source code.
The source code is organized as follows:
- The "main" folder contains all the java files for the jar generation.
- "selector.classification" contains the source code of the implemented sampling algorithms for classification, plus DeepEST.
- "selector.regression" contains the source code of the implemented sampling algorithms for regression, plus DeepEST.
- "utility" and "utility.regression" contain utilities and data structures useful for the selectors.

The code has been developed with Eclipse.

## Other samplers
CES and SRS implementations are available at: 'https://github.com/Lizenan1995/DNNOpAcc'.
The results can be replicated by:
1. cloning the repository;
2. importing the models in 'CE method/MNIST/normal';
3. importing and running in 'CE method/MNIST/normal' the files available in the 'CES_SRS' folder of our repository ('crossentropy*.py' files). 

In the following, we reported the chosen layers (LAY parameter in 'crossentropy*.py') required to run CES for each model: 
- A: -2 
- B: -4
- C: -2
- D: -4
- E: -4
- F: -2
- G: -2
- H: -4
- I: -4
- Dave_orig: -3
- Dave_dropout: -3

## Neural Networks availability
The trained models are included in the 'models' folder. Model G, Dave_orig and Dave_dropout models are available at 'https://file.io/0405r74sgwqV'.
The folder 'dataset' contains the dataset with the predictions and the auxiliary variables for all models.
The source code for the classification models is available at: 'https://github.com/ICOS-OAA/ICOS.git'

## Paper results
The folder 'Results_paper' contains the results reported in the paper.

A Python notebook is provided for the practitioners to query the results, for instance to ask for the best performing technique (based on their occurrences in the top-3 rankings) given a main objective (low RMSE, RMedSE or large failure exposure ability) and a configuration as input (e.g., small sample size, MNIST dataset, confidence as auxiliary variable) ([notebook](./Results_paper/_Discussion/interactive_notebook/summary.ipynb))

## Requirements and Dependencies
The provided code requires Java 8.
The "libs" folder contains all the libraries required to run the experiments.
