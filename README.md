# Masterthesis
This repository contains the code I used for my master thesis "Influence of Convolutional Neural Network Architectures on Membership Inference Attacks". Data used to train the models can be found here: https://drive.google.com/drive/folders/1JO3UldQiDEBlv148VKur58QDKLS3VgAR?usp=share_link

Unfortunaly, due to multiple computing environments, the python programs may need some adjustments before launching them in your own environment. The order, in which I ran the scripts, is the following:
1. random_search.py: performs the random search in order to identify suitable architectures for later analyses
2. train_shadow_models.py: trains shadow models for previously identified architectures. This script was run in a job-processing environment and takes an integer value as input value, specifying which architecture to load from ../architectures
3. retrieve_shadow_data.py: queries shadow models with their own test and training set, calculating individual loss values. Within this script, the ID of the victim model for which the shadow models are to be processed needs to be set in the parameter "victim_model".
4. generate_attack_datasets.py: uses the shadow model data calculated in the third script in order to calculate the means of the in and out loss distributions of every data point for twenty different victim models within the set of 101 shadow models. Within this script, the IDs of the victim models to be processed need to be set within the array "model_evaluations".
5. mia_attack_final.py: finally calculates the attack statistics for every victim architecture using the dataframes generated in the fourth script. Within this script, the IDs of the victim models to be processed need to be set within the array "model_evaluations".
