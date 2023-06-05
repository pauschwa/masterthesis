'''
DISCLAIMER: This script uses open-source libraries from the paper "An Efficient Approach for Assessing Hyperparameter Importance" by Hutter et al. (2014). 

The code can be found in the following Git: https://github.com/automl/fanova
The documentation can be found here: https://automl.github.io/fanova/manual.html

Steps to install the required libraries of the functional ANOVA:
1. !git clone https://github.com/automl/fanova.git
2. cd /content/fanova/
3. !pip install -r /content/fanova/requirements.txt
4. !python /content/fanova/setup.py install
'''

import os
import pandas as pd
import numpy as np
from pathlib import Path
from fanova import fANOVA
import ConfigSpace
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, NumericalHyperparameter, OrdinalHyperparameter

# Sets the target column, which is treated as dependent variable
y_col = 'TPR01_600'

'''
This script calculates the importance of n independent variables (features) on one dependent variable using the fuctional-ANOVA framework, proposed by Hutter et al. (2014).
We used their open-source implementation to calculate the importance of considered architectural components with regard to the vulnerability of the resulting model to MIA.
The open-source implementation by Hutter et al. can be found here: https://automl.github.io/fanova/manual.html.
'''
if __name__ == "__main__":
    # Load attack results, containing the vulnerability statistics for every considered architecture
    dir = Path(os.path.dirname(__file__))
    dir = dir.parent.absolute()
    attack_results_data = os.path.join(dir, 'attack_results.csv')

    # Non-relevant columns, which will be dropped from the dataframe, as they are neither DV nor IV.
    non_relevant_cols = ['Unnamed: 0', 'model_size', 'val_acc', 'train_acc', 'TPR1', 'TPR01', 'attack_acc', 'auc', 'TPR1_600', 
                         'TPR01_600', 'attack_acc_600', 'auc_600', 'TPR1_3000', 'TPR01_3000', 'attack_acc_3000', 'auc_3000', 
                         'TPR1_6000', 'TPR01_6000', 'attack_acc_6000', 'auc_6000']

    # If defined target column (dependent variable) is in non-relevant columns, then remove it before dropping
    if y_col in non_relevant_cols:
        non_relevant_cols.remove(y_col)
    
    # Read attack results and drop non-relevant columns
    data = pd.read_csv(attack_results_data, sep=',').set_index('victim_architecture').drop(non_relevant_cols, axis=1)

    # Data still contains categorical variables as string and one-hot encoded variables. Drop one-hot encoded columns.
    one_hot_cols = ['norm_BatchNorm2d', 'norm_GroupNorm', 'norm_InstanceNorm2d', 'norm_LayerNorm', 'norm_None', 'act_func_ELU', 
                    'act_func_LeakyReLU', 'act_func_Mish', 'act_func_ReLU','act_func_Tanh']
    data = data.drop(one_hot_cols, axis=1)

    # The implementation of Hutter et al. requires this format, therefore we will adjust our data accordingly.
    data['feature_maps'] = data['feature_maps'].replace([16], 0)
    data['feature_maps'] = data['feature_maps'].replace([32], 1)
    data['feature_maps'] = data['feature_maps'].replace([64], 2)
    data['feature_maps'] = data['feature_maps'].replace([96], 3)
    data['feature_maps'] = data['feature_maps'].replace([128], 4)

    data['layers_b1'] = data['layers_b1'].replace([1], 0)
    data['layers_b1'] = data['layers_b1'].replace([2], 1)
    data['layers_b1'] = data['layers_b1'].replace([3], 2)

    data['layers_b2'] = data['layers_b2'].replace([1], 0)
    data['layers_b2'] = data['layers_b2'].replace([2], 1)
    data['layers_b2'] = data['layers_b2'].replace([3], 2)

    data['layers_b3'] = data['layers_b3'].replace([1], 0)
    data['layers_b3'] = data['layers_b3'].replace([2], 1)
    data['layers_b3'] = data['layers_b3'].replace([3], 2)

    data['kernelsize_b1'] = data['kernelsize_b1'].replace([3], 0)
    data['kernelsize_b1'] = data['kernelsize_b1'].replace([5], 1)
    data['kernelsize_b1'] = data['kernelsize_b1'].replace([7], 2)

    data['kernelsize_b2'] = data['kernelsize_b2'].replace([3], 0)
    data['kernelsize_b2'] = data['kernelsize_b2'].replace([5], 1)
    data['kernelsize_b2'] = data['kernelsize_b2'].replace([7], 2)

    data['kernelsize_b3'] = data['kernelsize_b3'].replace([3], 0)
    data['kernelsize_b3'] = data['kernelsize_b3'].replace([5], 1)
    data['kernelsize_b3'] = data['kernelsize_b3'].replace([7], 2)

    data['norm_method'] = data['norm_method'].replace(['BatchNorm2d'], 0)
    data['norm_method'] = data['norm_method'].replace(['InstanceNorm2d'], 1)
    data['norm_method'] = data['norm_method'].replace(['GroupNorm'], 2)
    data['norm_method'] = data['norm_method'].replace(['LayerNorm'], 3)
    data['norm_method'] = data['norm_method'].replace(['None'], 4)

    data['act_func'] = data['act_func'].replace(['ELU'], 0)
    data['act_func'] = data['act_func'].replace(['Mish'], 1)
    data['act_func'] = data['act_func'].replace(['ReLU'], 2)
    data['act_func'] = data['act_func'].replace(['LeakyReLU'], 3)
    data['act_func'] = data['act_func'].replace(['Tanh'], 4)

    # Create target numpy array (DV) and drop DV from dataframe of features (IV)
    Y = data[y_col].to_numpy()
    X = data.drop([y_col], axis=1)

    # Define the configuration space as required by the implementation of Hutter et al.
    cs = ConfigurationSpace({
        "layers_b1": [1, 2, 3], 
        "layers_b2": [1, 2, 3], 
        "layers_b3": [1, 2, 3],
        "dropout_b1": (0.0, 0.5),
        "dropout_b2": (0.0, 0.5),
        "dropout_b3": (0.0, 0.5),
        "kernelsize_b1": [3, 5, 7], 
        "kernelsize_b2": [3, 5, 7], 
        "kernelsize_b3": [3, 5, 7],
        "feature_maps": [16, 32, 64, 96, 128],
        "strided_conv": [0, 1],

        "act_func": [0, 1, 2, 3, 4],
        "norm_method": [0, 1, 2, 3, 4],
    })

    # Finally, perform the functional ANOVA
    f = fANOVA(X=X, Y=Y, config_space=cs)

    # Retrieve the importance of every architectural component (IV) with regard to the vulnerability to MIA (DV)
    features = ['layers_b1', 'layers_b2', 'layers_b3', 'kernelsize_b1', 'kernelsize_b2',
       'kernelsize_b3', 'dropout_b1', 'dropout_b2', 'dropout_b3',
       'feature_maps', 'strided_conv', 'act_func', 'norm_method']
    importance_dict = {}
    for feat in features:
        res = f.quantify_importance((feat, ))
        _, value = res.popitem()
        importance_dict[feat] = value['individual importance']

    for feat in importance_dict:
        imp = importance_dict[feat]
        print(feat, ': ', imp)