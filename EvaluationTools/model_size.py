import os, torch
import torch.nn as nn
import pandas as pd

def build_block(ksize, nlayers, norm_method, act_func, fmaps, in_fmaps, do, strided, psize):
    # Constructs model blocks as described in the work.
    layers = []
    padding = int((ksize-1)/2)
    for i in range(nlayers):
        # Convolution layer
        layers.append(nn.Conv2d(in_fmaps, fmaps, kernel_size=ksize, padding=padding, stride=1))
        # Normalization layer
        if norm_method != "None":
            if norm_method == "GroupNorm":
                num_groups = int(fmaps/16)   # found by the authors to perform best
                layers.append(getattr(nn, norm_method)(num_groups, fmaps))
            elif norm_method == "LayerNorm":
                layers.append(getattr(nn, norm_method)(normalized_shape=[fmaps, psize, psize]))
            else:
                layers.append(getattr(nn, norm_method)(fmaps))
        # Activation function
        layers.append(getattr(nn, act_func)())
        # Dropout
        layers.append(nn.Dropout(do))

        in_fmaps = fmaps
    if strided == 0:
        layers.append(nn.MaxPool2d(2, 2))
    else:
        # Strided convolution instead of MaxPooling
        layers.append(nn.Conv2d(in_fmaps, fmaps, kernel_size=ksize, padding=padding, stride=2))
    return layers
    
def define_model(params):
    # Constructs model using architectural params defined within the random search. 

    # Convolution layers per block
    n_conv_layers_b1 = int(params['n_conv_layers_b1'])
    n_conv_layers_b2 = int(params['n_conv_layers_b2'])
    n_conv_layers_b3 = int(params['n_conv_layers_b3'])

    # Convolution kernel size per block
    size_conv_kernel_b1 = int(params['size_conv_kernel_b1'])
    size_conv_kernel_b2 = int(params['size_conv_kernel_b2'])
    size_conv_kernel_b3 = int(params['size_conv_kernel_b3'])

    # Feature maps in first block. Doubling for every following block.
    feature_maps = int(params['feature_maps'])

    # Dropout applied in each block
    dropout_b1 = float(params['dropout_b1'])
    dropout_b2 = float(params['dropout_b2'])
    dropout_b3 = float(params['dropout_b3'])

    # Activation function
    act_func = params['act_func']

    # Normalization method
    norm_method = params['norm_method']

    # Strided convolution instead of MaxPooling after each block
    strided_conv = int(params['strided_conv'])

    in_features = 3
    model_layers = []
    # Build block 1
    model_layers.extend(build_block(size_conv_kernel_b1, n_conv_layers_b1, norm_method, act_func, feature_maps, in_features, dropout_b1, strided_conv, psize=32))
    in_features = feature_maps
    feature_maps = feature_maps * 2

    # Build block 2
    model_layers.extend(build_block(size_conv_kernel_b2, n_conv_layers_b2, norm_method, act_func, feature_maps, in_features, dropout_b2, strided_conv, psize=16))
    in_features = feature_maps
    feature_maps = feature_maps * 2

    # Build block 3
    model_layers.extend(build_block(size_conv_kernel_b3, n_conv_layers_b3, norm_method, act_func, feature_maps, in_features, dropout_b3, strided_conv, psize=8))
    in_features = feature_maps

    # Classifier
    model_layers.append(nn.Flatten())
    model_layers.append(nn.Linear(in_features*4*4, 1024))
    model_layers.append(nn.ReLU())
    model_layers.append(nn.Dropout(0.4))
    model_layers.append(nn.Linear(1024, 128))
    model_layers.append(nn.ReLU())
    model_layers.append(nn.Dropout(0.2))
    model_layers.append(nn.Linear(128, 10))

    return nn.Sequential(*model_layers)

def retrieve_params(df):
    # Loads architectural params calculated and saved within the random search.
    params = {}
    for _, row in df.iterrows():
        key, value = row['keys'], row['values']
        params[key] = value
    return params

'''
This is a helper script, which calculates the amount of parameters (the model size) for each architecture under consideration.
'''
if __name__ == "__main__":
    dir = os.path.dirname(__file__)
    result_path = os.path.join(dir, 'results/')

    # Get list of all the directories of all processed victim architectures
    for _, pmd, _ in os.walk(result_path):
        processed_model_dirs = pmd
        break

    # For every processed victim architecture, calculate the model size
    vms, params = [], []
    for pm in processed_model_dirs:
        # Load architecture parameters
        victim_model_ID = int(pm.split('_')[2])
        architecture_params = pm + "/model_architecture_{}.csv".format(victim_model_ID)
        architecture_params = pd.read_csv(architecture_params, sep=',')
        architecture_params = retrieve_params(architecture_params)

        # Construct model using architecture parameters
        model = define_model(architecture_params)

        # Calculate the size of the model
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        # Those lists can later be used to append the model size as feature for every considered victim architecture
        vms.append(victim_model_ID)
        params.append(param_size)