import torch, random, os, math
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

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

def get_data(model, ds, criterion, members, i):
    # Calculates individual loss values for every data point for received model and dataset
    losses, labels = [],[]
    model.eval()
    with torch.no_grad():
        for elem in ds:
            image = elem[0].unsqueeze(0).cpu()
            label = torch.tensor(elem[1]).view(1).cpu()
            image, label = image.to(device), label.to(device)
            labels.append(int(label[0]))

            # Get model output
            out = model(image)

            # Calculate loss for Sablayrolles19 attack
            loss = criterion(out, label)
            losses.append(loss.item())

            del image, label, out, loss
    
    s19_df = pd.DataFrame(np.column_stack([members, losses]), columns=['members', 'losses{}'.format(i)], index=range(len(ds)))
    return s19_df

def retrieve_params(df):
    # Loads architectural params calculated and saved within the random search.
    params = {}
    for _, row in df.iterrows():
        key, value = row['keys'], row['values']
        params[key] = value
    return params

if __name__ == '__main__':
    '''
    Queries every shadow model of received victim architecture with its own train and test dataset, caluculating individual
    loss values for every data point. Saves individual loss values with corresponding membership status for every shadow model 
    as csv. 
    '''
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise Exception('CUDA could not be loaded. Training time would take too long.')

    # ID of victim model whose shadow models are to be queried.
    victim_model = 1

    dir = os.path.dirname(__file__)
    data_path = os.path.join(dir, 'BaseData/')
    result_path = os.path.join(dir, 'results/')
    base_path = os.path.join(result_path, 'victim_model_{}/'.format(victim_model))
    mia_data_path = base_path + 'mia_data/'

    try: os.mkdir(mia_data_path)
    except: pass

    datapool = torch.load(data_path + 'datapool.pt')

    num_shadow_models = len([file for file in os.listdir(base_path) if file.endswith('.pth')])
    print('Retrieving MIA data for ', num_shadow_models, ' shadow models...')

    architecture_params = pd.read_csv(base_path + '/model_architecture_{}.csv'.format(victim_model))
    architecture_params = retrieve_params(architecture_params)

    criterion = nn.CrossEntropyLoss()

    for i in range(0, num_shadow_models):
        print('Evaluation shadow model ', i, '...')
        model = define_model(architecture_params)
        model.load_state_dict(torch.load(base_path + 'shadow_model{}.pth'.format(i)))
        model.to(device)
        model.eval()

        indices = pd.read_csv(base_path + 'indices{}.csv'.format(i))

        s19_df, s17_df = get_data(model, datapool, criterion, indices, i)

        try: 
            os.mkdir(mia_data_path + 's19/')
        except: pass

        s19_df.to_csv(mia_data_path + 's19/sab19_data{}.csv'.format(i), sep=',', index=False)