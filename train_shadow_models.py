import torch, random, os, math, shutil, click
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np

def training_step(model, batch, criterion):
    # Receives the next training batch from the data loader, calculates the corresponding loss and returns that loss.
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    out = model(images)
    loss = criterion(out, labels)
    return loss

def validation_step(model, batch, criterion):
    # Receives the next validation batch from the data loader, calculates the corresponding loss and accuracy and returns that both as a dict.
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    out = model(images)
    loss = criterion(out, labels)
    acc = accuracy(out, labels)
    return {'val_loss': loss.detach(), 'acc': acc}
    
def validation_epoch_end(outputs):
    # Calculates the average accuracy of the model and corresponding loss after each epoch.
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
    batch_accs = [x['acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
    return {'val_loss': epoch_loss.item(), 'acc': epoch_acc.item()}

def epoch_end(epoch, val_result, train_result):
     # Prints statistics of epoch after epoch end.
    print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, train_acc: {:.4f}".format(
        epoch+1, val_result['train_loss'], val_result['val_loss'], val_result['acc'], train_result['acc']))
        
def accuracy(output, labels):
    # Calculates accuracy of received output and ground truth label pairs.
    _, predictions = torch.max(output, dim=1)
    return torch.tensor(torch.sum(predictions == labels).item() / len(predictions))

def evaluate(model, loader, criterion):
    # Evaluates model performance on validation data.
    with torch.no_grad():
        model.eval()
        outputs = [validation_step(model, batch, criterion) for batch in loader]
        return validation_epoch_end(outputs)

def fit(epochs, model, train_loader, val_loader, train_loader_small, optimizer, criterion):
    # Actual training loop. Trains model until its suitable to be considered in this work.
    for epoch in range(epochs):
        # Training phase 
        model.train()
        train_losses = []

        # Iterate trough whole training set and adjust weights
        for batch in train_loader:
            loss = training_step(model, batch, criterion)
            train_losses.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation phase
        val_result = evaluate(model, val_loader, criterion)
        train_result = evaluate(model, train_loader_small, criterion)
        val_result['train_loss'] = torch.stack(train_losses).mean().item()
        if epoch == 0 or epoch%10 == 0:
            epoch_end(epoch, val_result, train_result)

        val_accuracy = val_result['acc']
        train_accuracy = train_result['acc']
        generalization_gap = train_accuracy - val_accuracy

        # Stop training when generalisation gap gets too large and validation accuracy is good enough
        if generalization_gap >= 0.1 and val_accuracy >= 0.7:
            break
        # Else stop training if generalisation gap gets too large to be considered in the experiments
        if generalization_gap >= 0.12:
            break
    
    return generalization_gap, val_accuracy

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

def check_model_quality(gap, acc):
    # In oder to only save suitable models.
    if gap <= 0.12 and acc >= 0.7:
        return True
    else:
        return False

@click.command()
@click.argument('architecture', type=click.INT)
def process_victim_architecture(architecture):
    '''
    This program has been build for a job processing system, receiving an integer value defining the architecture for which 
    the shadow models are to be trained. Architectures have been selected using a prelimary random search.

    Architectures need to be present in sub-dir ../architectures/
    '''
    dir = os.path.dirname(__file__)
    data_path = os.path.join(dir, 'BaseData/')
    architecture_path = os.path.join(dir, 'architectures/')
    result_path = os.path.join(dir, 'results/')

    global device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise Exception('CUDA could not be loaded. Training time would take too long.')

    # Load data sets from ../BaseData
    datapool = torch.load(data_path + 'datapool.pt')
    val_ds = torch.load(data_path + 'val_ds.pt')
    val_ds, _ = random_split(val_ds, [5000, len(val_ds)-5000])

    # Initialize shadow model hyperparameters
    num_shadows = 101                   # 100 shadow models and one victim model. Allows easy calculation of mean attack statistics 
                                        # for multiple victim models.
    train_size = int(len(datapool)/2)   # This way on average every data instance is member in half of the shadow models.
    batch_size = 256
    num_epochs = 150

    # Initialize robust hyperparameters already used in random search
    learning_rate = 0.0012
    weight_decay = 0.00054
    epsilon = 1.12e-8

    # Initialize loss function
    criterion = nn.CrossEntropyLoss()

    # ID of victim model to be processed
    victim_model = architecture
    
    # Processing provided victim architectures
    print('Processing victim architecture ', victim_model)

    # Generating directory to save shadow model data
    working_path = os.path.join(result_path, 'victim_model_{}/'.format(victim_model))
    try: os.mkdir(working_path)
    except: print('Directory already exists!')

    # Copy model architecture in working directory
    victim_architecture = os.path.join(architecture_path, 'model_architecture_{}.csv'.format(victim_model))
    shutil.copyfile(victim_architecture, os.path.join(working_path, 'model_architecture_{}.csv'.format(victim_model)))

    # Load victim architecture parameters
    architecture_params = pd.read_csv(victim_architecture)
    architecture_params = retrieve_params(architecture_params)
    
    shadow_counter = len([file for file in os.listdir(working_path) if 'indices' in file and file.endswith('.csv')])

    # As long there are not 101 trained models, continue training more models
    while(shadow_counter < num_shadows):
        # Subsampling train dataset for i-th shadow model from datapool. Validation dataset will be reused for every shadow model.
        indices = random.sample(range(len(datapool)), train_size)
        indices.sort()
        train_ds = torch.utils.data.Subset(datapool, indices)

        # Create small training dataset in order to get training accuracy during training
        train_ds_small, _ = random_split(train_ds, [5000, len(train_ds)-5000])

        # Initialize dataloaders
        train_dl = DataLoader(train_ds, batch_size, num_workers=1, shuffle=True, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size, num_workers=1, pin_memory=True)
        train_dl_small = DataLoader(train_ds_small, batch_size, num_workers=1, shuffle=True, pin_memory=True)

        # Initialize model
        shadow_model = define_model(architecture_params).to(device)

        # Initialize optimizer
        optimizer = torch.optim.Adam(shadow_model.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=epsilon)

        # Fit model
        print('Fitting shadow model {} with training set length: {} and validation set size: {}'.format(shadow_counter+100, len(train_ds), len(val_ds)))
        gap, acc = fit(num_epochs, shadow_model, train_dl, val_dl, train_dl_small, optimizer, criterion)

        # Save shadow model and its indices if model fulfills requirements
        if check_model_quality(gap, acc):
            torch.save(shadow_model.state_dict(), working_path + "shadow_model{}.pth".format(shadow_counter+100))
            members = []
            for j in range(len(datapool)):
                members.append(True) if j in indices else members.append(False)
            df = pd.DataFrame({"members": members})
            df.to_csv(working_path + "indices{}.csv".format(shadow_counter+100), sep=",", index=False)
            shadow_counter += 1
        else:
            continue

if __name__ == '__main__':
    # Needs to have this format due to the library click.
    process_victim_architecture()
