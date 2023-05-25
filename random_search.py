import torch, random, os
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, TensorDataset, ConcatDataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import optuna
from optuna.trial import TrialState
from optuna.samplers import RandomSampler

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    raise Exception('CUDA could not be loaded. Training time would take too long.')

dir = os.path.dirname(__file__)
data_path = dir + "BaseData/"
base_path = dir + "RandomSearch/"

def training_step(model, batch, criterion):
    # Receives the next training batch from the data loader, calculates the corresponding loss and returns that loss.
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)
    out = model(images)             # Generate predictions
    loss = criterion(out, labels)   # Calculate loss
    return loss

def validation_step(model, batch, criterion):
    # Receives the next validation batch from the data loader, calculates the corresponding loss and accuracy and returns that both as a dict.
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)
    out = model(images)                   # Generate predictions
    loss = criterion(out, labels)         # Calculate loss
    acc = accuracy(out, labels)           # Calculate accuracy
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

def save_model_params(trial, model_count):
    # Saves architectural configuration of model identified within the random search.
    keys = []
    values = []
    for key, value in trial.params.items():
        keys.append(key)
        values.append(value)
    df = pd.DataFrame({"keys": keys, "values": values})
    df.to_csv(base_path + "model_architecture_{}.csv".format(model_count), sep=",", index=False)

def prune_trial(trial, model_count):
    # Saves architectural configuration of models pruned due to overfitting or validation accuracy.
    keys = []
    values = []
    for key, value in trial.params.items():
        keys.append(key)
        values.append(value)
    df = pd.DataFrame({"keys": keys, "values": values}, index=range(len(keys)))
    df.to_csv(base_path + "pruned/pruned_model_architecture_{}.csv".format(model_count), sep=",", index=False)

def define_model(trial):
    # Function to randomly construct the next model as part as the random search conducted using optuna. 

    # Convolution layers per block
    n_conv_layers_b1 = trial.suggest_int("n_conv_layers_b1", 1, 3)
    n_conv_layers_b2 = trial.suggest_int("n_conv_layers_b2", 1, 3)
    n_conv_layers_b3 = trial.suggest_int("n_conv_layers_b3", 1, 3)

    # Convolution kernel size per block
    size_conv_kernel_b1 = trial.suggest_int("size_conv_kernel_b1", 3, 7, step=2)
    size_conv_kernel_b2 = trial.suggest_int("size_conv_kernel_b2", 3, 7, step=2)
    size_conv_kernel_b3 = trial.suggest_int("size_conv_kernel_b3", 3, 7, step=2)

    # Helper to implement starting with 16 feature maps
    feat_map_dec = trial.suggest_int("feat_map_dec", 1, 5)

    # Feature maps in first block. Doubling for every following block.
    feature_maps = trial.suggest_int("feature_maps", 16 if feat_map_dec==1 else 32, 16 if feat_map_dec==1 else 128, step=1 if feat_map_dec==1 else 32)

    # Dropout applied in each block
    dropout_b1 = trial.suggest_float("dropout_b1", 0, 0.5)
    dropout_b2 = trial.suggest_float("dropout_b2", 0, 0.5)
    dropout_b3 = trial.suggest_float("dropout_b3", 0, 0.5)

    # Activation function
    act_func = trial.suggest_categorical("act_func", ["ReLU", "LeakyReLU", "Tanh", "Mish", "ELU"])

    # Normalization method
    '''
    BatchNorm: num_features
    InstanceNorm: num_features
    GroupNorm: num_groups, num_channels
    LayerNorm: normalized_shape
    '''
    norm_method = trial.suggest_categorical("norm_method", ["None", "BatchNorm2d", "GroupNorm", "InstanceNorm2d", "LayerNorm"])

    # Strided convolution instead of MaxPooling after each block
    strided_conv = trial.suggest_int("strided_conv", 0, 1)

    in_features = 3
    layers = []
    # Build block 1
    padding = int((size_conv_kernel_b1-1)/2)
    for i in range(n_conv_layers_b1):
        # Convolution layer
        layers.append(nn.Conv2d(in_features, feature_maps, kernel_size=size_conv_kernel_b1, padding=padding, stride=1))
        # Normalization layer
        if norm_method != "None":
            if norm_method == "GroupNorm":
                num_groups = int(feature_maps/16)   # found by the authors to perform best
                layers.append(getattr(nn, norm_method)(num_groups, feature_maps))
            elif norm_method == "LayerNorm":
                layers.append(getattr(nn, norm_method)(normalized_shape=[feature_maps, 32, 32]))
            else:
                layers.append(getattr(nn, norm_method)(feature_maps))
        # Activation function
        layers.append(getattr(nn, act_func)())
        # Dropout
        layers.append(nn.Dropout(dropout_b1))

        in_features = feature_maps
    if strided_conv == 0:
        layers.append(nn.MaxPool2d(2, 2))
    else:
        # Strided convolution instead of MaxPooling
        layers.append(nn.Conv2d(in_features, feature_maps, kernel_size=size_conv_kernel_b1, padding=padding, stride=2))

    # Build block 2
    padding = int((size_conv_kernel_b2-1)/2)
    feature_maps = feature_maps * 2
    for i in range(n_conv_layers_b2):
        # Convolution layer
        layers.append(nn.Conv2d(in_features, feature_maps, kernel_size=size_conv_kernel_b2, padding=padding, stride=1))
        # Normalization layer
        if norm_method != "None":
            if norm_method == "GroupNorm":
                num_groups = int(feature_maps/16)   # found by the authors to perform best
                layers.append(getattr(nn, norm_method)(num_groups, feature_maps))
            elif norm_method == "LayerNorm":
                layers.append(getattr(nn, norm_method)(normalized_shape=[feature_maps, 16, 16]))
            else:
                layers.append(getattr(nn, norm_method)(feature_maps))
        # Activation function
        layers.append(getattr(nn, act_func)())
        # Dropout
        layers.append(nn.Dropout(dropout_b2))

        in_features = feature_maps
    if strided_conv == 0:
        layers.append(nn.MaxPool2d(2, 2))
    else:
        # Strided convolution instead of MaxPooling
        layers.append(nn.Conv2d(in_features, feature_maps, kernel_size=size_conv_kernel_b2, padding=padding, stride=2))

    # Build block 3
    padding = int((size_conv_kernel_b3-1)/2)
    feature_maps = feature_maps * 2
    for i in range(n_conv_layers_b3):
        # Convolution layer
        layers.append(nn.Conv2d(in_features, feature_maps, kernel_size=size_conv_kernel_b3, padding=padding, stride=1))
        # Normalization layer
        if norm_method != "None":
            if norm_method == "GroupNorm":
                num_groups = int(feature_maps/16)   # found by the authors to perform best
                layers.append(getattr(nn, norm_method)(num_groups, feature_maps))
            elif norm_method == "LayerNorm":
                layers.append(getattr(nn, norm_method)(normalized_shape=[feature_maps, 8, 8]))
            else:
                layers.append(getattr(nn, norm_method)(feature_maps))
        # Activation function
        layers.append(getattr(nn, act_func)())
        # Dropout
        layers.append(nn.Dropout(dropout_b3))

        in_features = feature_maps
    if strided_conv == 0:
        layers.append(nn.MaxPool2d(2, 2))
    else:
        # Strided convolution instead of MaxPooling
        layers.append(nn.Conv2d(in_features, feature_maps, kernel_size=size_conv_kernel_b3, padding=padding, stride=2))

    # Classifier
    layers.append(nn.Flatten())
    layers.append(nn.Linear(in_features*4*4, 1024))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(0.4))
    layers.append(nn.Linear(1024, 128))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(0.2))
    layers.append(nn.Linear(128, 10))
    return nn.Sequential(*layers)

def objective(trial):
    global model_count
    global pruned_model_count
    # Generate model
    model = define_model(trial).to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=eps, weight_decay=wd)

    # Training of the model
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for batch in train_dl:
            loss = training_step(model, batch, criterion)
            train_losses.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation phase
        val_result = evaluate(model, val_dl, criterion)
        train_result = evaluate(model, train_dl_small, criterion)
        val_result['train_loss'] = torch.stack(train_losses).mean().item()
        epoch_end(epoch, val_result, train_result)

        val_accuracy = val_result['acc']
        train_accuracy = train_result['acc']
        train_loss = val_result['train_loss']
        val_loss = val_result['val_loss']
        generalization_gap = train_accuracy - val_accuracy

        trial.report(val_accuracy, epoch)

        # Prune trial if generalisation gap is too large while validation accuracy is still not good enough. 
        if val_accuracy < 0.7 and generalization_gap >= 0.12:
            prune_trial(trial, pruned_model_count)
            pruned_model_count = pruned_model_count + 1
            raise optuna.exceptions.TrialPruned()
        # Prune trial if model is unable to adapt its weights (happens with some architectures). 
        if epoch > 50 and val_accuracy < 0.3:
            prune_trial(trial, pruned_model_count)
            pruned_model_count = pruned_model_count + 1
            raise optuna.exceptions.TrialPruned()
        
        # Break training to save model if validation accuracy is good enough and generalisation gap gets larger than 10%
        if generalization_gap >= 0.1 and val_accuracy >= 0.7:
            break

    if val_accuracy >= 0.7:
        torch.save(model.state_dict(), base_path + "model_state_dict_{}.pth".format(model_count))
        save_model_params(trial, model_count)
        df = pd.DataFrame({"Epoch": epoch, "Val-Loss": val_loss, "Train-Loss": train_loss, "Val-Acc": val_accuracy, "Train-Acc": train_accuracy, "Generalization-Gap": generalization_gap}, index=[0])
        df.to_csv(base_path + "results_model_{}.csv".format(model_count), sep=",", index=True)
        model_count = model_count + 1
    return val_accuracy

if __name__ == "__main__":
    try:
        os.mkdir(base_path)
    except:
        pass
    
    # params in case random search needs to be restarted.
    model_count = 0
    pruned_model_count = 0


    # Initialize robust hyperparameters 
    batch_size = 256
    num_epochs = 150

    lr = 0.0012
    eps = 1.12e-8
    wd = 0.00054

    # Initialize loss function
    criterion = nn.CrossEntropyLoss()

    # Load datasets generated priviously
    datapool = torch.load(data_path + 'datapool.pt')

    # Create training and val datasets
    train_ds, val_ds = random_split(datapool, [30000, 30000])

    # Create small training dataset to get fast training accuracy during training
    train_ds_small, _ = random_split(train_ds, [5000, len(train_ds)-5000])

    print("Length of the target train dataset: {}".format(len(train_ds)))
    print("Length of the val dataset: {}".format(len(val_ds)))

    # Initialize dataloaders
    train_dl = DataLoader(train_ds, batch_size, num_workers=1, shuffle=True, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size, num_workers=1, pin_memory=True)
    train_dl_small = DataLoader(train_ds_small, batch_size, num_workers=1, pin_memory=True)

    # Create optuna random search
    study = optuna.create_study(direction="maximize", sampler=RandomSampler(seed=42))
    study.optimize(objective, n_trials=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))