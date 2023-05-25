import torch, random, os, math, re, sklearn
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

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

def evaluate(model, loader, criterion):
    # Evaluates model performance on validation data.
    with torch.no_grad():
        model.eval()
        outputs = [validation_step(model, batch, criterion) for batch in loader]
        return validation_epoch_end(outputs)

def accuracy(output, labels):
    # Calculates accuracy of received output and ground truth label pairs.
    _, predictions = torch.max(output, dim=1)
    return torch.tensor(torch.sum(predictions == labels).item() / len(predictions))

def get_val_train_acc(model, vm):
    # Calculates mean validation and training accuracy of victim model.

    datapool = torch.load(data_path + 'datapool.pt')
    val_ds = torch.load(data_path + 'val_ds.pt')
    val_ds, _ = random_split(val_ds, [5000, len(val_ds)-5000])

    global device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise Exception('CUDA could not be loaded. Validation time would take too long.')

    val_dl = DataLoader(val_ds, batch_size=256, num_workers=1, pin_memory=True)
    criterion = nn.CrossEntropyLoss()

    model.load_state_dict(torch.load(victim_model_data + '/shadow_model{}.pth'.format(vm)))
    model.to(device)
    indices = pd.read_csv(victim_model_data + 'indices{}.csv'.format(vm))['members']

    members = []
    for i, status in enumerate(indices):
        if status == True:
            members.append(i)

    members = random.sample(members, 5000)
    train_ds_small = torch.utils.data.Subset(datapool, members)
    train_dl_small = DataLoader(train_ds_small, batch_size=256, num_workers=1, shuffle=True, pin_memory=True)

    # Validation phase
    model.eval()
    val_result = evaluate(model, val_dl, criterion)
    train_result = evaluate(model, train_dl_small, criterion)

    return val_result['acc'], train_result['acc']

def calc_roc_characteristics(ans, sco):
    fpr, tpr, _ = roc_curve(np.array(ans), np.array(sco), drop_intermediate=True)
    acc = np.max(1-(fpr+(1-tpr))/2)
    auc = sklearn.metrics.auc(fpr, tpr)
    low01 = tpr[np.where(fpr<.001)[0][-1]]
    low1 = tpr[np.where(fpr<.01)[0][-1]]

    return acc, auc, low01, low1, fpr, tpr

def return_tuple_as_lists(tuples):
    ans, sco = [],[]
    for score, answer in tuples:
        sco.append(score)
        ans.append(answer)
    
    return ans, sco

def average_pr_array(tpr_aca, fpr_aca):
    # aca: array containing arrays
    same_length = []

    for i in range(len(tpr_aca)):
        tpr = tpr_aca[i]
        fpr = fpr_aca[i]

        pr_same_length = []
        for t in np.arange(0.0001, 1, 0.0001):
            n_tpr = tpr[np.where(fpr<t)[0][-1]]
            pr_same_length.append(n_tpr)
        same_length.append(pr_same_length)

    averaged_tpr = []
    for i in range(len(same_length[0])):
        tmp = []
        for elem in same_length:
            tmp.append(elem[i])
        averaged_tpr.append(np.mean(tmp))
    
    return averaged_tpr

def save_tpr_fpr_graph(tpr, tpr_600, tpr_3000, tpr_6000, path):
    fpr_range = np.arange(0.0001, 1, 0.0001)
    plt.plot(fpr_range, averaged_tpr, label="Standard", color="red")
    plt.plot(fpr_range, averaged_tpr_600, label="1% most confident", color="green")
    plt.plot(fpr_range, averaged_tpr_3000, label="5% most confident", color="blue")
    plt.plot(fpr_range, averaged_tpr_6000, label="10% most confident", color="orange")

    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-4,1)
    plt.ylim(1e-4,1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls='--', color='gray')
    plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
    plt.legend(fontsize=12)
    plt.savefig(path + 'averaged_tpr_fpr_plot.png')
    plt.clf()


if __name__ == "__main__":
        '''
        This script receives an array of victim architecture IDs, which are to be attacked using the data generated previously.
        It attacks every of the 20 victim models by comparing the loss value of every data point to the respective in and out 
        loss distribution.
        '''
        dir = os.path.dirname(__file__)
        global data_path
        data_path = os.path.join(dir, 'BaseData/')
        result_path = os.path.join(dir, 'results/')
        attack_result_path = os.path.join(dir, 'attack_results/')

        try: os.mkdir(attack_result_path)
        except: pass
        
        model_evaluations = []

        for model_evaluation in model_evaluations:
            global victim_model_data, mia_path, model_architecture
            victim_model_data = result_path + 'victim_model_{}/'.format(model_evaluation)
            mia_path = victim_model_data + '/attack_datasets/{}/'.format(model_evaluation)
            model_architecture = victim_model_data + 'model_architecture_{}.csv'.format(model_evaluation)

            architecture_params = pd.read_csv(model_architecture)
            architecture_params = retrieve_params(architecture_params)
            model = define_model(architecture_params)
        
            val_accs, train_accs = [],[]
            tpr_01, tpr_1, attack_accs, aucs, fprs, tprs = [],[],[],[],[],[]
            tpr_01_600, tpr_1_600, attack_accs_600, aucs_600, fpr_600, tpr_600 = [],[],[],[],[],[]
            tpr_01_3000, tpr_1_3000, attack_accs_3000, aucs_3000, fpr_3000, tpr_3000 = [],[],[],[],[],[]
            tpr_01_6000, tpr_1_6000, attack_accs_6000, aucs_6000, fpr_6000, tpr_6000 = [],[],[],[],[],[]

            for file in os.listdir(mia_path):
                if file.endswith('.csv'):
                    victim_model = int(file.split('.')[0])

                    val_acc, train_acc = get_val_train_acc(model, victim_model)
                    val_accs.append(val_acc)
                    train_accs.append(train_acc)

                    mia_df = pd.read_csv(mia_path + '{}.csv'.format(victim_model), sep=',')
                    answers = mia_df['members']
                    mia_df = mia_df.drop(['members'], axis=1)
                
                    membership_scores = []
                    for _, row in mia_df.iterrows():
                        in_loss, out_loss, victim_loss = row['in_mean'], row['out_mean'], row['victim_losses']

                        threshold = (in_loss + out_loss)/2
                        membership_score = -victim_loss + threshold
                        membership_scores.append(membership_score)

                    acc, auc, low01, low1, fpr, tpr = calc_roc_characteristics(answers, membership_scores)

                    tpr_01.append(low01)
                    tpr_1.append(low1)
                    attack_accs.append(acc)
                    aucs.append(auc)
                    fprs.append(fpr)
                    tprs.append(tpr)

                    # Performing attack on n-highest scored data points
                    tuples = []
                    for s, an in zip(membership_scores, answers):
                        tuples.append((s,an))

                    tuples.sort(key=lambda an: abs(an[0]), reverse=True)
                    tuples600 = tuples[:600]
                    tuples3000 = tuples[:3000]
                    tuples6000 = tuples[:6000]

                    # For 1% of data samples with highest confidence
                    ans, scores = return_tuple_as_lists(tuples600)
                    acc, auc, low01, low1, fpr, tpr = calc_roc_characteristics(ans, scores)
                    tpr_01_600.append(low01)
                    tpr_1_600.append(low1)
                    attack_accs_600.append(acc)
                    aucs_600.append(auc)
                    fpr_600.append(fpr)
                    tpr_600.append(tpr)
                    
                    # For 5% of data samples with highest confidence
                    ans, scores = return_tuple_as_lists(tuples3000)
                    acc, auc, low01, low1, fpr, tpr = calc_roc_characteristics(ans, scores)
                    tpr_01_3000.append(low01)
                    tpr_1_3000.append(low1)
                    attack_accs_3000.append(acc)
                    aucs_3000.append(auc)
                    fpr_3000.append(fpr)
                    tpr_3000.append(tpr)
                    
                    # For 10% of data samples with highest confidence
                    ans, scores = return_tuple_as_lists(tuples6000)
                    acc, auc, low01, low1, fpr, tpr = calc_roc_characteristics(ans, scores)
                    tpr_01_6000.append(low01)
                    tpr_1_6000.append(low1)
                    attack_accs_6000.append(acc)
                    aucs_6000.append(auc)
                    fpr_6000.append(fpr)
                    tpr_6000.append(tpr)

            averaged_tpr = average_pr_array(tprs, fprs)
            averaged_tpr_600 = average_pr_array(tpr_600, fpr_600)
            averaged_tpr_3000 = average_pr_array(tpr_3000, fpr_3000)
            averaged_tpr_6000 = average_pr_array(tpr_6000, fpr_6000)
            
            victim_model_attack_result_path = attack_result_path + 'ID_{}'.format(model_evaluation)

            save_tpr_fpr_graph(averaged_tpr, averaged_tpr_600, averaged_tpr_3000, averaged_tpr_6000, victim_model_attack_result_path)

            result_df = pd.DataFrame({
                'val_acc': np.mean(val_accs), 'train_acc': np.mean(train_accs),
                'TPR1': np.mean(tpr_1), 'TPR01': np.mean(tpr_01), 'attack_acc': np.mean(attack_accs), 'auc': np.mean(aucs),
                'TPR1_600': np.mean(tpr_1_600), 'TPR01_600': np.mean(tpr_01_600), 'attack_acc_600': np.mean(attack_accs_600), 'auc_600': np.mean(aucs_600),
                'TPR1_3000': np.mean(tpr_1_3000), 'TPR01_3000': np.mean(tpr_01_3000), 'attack_acc_3000': np.mean(attack_accs_3000), 'auc_3000': np.mean(aucs_3000),
                'TPR1_6000': np.mean(tpr_1_6000), 'TPR01_6000': np.mean(tpr_01_6000), 'attack_acc_6000': np.mean(attack_accs_6000), 'auc_6000': np.mean(aucs_6000),
            }, index=[1])
            result_df.to_csv(victim_model_attack_result_path + 'results.csv', sep = ',')

            pr_df = pd.DataFrame({'tpr': averaged_tpr, 'tpr_600': averaged_tpr_600, 'tpr_3000': averaged_tpr_3000, 'tpr_6000': averaged_tpr_6000})
            pr_df.to_csv(victim_model_attack_result_path + 'tpr_fpr_rates.csv')