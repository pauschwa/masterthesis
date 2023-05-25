import torch, random, os, math, re, sklearn
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calc_mean(df, cols):
    # Calculates mean loss of every data point of the received dataframe.
    loss_means = []
    for _, row in df.iterrows():
        losses = [loss for loss in row[cols]]
        loss_means.append(np.nanmean(losses))
    return loss_means

if __name__ == '__main__':
    '''
    This script receives an array of victim architecture IDs, whose shadow model data is to be processed further.
    It randomly chooses 20 shadow models to be treated as victim model and generates attack data sets for these victim
    models. The attack data sets consist of in-means, out-means and the occurred loss for every data point.
    '''
    dir = os.path.dirname(__file__)

    # Array of victim model IDs whose shadow model data is to be condensed into the final mia dataframe.
    model_evaluations = []

    for model_evaluation in model_evaluations:
        result_path = os.path.join(dir, 'results/')
        base_path = os.path.join(result_path, 'victim_model_{}/'.format(model_evaluation))
        shadow_data_path = base_path + 'mia_data/s19/'
        mia_data_path = base_path + '/attack_datasets/'
        
        try: os.mkdir(mia_data_path)
        except: pass
        
        num_shadow_models = len([file for file in os.listdir(shadow_data_path) if file.endswith('.csv')])
        print('Conducting MIAs based on {} shadow models'.format(num_shadow_models-1))

        try: processed_victim_models = [file for file in os.listdir(mia_data_path + '{}/'.format(model_evaluation)) if file.endswith('.csv')]
        except: processed_victim_models = []

        if len(processed_victim_models) < 20:
            victim_model_list = random.sample(range(0, num_shadow_models), 20-len(processed_victim_models))
            victim_model_list.sort()
            print('Victim models: ', victim_model_list)

            # Preprocessing data
            for victim_model in victim_model_list:
                print('Processing victim: ', victim_model)

                in_data = pd.DataFrame(index=range(60000))
                out_data = pd.DataFrame(index=range(60000))

                for file in os.listdir(shadow_data_path):
                    if file.endswith('.csv'):
                        if file != 'sab19_data{}.csv'.format(victim_model):
                            data_df = pd.read_csv(os.path.join(shadow_data_path, file), sep=',')
                            members = pd.Series(data_df['members'].tolist(), dtype='bool')
                            data_df = data_df.drop('members', axis=1)

                            in_df = data_df[members.values]
                            out_df = data_df[~members.values]

                            in_data = pd.concat([in_data, in_df], axis=1)
                            out_data = pd.concat([out_data, out_df], axis=1)
                        else: 
                            print('Skipping victim models data ...')
                            victim_df = pd.read_csv(os.path.join(shadow_data_path, file), sep=',')
                            victim_df = victim_df.rename({'losses{}'.format(victim_model): 'victim_losses'}, axis=1)
                
                loss_cols = [col for col in in_data.columns if 'losses' in col]

                in_loss_means = calc_mean(in_data, loss_cols)
                out_loss_means = calc_mean(out_data, loss_cols)

                mia_df = pd.DataFrame(np.column_stack([in_loss_means, out_loss_means]), 
                                    columns=['in_mean', 'out_mean'], 
                                    index=range(60000))
                
                mia_df = pd.concat([mia_df, victim_df], axis=1)

                try: os.mkdir(mia_data_path + '{}'.format(model_evaluation))
                except: pass
                mia_df.to_csv(mia_data_path + '{}/'.format(model_evaluation) + '{}.csv'.format(victim_model), sep=',', index=False)