import os
import pandas as pd
import numpy as np
from pathlib import Path

'''
This script compares the vulnerability to MIA of the model built for maximum resistance of outlier data points against the average outlier data point vulnerability of all considered models.
'''
if __name__ == "__main__":
    dir = Path(os.path.dirname(__file__))
    dir = dir.parent.absolute()
    attack_results_data = os.path.join(dir, 'attack_results.csv')
    resistant_model_data = os.path.join(dir, 'resistant_model_results.csv')

    all_data = pd.read_csv(attack_results_data, sep=',')
    resistant_data = pd.read_csv(resistant_model_data, sep=',')

    tpr01 = all_data['TPR01'].tolist()
    tpr_01_1_perc = all_data['TPR01_600'].tolist()
    attack_acc_list = all_data['attack_acc'].tolist()
    attack_acc_1_perc_list = all_data['attack_acc_600'].tolist()
      
    print('Average TPR (all): {.2f} ||| Resistant model TPR (all): {.2f}'.format(np.mean(tpr01_list)*100, float(resistant_data['TPR01'])*100))
    print('Average TPR (outlier): {.2f} ||| Resistant model TPR (outlier): {.2f}'.format(np.mean(tpr_01_1_perc_list)*100, float(resistant_data['TPR01_600'])*100))
    print('Average TPR (outlier): {.2f} ||| Resistant model TPR (outlier): {.2f}'.format(np.mean(attack_acc_list)*100, float(resistant_data['attack_acc'])*100))
    print('Average TPR (outlier): {.2f} ||| Resistant model TPR (outlier): {.2f}'.format(np.mean(attack_acc_1_perc_list)*100, float(resistant_data['attack_acc_600'])*100))

    # Calculate amount of models, which were more resistant with respect to the general vulnerability to MIA.
    tmp_list = [i for i in tpr01_list if i < float(resistant_data['TPR01'])]
    print('TPR (all):                  {} models were more resistant: '.format(len(tmp_list)))
    tmp_list = [i for i in attack_acc_list if i < float(resistant_data['attack_acc'])]
    print('Attack Accuracy (all):      {} models were more resistant: '.format(len(tmp_list)))

    # Calculate amount of models, which were more resistant with respect to the vulnerability of outlier data points to MIA.
    tmp_list = [i for i in tpr_01_1_perc_list if i < float(resistant_data['TPR01_600'])]
    print('TPR (outlier):              {} models were more resistant: '.format(len(tmp_list)))
    tmp_list = [i for i in attack_acc_1_perc_list if i < float(resistant_data['attack_acc_600'])]
    print('Attack Accuracy (outlier):  {} models were more resistant: '.format(len(tmp_list)))