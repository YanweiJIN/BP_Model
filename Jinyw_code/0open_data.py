




import numpy as np
from numpy import *
import pandas as pd
import h5py
import os
import re
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.signal as signal
import scipy.io as sio
import scipy.stats
from scipy.fft import fft
import neurokit2 as nk
import heartpy as hp
import nolds
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import warnings



################################################################################################################################################################################



## Open the data

#### Open big '.mat' data and check the shape:

file_path = '/Users/jinyanwei/Desktop/BP_Model/Data/UCI/Part_1.mat'  

with h5py.File(file_path, 'r') as f:
    print(list(f.keys()))  # List all the variables in the file

    # Access a specific variable and convert it to a NumPy array
    variable_name = 'Part_1'  # Replace with the name of the variable you want to access
    references = np.array(f[variable_name])

    # Dereference the objects and store them in a list
    data = []
    for ref in references.flat:
        dereferenced_object = np.array(f[ref])
        data.append(dereferenced_object)

for i, array in enumerate(data):
    print(f"Shape of data[{i}]: {array.shape}")



'''
#### Merge many patients into one df

Include_Data = []
df = pd.DataFrame(columns=(('PPG','BP', 'ECG')))

for patient in range(len(data)):
    if len(data[patient]) > 7499:
        data_df = pd.DataFrame(data[patient][:1000], columns=(('PPG','BP', 'ECG')))
        df = pd.concat([df, data_df])
        Include_Data.append(patient)

display(df)

#### Open data:

RMSE_SBP, RMSE_DBP, MAE_SBP, MAE_DBP, Abnormal_location= [], [], [], [], []

for patient in range(10):
    if len(data[patient]) > 999:
        df = pd.DataFrame(data[patient], columns=(('PPG','BP', 'ECG'))) #选取前10000条数据
        ecg_data = df['ECG']
        ppg_data = df['PPG']
        bp_data = df['BP']

        #### Filter signals and ...

        .
        .
        .
        .
        .
        .

        #### Calculate root mean squared error and mean absolute error for both SBP and DBP
        rmse_sbp = metrics.mean_squared_error(y_test["SBP"], y_pred[:, 0])**0.5
        rmse_dbp = metrics.mean_squared_error(y_test["DBP"], y_pred[:, 1])**0.5
        mae_sbp = metrics.mean_absolute_error(y_test["SBP"], y_pred[:, 0])
        mae_dbp = metrics.mean_absolute_error(y_test["DBP"], y_pred[:, 1])
        
        if rmse_sbp > 5 or rmse_dbp >5:
            Abnormal_location.append(patient)

        RMSE_SBP.append(rmse_sbp)
        RMSE_DBP.append(rmse_dbp)
        MAE_SBP.append(mae_sbp)
        MAE_DBP.append(mae_dbp)

print('RMSE_SBP: ', RMSE_SBP)
print('RMSE_DBP: ', RMSE_DBP)
print('MAE_SBP: ', MAE_SBP)
print('MAE_DBP: ', MAE_DBP)
print('Abnormal location: ', Abnormal_location)
print('RMSE_SBP: ', mean(RMSE_SBP), 'RMSE_DBP', mean(RMSE_DBP), 'MAE_SBP: ', mean(MAE_SBP), 'MAE_DBP', mean(MAE_DBP))

'''