#### dataset: UCI Machine Learning Repository_Cuff-Less Blood Pressure Estimation Data Set
#### data address: https://archive.ics.uci.edu/ml/datasets/Cuff-Less+Blood+Pressure+Estimation
#### Citation Request: M. Kachuee, M. M. Kiani, H. Mohammadzade, M. Shabany, Cuff-Less High-Accuracy Calibration-Free Blood Pressure Estimation Using Pulse Transit Time, IEEE International Symposium on Circuits and Systems (ISCAS'15), 2015.
####    The data set is in matlab's v7.3 mat file, accordingly it should be opened using new versions of matlab or HDF libraries in other environments.(Please refer to the Web for more information about this format) 
####    This database consist of a cell array of matrices, each cell is one record part. 
####    In each matrix each row corresponds to one signal channel: 
####        1: PPG signal, FS=125Hz; photoplethysmograph from fingertip 
####        2: ABP signal, FS=125Hz; invasive arterial blood pressure (mmHg) 
####        3: ECG signal, FS=125Hz; electrocardiogram from channel II


################################################################################################################################################################################


import numpy as np
import h5py


################################################################################################################################################################################



## Open the data

#### Open big '.mat' data and check the shape:

def open_data(file_path = '/Users/jinyanwei/Desktop/BP_Model/Data/UCI/Part_1.mat'):

    with h5py.File(file_path, 'r') as file:
        references = np.array(file[list(file.keys())[1]]) #### Access a specific variable and convert it to a NumPy array
       
        # Dereference the objects and store them in a list
        data = []
        for ref in references.flat:
            dereferenced_object = np.array(file[ref])
            data.append(dereferenced_object)

    '''    
    #### Show the shape of the data
    for i, array in enumerate(data):
        print(f"Shape of data[{i}]: {array.shape}")
    '''
    
    return data    
        


################################################################################################################################################################################

## Open data in small.mat
'''
import scipy.io as scio
data = scio.loadmat("/Users/jinyanwei/Desktop/BP_Model/Data/UCI/Part_1.mat")'''


################################################################################################################################################################################



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



################################################################################################################################################################################
