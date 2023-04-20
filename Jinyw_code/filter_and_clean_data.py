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


import sys
import numpy as np
import pandas as pd
import h5py
import os
import re
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.signal as signal
import scipy.io as sio
import scipy.stats
import neurokit2 as nk
import heartpy as hp
import nolds
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import warnings




################################################################################################################################################################################



def clean_data(data, number):
    df = pd.DataFrame(data, columns=(('PPG','BP','ECG')))
    ppg_data = df['PPG']
    bp_data = df['BP']
    ecg_data = df['ECG']

    ## Filter signals
    sampling_rate = 125
    ppg_filtered = nk.signal_filter(ppg_data, lowcut=0.5, highcut=50, method='butterworth', order=2, sampling_rate=sampling_rate)
    ecg_filtered = nk.signal_filter(ecg_data, lowcut=0.5, highcut=50, method='butterworth', order=2, sampling_rate=sampling_rate)
    
    filtered_df = df
    filtered_df['PPG'], filtered_df['ECG'] = ppg_filtered, ecg_filtered
    

    ## Clean data: make every beat only have one PPG/BP peak
    #### Find ECG R-peaks
    ecgpeaks, _ = signal.find_peaks(ecg_filtered, distance=sampling_rate//2.5)

    #### Make sure onle one PPG peak and one BP peak between two adjacent ECG R-peaks
    cleaned_df = pd.DataFrame()
    sum_beats = 0
    times_recorder = []

    for R_peak_number in range(len(ecgpeaks)-1):
        sum_beats += 1
        if ecg_filtered[ecgpeaks[R_peak_number]] > 0.2 and ecg_filtered[ecgpeaks[R_peak_number + 1]] > 0.2: # make sure the peaks are R-peak
            onebeat_ppgpeak, _ = signal.find_peaks(ppg_filtered[ecgpeaks[R_peak_number]:ecgpeaks[R_peak_number + 1]], distance = sampling_rate//2.5)
            onebeat_bppeak, _ = signal.find_peaks(bp_data[ecgpeaks[R_peak_number]:ecgpeaks[R_peak_number + 1]], distance = sampling_rate//2.5)
            if len(onebeat_ppgpeak) == 1 and ppg_filtered[ecgpeaks[R_peak_number] + onebeat_ppgpeak] > 0 and len(onebeat_bppeak) == 1 : # make sure only one BP signal for one beat 
                cleaned_df = pd.concat([cleaned_df, filtered_df[ecgpeaks[R_peak_number]:ecgpeaks[R_peak_number + 1]]])
                times_recorder.append(sum_beats)                
    
    #### Reset indexs:
    cleaned_df = cleaned_df.reset_index()

    if len(cleaned_df) > 18000:
        return cleaned_df.to_csv(f'/Users/jinyanwei/Desktop/BP_Model/Model_record/cleaned_data/Part1_cleaned{number}.csv')
    else:
        return
    
    
    


################################################################################################################################################################################


'''  

#### show the filtered PPG and ECG signals
ppg_df = pd.DataFrame(columns=(('PPG_original', 'PPG_filtered')))
ppg_df['PPG_original'] = ppg_data
ppg_df['PPG_filtered'] = ppg_filtered
display(ppg_df)
figPPG = px.line(ppg_df)
figPPG.show()

ecg_df = pd.DataFrame(columns=(('ECG_original', 'ECG_filtered')))
ecg_df['ECG_original'] = ecg_data
ecg_df['ECG_filtered'] = ecg_filtered
display(ecg_df)
figECG = px.line(ecg_df)
figECG.show()

'''



'''
#### show peaks in PPG and ECG signals

ppgpeaks, _ = signal.find_peaks(ppg_filtered, distance=sampling_rate//2.5)
print(ppgpeaks, len(ppgpeaks))
plt.figure(figsize=(50, 10))
for index in ppgpeaks:
    plt.scatter(index, ppg_filtered[index], marker="*")
plt.plot(ppg_filtered)
plt.show()

print(ecgpeaks, len(ecgpeaks))
plt.figure(figsize=(50, 10))
for index in ecgpeaks:
    plt.scatter(index, ecg_filtered[index], marker="*")
plt.plot(ecg_filtered)
plt.show()

'''




'''
#### Show  original cleaned data:

print('All beats: ', sum_beats, ', Reserved beats: ', len(times_recorder))
display(cleaned_df)
figcleaned = px.line(cleaned_df)
figcleaned.show()


#### Show final cleaned data:

display(cleaned_df)
figcleaned = px.line(cleaned_df)
figcleaned.show()

'''


#### Save acceptable cleaned data to Part_Cleaned_.cvs
#### Warning!!!! NEVER FORGET to change the name of the csv!!!!!!!!!


################################################################################################################################################################################