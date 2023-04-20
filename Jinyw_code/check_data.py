df = pd.DataFrame(all_data[patient_number], columns=(('PPG','BP','ECG')))
display(df)
ppg_data = df['PPG']
bp_data = df['BP']
ecg_data = df['ECG']

## Filter signals
sampling_rate = 125
ppg_filtered = nk.signal_filter(ppg_data, lowcut=0.5, highcut=50, method='butterworth', order=2, sampling_rate=sampling_rate)
ecg_filtered = nk.signal_filter(ecg_data, lowcut=0.5, highcut=50, method='butterworth', order=2, sampling_rate=sampling_rate)

filtered_df = df
filtered_df['PPG'], filtered_df['ECG'] = ppg_filtered, ecg_filtered

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

## Clean data: make every beat only have one PPG/BP peak
#### Find ECG R-peaks
ecgpeaks, _ = signal.find_peaks(ecg_filtered, distance=sampling_rate//2.5)

#### show peaks in ECG signals

print(ecgpeaks, len(ecgpeaks))
plt.figure(figsize=(50, 10))
for index in ecgpeaks:
    plt.scatter(index, ecg_filtered[index], marker="*")
plt.plot(ecg_filtered)
plt.show()

#### Make sure onle one PPG peak and one BP peak between two adjacent ECG R-peaks
cleaned_df = pd.DataFrame()
sum_beats = 0
times_recorder = []

for R_peak_number in range(len(ecgpeaks)-1):
    sum_beats += 1
    if ecg_filtered[ecgpeaks[R_peak_number]] > 0.4 and ecg_filtered[ecgpeaks[R_peak_number + 1]] > 0.4: # make sure the peaks are R-peak
        onebeat_ppgpeak, _ = signal.find_peaks(ppg_filtered[ecgpeaks[R_peak_number]:ecgpeaks[R_peak_number + 1]], distance = sampling_rate//2.5)
        onebeat_bppeak, _ = signal.find_peaks(bp_data[ecgpeaks[R_peak_number]:ecgpeaks[R_peak_number + 1]], distance = sampling_rate//2.5)
        if len(onebeat_ppgpeak) == 1 and ppg_filtered[ecgpeaks[R_peak_number] + onebeat_ppgpeak] > 0 and len(onebeat_bppeak) == 1 : # make sure only one BP signal for one beat 
            cleaned_df = pd.concat([cleaned_df, filtered_df[ecgpeaks[R_peak_number]:ecgpeaks[R_peak_number + 1]]])
            times_recorder.append(sum_beats)                

#### Reset indexs:
cleaned_df = cleaned_df.reset_index()

#### Show  original cleaned data:

print('All beats: ', sum_beats, ', Reserved beats: ', len(times_recorder))
display(cleaned_df)
figcleaned = px.line(cleaned_df)
figcleaned.show()


#### Show final cleaned data:

display(cleaned_df)
figcleaned = px.line(cleaned_df)
figcleaned.show()

df = cleaned_df
ppg_data = df['PPG']
bp_data = df['BP']
ecg_data = df['ECG']

## Segment signals from R-peak to get data of each beat.
sampling_rate = 125
ecgpeaks, _ = signal.find_peaks(ecg_data, distance=sampling_rate//2.5)

ppg_segments, ecg_segments, SBPlist, DBPlist, bplist = [], [], [], [], []

for peak_number in range(1, len(ecgpeaks)-2):    # data will be more stable from the second R-peak
    ppg_segments.append(ppg_data[ecgpeaks[peak_number]:ecgpeaks[peak_number + 1]].values)
    ecg_segments.append(ecg_data[ecgpeaks[peak_number]:ecgpeaks[peak_number + 1]].values)
    bplist = bp_data[ecgpeaks[peak_number]:ecgpeaks[peak_number+1]]
    SBPlist.append(max(bplist))
    DBPlist.append(min(bplist))
## Extract features of PPG and ECG.

#### Def features:

from numpy.fft import fft

def extract_ppg_features(signal):

    '''
    #### Chaotic features

    lyap_r = nolds.lyap_r(signal)
    hurst_exp = nolds.hurst_rs(signal)
    corr_dim = nolds.corr_dim(signal, 1)

    '''

    #### Time domain features
    mean = np.mean(signal)
    std_dev = np.std(signal)
    skewness = scipy.stats.skew(signal)
    kurtosis = scipy.stats.kurtosis(signal)

    #### Frequency domain features
    fft_values = fft(signal)
    power_spectrum = np.abs(fft_values)**2
    total_power = np.sum(power_spectrum)
    low_freq_power = np.sum(power_spectrum[:len(power_spectrum)//2]) / total_power
    high_freq_power = np.sum(power_spectrum[len(power_spectrum)//2:]) / total_power

    ppg_features = {
        #'lyap_r': lyap_r,
        #'hurst_exp': hurst_exp,
        #'corr_dim': corr_dim,
        'ppg_mean': mean,
        'ppg_std_dev': std_dev,
        'ppg_skewness': skewness,
        'ppg_kurtosis': kurtosis,
        'ppg_low_freq_power': low_freq_power,
        'ppg_high_freq_power': high_freq_power
    }

    return ppg_features


def extract_ecg_features(signal):

    '''
    #### Chaotic features

    lyap_r = nolds.lyap_r(signal)
    hurst_exp = nolds.hurst_rs(signal)
    corr_dim = nolds.corr_dim(signal, 1)

    '''

    # Time domain features
    mean = np.mean(signal)
    std_dev = np.std(signal)
    skewness = scipy.stats.skew(signal)
    kurtosis = scipy.stats.kurtosis(signal)

    # Frequency domain features
    fft_values = fft(signal)
    power_spectrum = np.abs(fft_values)**2
    total_power = np.sum(power_spectrum)
    low_freq_power = np.sum(power_spectrum[:len(power_spectrum)//2]) / total_power
    high_freq_power = np.sum(power_spectrum[len(power_spectrum)//2:]) / total_power

    ecg_features = {
        #'lyap_r': lyap_r,
        #'hurst_exp': hurst_exp,
        #'corr_dim': corr_dim,
        'ecg_mean': mean,
        'ecg_std_dev': std_dev,
        'ecg_skewness': skewness,
        'ecg_kurtosis': kurtosis,
        'ecg_low_freq_power': low_freq_power,
        'ecg_high_freq_power': high_freq_power
    }

    return ecg_features


#### Get features and save to csv:
ppg_feature_list = [extract_ppg_features(ppg_segment) for ppg_segment in ppg_segments]
ecg_feature_list = [extract_ecg_features(ecg_segment) for ecg_segment in ecg_segments]
features_df = pd.concat([pd.DataFrame(ppg_feature_list), pd.DataFrame(ecg_feature_list), pd.DataFrame({'SBP': SBPlist, 'DBP':DBPlist})], axis=1)

display(features_df)


################################################################################################################################################################################


