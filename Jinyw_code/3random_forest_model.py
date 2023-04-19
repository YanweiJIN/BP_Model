#### Random Forest Model
#### Please take a look on the Features Data before use it


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


df = pd.read_csv('/Users/jinyanwei/Desktop/BP_Model/Model_record/features_data/Part1_feature0.csv')
features_df = df.iloc[:, 1:-2]
bp_df = df.iloc[:, -2:]

X_scaled = features_df.values
y = bp_df

#### Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#### Initialize the random forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

#### Train the model on the training data
model.fit(X_train, y_train)

#### Predict on the test set
y_pred = model.predict(X_test)

#### Calculate root mean squared error and mean absolute error for both SBP and DBP
rmse_sbp = metrics.mean_squared_error(y_test["SBP"], y_pred[:, 0])**0.5
rmse_dbp = metrics.mean_squared_error(y_test["DBP"], y_pred[:, 1])**0.5
mae_sbp = metrics.mean_absolute_error(y_test["SBP"], y_pred[:, 0])
mae_dbp = metrics.mean_absolute_error(y_test["DBP"], y_pred[:, 1])

print(len(bp_df), ' beats in total.')
print(f"Root mean squared error for SBP: {rmse_sbp:.3f}")
print(f"Root mean squared error for DBP: {rmse_dbp:.3f}")
print(f"Mean absolute error for SBP: {mae_sbp:.3f}")
print(f"Mean absolute error for DBP: {mae_dbp:.3f}")


#### gain BP result df

estimate_bp_df = pd.DataFrame()
estimate_bp_df['SBP_actu'] = y_test['SBP']
estimate_bp_df['SBP_pred'] = y_pred[:, 0]
estimate_bp_df['DBP_actu'] = y_test['DBP']
estimate_bp_df['DBP_pred'] = y_pred[:, 1]
estimate_bp_df.reset_index(drop=True)

#### Draw pictures

plt.figure(figsize=(30, 15))

plt.subplot(2, 1, 1)
x1 = np.array((range(len(estimate_bp_df))))
sbp1 = np.array(estimate_bp_df['SBP_pred'])
sbp2 = np.array(estimate_bp_df['SBP_actu'])
plt.plot(x1, sbp1, label='SBP_pred')
plt.plot(x1, sbp2, label='SBP_actu')
plt.title('SBP')

plt.subplot(2, 1, 2)
x2 = np.array((range(len(estimate_bp_df))))
dbp1 = np.array(estimate_bp_df['DBP_pred'])
dbp2 = np.array(estimate_bp_df['DBP_actu'])
plt.plot(x1, dbp1, label='DBP_pred')
plt.plot(x1, dbp2, label='DBP_actu')
plt.title('DBP')

plt.legend()
plt.show()


################################################################################################################################################################################


