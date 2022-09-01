import os
import numpy as np
import pandas as pd

data_path='./ML_100days/'
train_path=os.path.join(data_path,'application_train.csv')
app_train=pd.read_csv(train_path)
sub_train=pd.DataFrame(app_train['WEEKDAY_APPR_PROCESS_START'])
print(sub_train.shape)
print(sub_train.head())

# one-hot encoding (dummies)
sub_train_encoding=pd.get_dummies(sub_train)
print(sub_train_encoding.shape)
print(sub_train_encoding.head())