import os
import numpy as np
import pandas as pd

dir_data='C:/Users/User/Documents/'
f_app=os.path.join(dir_data,'application_train.csv')
print('Path fo  read data: %s' %(f_app))
app_train=pd.read_csv(f_app)

print(app_train.shape)
print(app_train.columns)
print(app_train.iloc[:5,0:10])

