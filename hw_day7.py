import pandas as pd
import numpy as np

data_path='C:/Users/User/Documents/'
df_train=pd.read_csv(data_path +"titanic_train.csv")
df_test=pd.read_csv(data_path +"titanic_test.csv")
df_train.shape

# 整合訓練測試集資料
trian_y=df_train['Survived']
ids=df_test["PassengerId"]
df_train=df_train.drop(['PassengerId','Survived'],axis=1)
df_test=df_test.drop(['PassengerId'],axis=1)
df=pd.concat([df_train,df_test])
df.head()

# 秀資料欄位類型與數量
dtype_df=df.dtypes.reset_index()
dtype_df.columns=['Count','Column type']
dtype_df=dtype_df.groupby('Column type').aggregate('count').reset_index()
print(dtype_df)

# 分別將對應之名稱存int64,float64,object 的box中
int64_box=[]
float64_box=[]
object_box=[]
for dtype,feature in zip(df.dtypes,df.columns):
    if dtype=='float':
        float64_box.append(feature)
    elif dtype=='int64':
        int64_box.append(feature)
    else:
        object_box.append(feature)

print(f'{len(int64_box)} Integer: {int64_box}\n')
print(f'{len(float64_box)} float: {float64_box}\n')
print(f'{len(object_box)} object: {object_box}')

# 對欄位進行max,mean,unique 操作
def operation(data,group):
        print(f'Max value: \n{data[group].max()}')
        print(f'Mean value: \n{data[group].mean()}')
        print(f'Unique value: \n{data[group].unique()}')

print('=============inter deature================')
print(operation(df,int64_box))
print('=============float deature================')
print(operation(df,float64_box))
print('=============object deature================')
print(operation(df,object_box))

# Q1 試著執行作業程式，觀察三種類型的欄位分別進行( 平均 mean / 最大值 Max / 相異值 nunique )中的九次操作會有那些問題?並試著解釋那些發生Error的程式區塊的原因?
# 在int 資料型別下，只能進行 mean,max 運算，而無法進行unique運算 
# 在float資料型別下，無法進行 mean,max,unique 的運算
# 在object類型下，無法進行mean,max,unique 的運算 
# --------------------------------------------------------------------
# 取unique必須在資料轉為numpy陣列形式
# float的運算需藉由 np.finfo(np.float64)來執行 max,mean,unique
# 由於unique為返回數組中的所有唯一元素的方式，因此在float/object下無法進行數值上之運算。
# 而object因為為"物件"，不賦有數值意義，無法進行數值上的運算


# Q2 思考一下，試著舉出今天五種類型以外的一種或多種資料類型，你舉出的新類型是否可以歸在三大類中的某些大類? 所以三大類特徵中，哪一大類處理起來應該最複雜?
# str字串類型，為獨立之新類別。我認為時間型類別處理起來最為複雜，由於
# 必須先將和時間有關之單位以數字、字母作為區分、編排，因此轉換上相對複雜