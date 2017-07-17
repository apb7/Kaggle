import pandas
import matplotlib.pyplot as plt
import numpy as np
df_train=pandas.read_csv("train.csv")
labels_train=df_train.as_matrix(["SalePrice"]).ravel()
df_train=df_train.drop(["Id","SalePrice"],axis=1)
df_train=pandas.get_dummies(df_train,columns=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
       'PavedDrive', 'SaleType', 'SaleCondition'])
#print(df_train.select_dtypes(include=["object"]).columns)

#####################################
df_test=pandas.read_csv("test.csv")
df_test=df_test.drop(["Id"],axis=1)
df_test=pandas.get_dummies(df_test,columns=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
       'PavedDrive', 'SaleType', 'SaleCondition'])
df_train, df_test = df_train.align(df_test, join='inner', axis=1) # maybe 'outer' is better
#df_test.to_csv(path_or_buf="df_test.csv")
#df_train.to_csv(path_or_buf="df_train.csv")
features_test=df_test.as_matrix()
features_train=df_train.as_matrix()
#########################################
#print(df_train.isnull().sum().sum())
#print(df_train.describe())
#print(df_test.describe())
#print(labels_train)
#print(features_test)
#print(features_train)
def lrc(f_train,l_train,f_test):
    from sklearn.linear_model import LinearRegression as lr
    reg=lr()
    reg.fit(f_train,l_train)
    return reg.predict(f_test)

def lasso(f_train,l_train,f_test):
    from sklearn.linear_model import LassoCV
    clf=LassoCV(alphas=[1, 0.1, 0.001, 0.0005])
    clf.fit(f_train,l_train)
    return clf.predict(f_test)

def writer(pred):
    txt = []
    j = 1461
    for i in range(0,len(pred)):
        txt.append([str(j), str(pred[i])])
        j += 1
    print(txt)
    df_result = pandas.DataFrame(txt)
    df_result.columns = ["Id", "SalePrice"]
    df_result.to_csv(path_or_buf="output.csv", index=False)


pred=lrc(features_train,labels_train,features_test)
#print(len(pred))
#print(pred)
writer(pred)
