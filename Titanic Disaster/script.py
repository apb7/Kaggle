# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
ele=pd.read_csv('train.csv',as_recarray=True)
elet=pd.read_csv('test.csv',as_recarray=True)
#print(ele)
labels_train=[]
features_train=[]
features_test=[]
pclass=[]
pclass_test=[]
sex=[]
sex_test=[]
age=[]
sibsp=[]
sibsp_test=[]
parch=[]
parch_test=[]
fare=[]
age_test=[]
fare_test=[]
emb=[]
emb_test=[]
def reader():
    global age,age_test,labels_train,features_train,pclass,features_test,pclass_test,sex,sex_test,fare,fare_test,sibsp,sibsp_test,parch,parch_test,emb,emb_test
    for i in range(0,891):
        for j in range(0,12):
            if j==1:
                labels_train.append(ele[i][j])
            elif j==2:
                 pclass.append(ele[i][j])
            elif j==4:
                if ele[i][j]=='male':
                    sex.append(0)
                else:
                    sex.append(1)
            elif j==5:
                if math.isnan(ele[i][j]):
                    age.append(29.7)
                else:
                    age.append(ele[i][j])
            elif j==9:
                    fare.append(ele[i][j])
            elif j==6:
                    sibsp.append(ele[i][j])
            elif j==7:
                    parch.append(ele[i][j])
            elif j==11:
                if ele[i][j]=='S':
                    emb.append(0)
                elif ele[i][j]=='Q':
                    emb.append(1)
                else:
                    emb.append(2)
    for i in range(0,891):
        features_train.append([pclass[i],sex[i],age[i],sibsp[i],parch[i],fare[i],emb[i]])


    for i in range(0,418):
        for j in range(0,11):
            if j==1:
                pclass_test.append(elet[i][j])
            elif j==3:
                if elet[i][j]=='male':
                    sex_test.append(0)
                else:
                    sex_test.append(1)
            elif j==4:
                if math.isnan(elet[i][j]):
                    age_test.append(30.27)
                else:
                    age_test.append(elet[i][j])
            elif j==5:
                sibsp_test.append(elet[i][j])
            elif j==6:
                parch_test.append(elet[i][j])
            elif j==8:
                fare_test.append(elet[i][j])
            elif j==10:
                if ele[i][j]=='S':
                    emb_test.append(0)
                elif ele[i][j]=='Q':
                    emb_test.append(1)
                else:
                    emb_test.append(2)

    #print(age_test)
    for i in range(0,418):
        features_test.append([pclass_test[i],sex_test[i],age_test[i],sibsp_test[i],parch_test[i],fare_test[i],emb_test[i]])

    labels_train=np.asarray(labels_train)
    features_train=np.asarray(features_train)
    features_test=np.asarray(features_test)


def random_forest():
    global features_train,labels_train,features_test
    from sklearn.ensemble import RandomForestClassifier
    #from sklearn.tree import export_graphviz
    from sklearn.grid_search import GridSearchCV
    param={'criterion' : ('gini','entropy'),'min_samples_split':[2,5,10,15,20,25,30],'n_estimators':[10,50,100]}
    svr=RandomForestClassifier()
    clf=GridSearchCV(svr,param)
    clf.fit(features_train,labels_train)
    #print(clf.best_params_)
    pred=clf.predict(features_test)
    #for tree in clf.estimators_ :
    #    with open("tree.dot",'r+') as my_file:
    #        my_file=export_graphviz(tree,out_file=my_file)
    return pred

def xg_boost():
    global features_train, labels_train, features_test
    from xgboost import XGBClassifier as xgb
    clf=xgb()
    clf.fit(features_train,labels_train)
    pred=clf.predict(features_test)
    return pred




def gbm():
    global features_train, labels_train, features_test
    from sklearn.ensemble import GradientBoostingClassifier as gbc
    from sklearn.grid_search import GridSearchCV as gscv
    param={'learning_rate':[0.1,0.01,0.3,0.4,0.5,0.2],"n_estimators":[10,50,100],'min_samples_split':[2,5,10,15,20,25,30]}
    svr=gbc()
    clf=gscv(svr,param)
    clf.fit(features_train,labels_train)
    pred=clf.predict(features_test)
    return pred

def writer():
    global pred
    txt=[]
    j=892
    for i in pred:
        txt.append([str(j),str(i)])
        j+=1
    return txt




reader()
#print(features_train[0])
#print(features_test[0])
pred=random_forest()
txt=writer()
#print(txt)
df=pd.DataFrame(txt)
df.columns=["PassengerId","Survived"]
#print(df)
df.to_csv(path_or_buf="output.csv",index=False)

