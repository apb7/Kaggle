import pandas as pd
import numpy as np

df_train=pd.read_csv("train.csv")
labels_train=df_train.as_matrix(["label"]).ravel()
df_train=df_train.drop(["label"],axis=1)
features_train=df_train.as_matrix()
df_test=pd.read_csv("test.csv")
features_test=df_test.as_matrix()
#print(features_train)
#print(type(labels_train))
#print(features_test)

def support_vector_classifier(f_train,l_train,f_test):
    from sklearn.grid_search import GridSearchCV as gscv
    from sklearn.svm import  SVC
    #param={'kernel':('linear','rbf'),'C':[1,2,5,10]}
    #svr=SVC()
    #clf=gscv(svr,param_grid=param)
    print("#############################1")
    clf=SVC(kernel='linear')
    clf.fit(f_train,l_train)
    print("#############################2")
    pred=clf.predict(f_test)
    #print(clf.best_params_)
    print("#############################3")
    return pred


def knn(f_train,l_train,f_test):
    from sklearn.neighbors import KNeighborsClassifier as knc
    clf=knc(n_neighbors=3)
    print("Starting Fitting!")
    clf.fit(f_train,l_train)
    print("Fitting Complete!")
    print("Pred Started!")
    pre=clf.predict(f_test)
    print("Pred Ended!")
    return pre

def writer(pred):
    txt=[]
    j=1
    for i in range(0,len(pred)):
        txt.append([str(j),str(pred[i])])
        print(str(j))
        j+=1
    print(txt)
    df_result = pd.DataFrame(txt)
    df_result.columns = ["ImageId", "Label"]
    df_result.to_csv(path_or_buf="output.csv", index=False)

#pred=support_vector_classifier(features_train,labels_train,features_test)
pred=knn(features_train,labels_train,features_test)
writer(pred)
