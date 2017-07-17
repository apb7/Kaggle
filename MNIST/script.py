import pandas as pd
import numpy as np

df_train=pd.read_csv("train.csv")
labels_train=df_train.as_matrix(["label"]).ravel()
df_train=df_train.drop(["label"],axis=1)
features_train=df_train.as_matrix()
df_test=pd.read_csv("test.csv")
features_test=df_test.as_matrix()

features_train=features_train.astype(np.float)
features_test=features_test.astype(np.float)
#labels_train=labels_train.astype(np.float)

print(features_train)
print(labels_train)
print(features_test)

from sklearn.preprocessing import MinMaxScaler as mms
scalar=mms()
features_train=scalar.fit_transform(features_train)
features_test=scalar.fit_transform(features_test)
#labels_train=scalar.fit_transform(labels_train)


print(features_train)
print(labels_train)
print(features_test)

def support_vector_classifier(f_train,l_train,f_test):
    from sklearn.grid_search import GridSearchCV as gscv
    from sklearn.svm import  SVC
    import time
    param={'kernel':('linear','rbf'),'C':[1,2,5,10,15,20]}
    svr=SVC()
    clf=gscv(svr,param_grid=param)
    #clf=SVC(kernel='linear')
    start_time=time.time()
    clf.fit(f_train,l_train)
    print("Training Time: %s seconds"%(time.time()-start_time))
    start_time=time.time()
    pred=clf.predict(f_test)
    print("Predicting Time: %s seconds"%(time.time()-start_time))
    print(clf.best_params_)

    return pred


def knn(f_train,l_train,f_test):
    from sklearn.neighbors import KNeighborsClassifier as knc
    import time
    clf=knc(n_neighbors=3)
    start_time=time.time()
    clf.fit(f_train,l_train)
    print("Training Time: %s seconds"%(time.time()-start_time))
    start_time=time.time()
    pre=clf.predict(f_test)
    print("Predicting Time: %s seconds"%(time.time()-start_time))
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

pred=support_vector_classifier(features_train,labels_train,features_test)
#pred=knn(features_train,labels_train,features_test)
writer(pred)
