import numpy as np
import pymysql
import statistics
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
#Classification for multi-class case. Because of need to calculate precision, recall, and f1_score, made evaluation function to get each seperately
def classification_performance_eval(y, y_predict):
    accuracy=[]
    precision=[]
    recall=[]
    f1_score=[]
    tp, fp= 0,0
    a_tp, b_tp, c_tp = 0,0,0
    a_fp,b_fp,c_fp=0,0,0
    a_fn,b_fn,c_fn=0,0,0
    # true positive is saved as tp, and false positive and negative is saved as fp, fn.
    for y, yp in zip(y,y_predict): 
        if y == 'A' and yp == 'A':
            a_tp += 1
        elif y == 'B' and yp == 'B':
            b_tp += 1
        elif y == 'C' and yp == 'C' :
            c_tp += 1
        else :
            a,b,c=fp_eval(y,yp)
            a_fp+=a
            b_fp+=b
            c_fp+=c
            a,b,c=fn_eval(y,yp)
            a_fn+=a
            b_fn+=b
            c_fn+=c
    tp=a_tp+b_tp+c_tp
    fp=a_fp+b_fp+c_fp
    accuracy = (tp)/(tp+fp)
    #precision, recall, f1_score values are saved as tuples for each class
    precision.append(a_tp/(a_tp+a_fp)) 
    precision.append(b_tp/(b_tp+b_fp))
    precision.append(c_tp/(c_tp+c_fp))
    recall.append(a_tp/(a_tp+a_fn))
    recall.append(b_tp/(b_tp+b_fn))
    recall.append(c_tp/(c_tp+c_fn))
    for p,r in zip(precision,recall):
        f1_score.append(2*p*r/(p+r))
    return accuracy,precision,recall,f1_score
#function for fp evalution
def fp_eval(y,yp):
    a_fp,b_fp,c_fp=0,0,0
    if y=='A':
        if yp != 'A':
             a_fp = 1
    elif y=='B':
        if yp != 'B':
             b_fp = 1
    elif y=='C':
        if yp != 'C':
             c_fp = 1
    return a_fp,b_fp,c_fp
#function for fn evalution
def fn_eval(y,yp):
    a_fn,b_fn,c_fn=0,0,0
    if yp=='A':
        if y != 'A':
             a_fn = 1
    elif yp=='B':
        if y != 'B':
             b_fn = 1
    elif yp=='C':
        if y != 'C':
             c_fn = 1
    
    return a_fn,b_fn,c_fn
#connection for mysql for the table
conn=pymysql.connect(host='localhost',user='root',password='deoxys0922-',db='scores_2')
curs=conn.cursor(pymysql.cursors.DictCursor);
sql = "select * from scores_2"
curs.execute(sql)
data = curs.fetchall()
curs.close()
conn.close()
X = [ (t['homework'],t['discussion'],t['final']) for t in data]
y = [t['grade'] for t in data]
X=np.array(X)
y = np.array(y)
#used train_test_split to split data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.05,random_state=42)
#Standarized datas
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#SVM model with data fitting, class_weight has been set as balanced because without it, y_pred seems to create one-sided result.
model = svm.SVC(kernel='rbf',class_weight = 'balanced').fit(X_train, y_train)
y_pred=model.predict(X_test)
acc,prec,rec,f1=classification_performance_eval(y_test,y_pred)
accuracy=[]
precision=[]
recall=[]
f1_score=[]
print("==============================")
print("SVM  multi class train_test_split result")
print("==============================")
print("accuracy  | %0.2f"%acc)
print("             A       B       C")
print("precision | %0.2f   %0.2f    %0.2f"%(prec[0],prec[1],prec[2]))
print("recall    | %0.2f   %0.2f    %0.2f"%(rec[0],rec[1],rec[2]))
print("f1_score  | %0.2f   %0.2f    %0.2f"%(f1[0],f1[1],f1[2]))
print("==============================")

accuracy=[]
precision=[]
recall=[]
f1_score=[]
#used StratifiedKFold to split data since just using KFold creates one-sided data
kf=StratifiedKFold(n_splits=4,random_state=42,shuffle=True)
for train_index, test_index in kf.split(X,y):
    X_train, X_test=X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = svm.SVC(kernel='rbf').fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc,prec,rec,f1=classification_performance_eval(y_test,y_pred)
    accuracy.append(acc)
    precision.append(prec)
    recall.append(rec)
    f1_score.append(f1)

accuracy=np.array(accuracy)
precision=np.array(precision)
recall=np.array(recall)
f1_score=np.array(f1_score)
pm=precision.mean(axis=0)
rm=recall.mean(axis=0)
fs=f1_score.mean(axis=0)

print("==============================")
print("SVM  multi class KFold result")
print("==============================")
print("accuracy  | %0.2f"%accuracy.mean())
print("             A       B       C")
print("precision | %0.2f   %0.2f    %0.2f"%(pm[0],pm[1],pm[2]))
print("recall    | %0.2f   %0.2f    %0.2f"%(rm[0],rm[1],rm[2]))
print("f1_score  | %0.2f   %0.2f    %0.2f"%(fs[0],fs[1],fs[2]))
print("==============================")
