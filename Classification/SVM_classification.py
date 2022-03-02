import numpy as np
import pymysql
import statistics
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#binary classification evaluation
def classification_performance_eval(y, y_predict):
    tp, tn, fp, fn = 0,0,0,0
    for y, yp in zip(y,y_predict):
        if y == 1 and yp == 1:
            tp += 1
        elif y == 1 and yp == -1:
            fn += 1
        elif y == -1 and yp == 1:
            fp += 1
        else:
            tn += 1
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    f1_score = 2*precision*recall / (precision+recall)
    
    return accuracy, precision, recall, f1_score
conn=pymysql.connect(host='localhost',user='root',password='deoxys0922-',db='scores_2')
curs=conn.cursor(pymysql.cursors.DictCursor);
sql = "select * from scores_2"
curs.execute(sql)
data = curs.fetchall()
curs.close()
conn.close()
X = [ (t['homework'],t['discussion'],t['final']) for t in data]
y = [1 if (t['grade']=='B') else -1 for t in data]
X=np.array(X)
y = np.array(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.05,random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#class weight was added to SVC function because the data gets very one-sided after split
model = svm.SVC(kernel='rbf',class_weight = 'balanced').fit(X_train, y_train) 
y_predict=model.predict(X_test)
acc,prec,rec,f1=classification_performance_eval(y_test,y_predict)
print("==============================")
print("SVM train_test_split result")
print("==============================")
print("accuracy=%0.2f" %acc)
print("precision=%0.2f" %prec)
print("recall=%0.2f" %rec)
print("f1_score=%0.2f\n" %f1)
print("==============================")

accuracy=[]
precision=[]
recall=[]
f1_score=[]
#Used StratifiedKFold just in case the data dont get one-sided
kf=StratifiedKFold(n_splits=4,random_state=42,shuffle=True)
for train_index, test_index in kf.split(X,y):
    X_train, X_test=X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = svm.SVC(kernel='rbf',class_weight = 'balanced').fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc, prec, rec, f1 = classification_performance_eval(y_test, y_pred)
    accuracy.append(acc)
    precision.append(prec)
    recall.append(rec)
    f1_score.append(f1)
print("==============================")
print("SVM KFold result")
print("==============================")
print("accuracy=%0.2f" %statistics.mean(accuracy))
print("precision=%0.2f" %statistics.mean(precision))
print("recall=%0.2f" %statistics.mean(recall))
print("f1_score=%0.2f" %statistics.mean(f1_score))
print("==============================")
