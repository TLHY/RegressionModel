import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymysql
import time
from sklearn import linear_model
from matplotlib.animation import FuncAnimation
def load_dbscore_data():
    conn = pymysql.connect(host='localhost',user='root',password='deoxys0922-',db='linear_score')
    curs = conn.cursor(pymysql.cursors.DictCursor)
    
    sql = "select * from linear_scores"
    curs.execute(sql)
    
    data  = curs.fetchall()
    
    curs.close()
    conn.close()
    
    X = [ ( t['attendance'],t['homework'],t['final']) for t in data ]
    X = np.array(X)
    
    y = [ (t['score']) for t in data]
    y = np.array(y)

    return X, y

X, y = load_dbscore_data()

import statsmodels.api as sm
X_const = sm.add_constant(X)
start_time_OLS = time.time()
model = sm.OLS(y, X_const)
ls = model.fit()
end_time_OLS = time.time()
m_data=[]
c_data=[]
print(ls.summary())
print("Time: %0.4f" %abs(start_time_OLS-end_time_OLS))

ls_c = ls.params[0]
ls_m=[ls.params[1],ls.params[2],ls.params[3]]
y_pred = ls_m*X + ls_c
plt.show()

def gradient_descent_vectored(X, y):
    epochs = 100000
    min_grad = 0.0001
    learning_rate = 0.001
     # appending first data of m, c
    n = len(y)
    X_flat=np.insert(X,0,1,axis=1)
    X_flat=X_flat.T
    y_new=y.reshape(92,1)
    m=np.zeros((4,1),dtype='float')
    for epoch in range(epochs):
        y_pred=X_flat.T@m
        m_grad = X_flat@(y_pred-y_new)/n
        m = (m - learning_rate * m_grad)
        if ( epoch % 1000 == 0):
            m_data.append(m)   
        if ( ((abs(m_grad[0]) < min_grad) and (abs(m_grad[1]) < min_grad) and (abs(m_grad[2]) < min_grad)) and abs(m_grad[3]) < min_grad ):
            break
          
    return m #c
start_time = time.time()
lists=gradient_descent_vectored(X, y)
end_time = time.time()
print("==============================")
print("Multi-Linear Vectorized")
print("==============================")
print("const=%0.4f" %lists[0])
print("x1=%0.4f" %lists[1])
print("x2=%0.4f" %lists[2])
print("x3=%0.4f" %lists[3])
print("Time: %0.4f" %abs(start_time-end_time))
print("==============================")
fig, ax=plt.subplots()
ax.scatter(X[:,0],y,color='r')
ax.scatter(X[:,1],y,color='b')
ax.scatter(X[:,2],y,color='g')
line1, =ax.plot([],[],'r')
line2, =ax.plot([],[],'b')
line3, =ax.plot([],[],'g')
label=ax.text(0,20,str(0)+'  '+str(0))
def animate(frame_num):
    k1=0
    k2=0
    k3=0
    x1=np.arange(0,100,1)
    x2=np.arange(0,100,1)
    x3=np.arange(0,100,1)
    k1=m_data[frame_num][1]*x1+m_data[frame_num][0]
    k2=m_data[frame_num][2]*x2+m_data[frame_num][0]
    k3=m_data[frame_num][3]*x3+m_data[frame_num][0]
    label.set_text('m1: '+str(m_data[frame_num][1])+'  '+'m2: '+str(m_data[frame_num][2])+'  '+'m3: '+str(m_data[frame_num][3])+'  '+'c: '+str(m_data[frame_num][0]))
    line1.set_data(x1,k1)
    line2.set_data(x2,k2)
    line3.set_data(x3,k3)
ani=FuncAnimation(fig,animate,frames=len(m_data),interval=1)
plt.show()
ani.save('result_multi_vector.gif',writer='imagemagick',fps=30)
