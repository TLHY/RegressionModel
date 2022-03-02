import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymysql
import time
from matplotlib.animation import FuncAnimation
def load_dbscore_data():
    conn = pymysql.connect(host='localhost',user='root',password='deoxys0922-',db='linear_score')
    curs = conn.cursor(pymysql.cursors.DictCursor)
    
    sql = "select * from linear_scores"
    curs.execute(sql)
    
    data  = curs.fetchall()
    
    curs.close()
    conn.close()
    
    #X = [ (t['attendance'], t['homework'], t['midterm'] ) for t in data ]
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
what_data=[]
def gradient_descent_naive(X, y):
    
    epochs = 100000
    min_grad = 0.0001
    learning_rate = 0.001
    m = [0.0,0.0,0.0]
    c = 0.0
    n = len(y)
    
    c_grad = 0.0
    m_grad = [0.0,0.0,0.0]
    y_pred=0.0
    for epoch in range(epochs):
        for i in range(n):
            y_pred=m[0]*X[i][0]+m[1]*X[i][1]+m[2]*X[i][2]+c
            for add_grad in range(3):
                m_grad[add_grad] += 2*(y_pred-y[i]) * X[i][add_grad]
            c_grad += 2*(y_pred - y[i])
        c_grad /= n
        for num in range(3):
            m_grad[num]/=n
            m[num]=(m[num] - learning_rate * m_grad[num])
        c = c - learning_rate * c_grad
        if (epoch % 1000 == 0):
            m_data.append(m[:])
            c_data.append(c)
        if ( ((abs(m_grad[0]) < min_grad) and (abs(m_grad[1]) < min_grad) and (abs(m_grad[2]) < min_grad)) and abs(c_grad) < min_grad ):
            break
          
    return m, c, m_data,c_data
start_time = time.time()
lists, c, m_data, c_data=gradient_descent_naive(X, y)
end_time = time.time()
print("==============================")
print("Multi-Linear Naive")
print("==============================")
print("const=%0.4f" %c)
print("x1=%0.4f" %lists[0])
print("x2=%0.4f" %lists[1])
print("x3=%0.4f" %lists[2])
print("Time: %0.4f" %abs(start_time-end_time))
print("==============================")
fig, ax=plt.subplots()
ax.scatter(X[:,0],y)
ax.scatter(X[:,1],y)
ax.scatter(X[:,2],y)
line1, =ax.plot([],[],color='r')
line2, =ax.plot([],[],color='b')
line3, =ax.plot([],[],color='g')
label=ax.text(0,20,str(0)+'  '+str(0))

def animate(frame_num):
    k1=0
    k2=0
    k3=0
    x1=np.arange(0,100,1)
    x2=np.arange(0,100,1)
    x3=np.arange(0,100,1)
    k1=m_data[frame_num][0]*x1+c_data[frame_num]
    k2=m_data[frame_num][1]*x2+c_data[frame_num]
    k3=m_data[frame_num][2]*x3+c_data[frame_num]
    label.set_text('m1: '+str(m_data[frame_num][0])+'  '+'m2: '+str(m_data[frame_num][1])+'  '+'m3: '+str(m_data[frame_num][2])+'  '+'c: '+str(c_data[frame_num]))
    line1.set_data(x1,k1)
    line2.set_data(x2,k2)
    line3.set_data(x3,k3)
ani=FuncAnimation(fig,animate,frames=len(m_data),interval=1)
plt.show()
ani.save('result_multi_naive.gif',writer='imagemagick',fps=30)
