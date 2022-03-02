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
    X = [ ( t['final'] ) for t in data ]
    X = np.array(X)
    
    y = [ (t['score']) for t in data]
    y = np.array(y)

    return X, y

X, y = load_dbscore_data()

'''
plt.scatter(X, y) 
plt.show()
'''

# y = mx + c
m_data=[]
c_data=[]
#initiate plot

def gradient_descent_naive(X, y):

    epochs = 100000
    min_grad = 0.0001
    learning_rate = 0.001
    
    m = 0.0
    c = 0.0
    m_data.append(m)
    c_data.append(c) # appending first data of m, c
    n = len(y)
    
    c_grad = 0.0
    m_grad = 0.0
    

    for epoch in range(epochs):
        
        for i in range(n):
            
            y_pred = m * X[i] + c
            m_grad += 2*(y_pred-y[i]) * X[i]
            c_grad += 2*(y_pred - y[i])

        c_grad /= n
        m_grad /= n
        
        m = m - learning_rate * m_grad
        c = c - learning_rate * c_grad
        
        
        if ( epoch % 1000 == 0):
            #print("epoch %d: m_grad=%f, c_grad=%f, m=%f, c=%f" %(epoch, m_grad, c_grad, m, c) )
            m_data.append(m)   
            c_data.append(c)
        if ( abs(m_grad) < min_grad and abs(c_grad) < min_grad ):
            break
        
    return m, c
fig, ax=plt.subplots()
ax.scatter(X,y)
line, =ax.plot([],[],'r')
label=ax.text(0,20,str(0)+'  '+str(0))
def animate(frame_num):
    x=np.arange(0,100,1)
    y=m_data[frame_num]*x+c_data[frame_num]
    label.set_text('m: '+str(m_data[frame_num])+'  '+'c: '+str(c_data[frame_num]))
    line.set_data(x,y)
start_time = time.time()
m, c = gradient_descent_naive(X, y)
end_time = time.time()
ani=FuncAnimation(fig,animate,frames=len(m_data),interval=1)
plt.show()
#ani.save('result_naive.gif',writer='imagemagick',fps=30)
#print("%f seconds" %(end_time - start_time))

#print("\n\nFinal:")
#print("gdn_m=%f, gdn_c=%f" %(m, c) )
#print("ls_m=%f, ls_c=%f" %(ls_m, ls_c) )




#print("\n\nFinal:")
#print("gdv_m=%f, gdv_c=%f" %(m, c) )
#print("ls_m=%f, ls_c=%f" %(ls_m, ls_c) )



