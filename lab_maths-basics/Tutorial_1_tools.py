#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from   scipy.stats import multivariate_normal
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


# In[ ]:


#n = 100
#mu_0_true = np.array([120.0,50.0])
#mu_1_true = np.array([220.0,125.0])
#cov_0_true = np.array([[400,100],[100,400]])
#cov_1_true = np.array([[1500,100],[100,1500]])

#X_0_test = np.random.multivariate_normal(mu_0_true,cov_0_true,int(n/4))
#X_1_test = np.random.multivariate_normal(mu_1_true,cov_1_true,n)


# In[ ]:


#def unison_shuffled_copies(a, b):
#    assert len(a) == len(b)
#    p = np.random.permutation(len(a))
#    return a[p], b[p]


# In[ ]:


#X_test = np.concatenate((X_0_test,X_1_test),0)
#Y_test = np.concatenate((np.zeros(len(X_0_test),),np.ones(len(X_1_test),)),0)

#X_test_shuffled,Y_test_shuffled = unison_shuffled_copies(X_test, Y_test)


# In[ ]:


#np.save('lab_test_features.npy', X_test_shuffled)
#np.save('lab_test_labels.npy', Y_test_shuffled)


# In[ ]:


def plot_features_and_label(X_0,X_1):

    plt.plot(X_0[:,0],X_0[:,1],'ro',label = 'monocyte')
    plt.plot(X_1[:,0],X_1[:,1],'bx',label = 'granulocyte')
    plt.legend()
    plt.ylabel('Granularity - SSC [nm]')
    plt.xlabel('Size - FFC [nm]')
    plt.show()


# In[ ]:


def generate_density_plots(mu_0=np.zeros(2,),cov_0=np.eye(2),mu_1=np.zeros(2,),cov_1=np.eye(2)):
    N    = 200
    X    = np.linspace(0, 350, N)
    Y    = np.linspace(0, 250, N)
    X, Y = np.meshgrid(X, Y)
    pos  = np.dstack((X, Y))
    rv_0   = multivariate_normal(mu_0, cov_0)
    Z_0    = rv_0.pdf(pos)

    rv_1   = multivariate_normal(mu_1, cov_1)
    Z_1    = rv_1.pdf(pos)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.contour(X, Y, Z_0)
    mycmap2 = plt.get_cmap('gnuplot2')
    ax2.contour(X, Y, Z_1, cmap=mycmap2)
    ax1.set(xlabel='Size - FFC [nm]',ylabel='Granularity - SSC [nm]')
    ax2.set(xlabel='Size - FFC [nm]',ylabel='Granularity - SSC [nm]')
    ax1.set_title('PDF of the monocytes')
    ax2.set_title('PDF of the granulocytes')

    plt.show()
    
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1,2,1,projection='3d')
    ax2 = fig.add_subplot(1,2,2, projection='3d')

    ax1.plot_surface(X, Y, Z_0, 
                  cmap=cm.coolwarm,
                  linewidth=0, 
                  antialiased=False)
    ax2.plot_surface(X, Y, Z_1, 
                  cmap=cm.coolwarm,
                  linewidth=0, 
                  antialiased=False)
    ax1.set(xlabel='Size - FFC [nm]',ylabel='Granularity - SSC [nm]',zlabel='density')
    ax2.set(xlabel='Size - FFC [nm]',ylabel='Granularity - SSC [nm]',zlabel='density')
    ax1.set_title('PDF of the monocytes')
    ax2.set_title('PDF of the granulocytes')
    ax1.set_zlim3d(0, 0.0004)
    ax2.set_zlim3d(0, 0.0004)
    ax1.set_zticks([])  
    ax2.set_zticks([])  

    plt.show()


# In[ ]:


def generate_decision_region_plots(classifier,mu_0,cov_0,mu_1,cov_1):
    N    = 200
    X    = np.linspace(0, 350, N)
    Y    = np.linspace(0, 250, N)
    X, Y = np.meshgrid(X, Y)
    pos  = np.dstack((X, Y))
    
    Z    =  classifier(mu_0,cov_0,mu_1,cov_1,pos)

    plt.contourf(X, Y, Z, cmap=plt.get_cmap('Pastel1'))
    plt.xlabel('Size - FFC [nm]')
    plt.ylabel('Granularity - SSC [nm]')
    

    plt.show()


# In[ ]:


def score_classifier(Y_predict,Y_test):
    return np.mean(Y_predict == Y_test)


# In[ ]:


def generate_test_plots(X_test,Y_test,classifier,mu_0,cov_0,mu_1,cov_1):
    N    = 200
    X    = np.linspace(0, 350, N)
    Y    = np.linspace(0, 250, N)
    X, Y = np.meshgrid(X, Y)
    pos  = np.dstack((X, Y))
    
    Z    =  classifier(mu_0,cov_0,mu_1,cov_1,pos)
    
    X_0_test = X_test[(Y_test == 0 ),:]
    X_1_test = X_test[(Y_test == 1 ),:]
    plt.plot(X_0_test[:,0],X_0_test[:,1],'ro',label = 'monocyte')
    plt.plot(X_1_test[:,0],X_1_test[:,1],'bx',label = 'granulocyte')

    plt.contourf(X, Y, Z, cmap=plt.get_cmap('Pastel1'))
    plt.xlabel('Size - FFC [nm]')
    plt.ylabel('Granularity - SSC [nm]')
    

    plt.show()
    
def generate_monty_plot(monty_python_simulation,swap):
    success = []

    for i in range(10000):
        car_is_behind_door = np.random.randint(1,3+1)
        door_chosen = monty_python_simulation(car_is_behind_door,swap)
        success.append(door_chosen==car_is_behind_door)

    plt.plot(np.arange(1,len(success)+1),np.cumsum(success)/np.arange(1,len(success)+1))
    plt.xlabel('iterations')
    plt.ylabel('average success')
    if swap==True: 
        plt.title('Monty Python Hall problem - changing strategy')
    else : 
        plt.title('Monty Python Hall problem - keeping strategy')
    print('Average success is {:.2f} %'.format(np.mean(success)*100))

