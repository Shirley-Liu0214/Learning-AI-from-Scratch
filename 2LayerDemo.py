import numpy as np
#sigmoid function
def nonlin(x,deriv=False):
    if deriv:  #generate the derivative of a sigmoid
        return x*(1-x)
    return 1/(1+np.exp(-x))
X=np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y=np.array([[0,0,1,1]]).T
np.random.seed(1)#ensure numbers being randomly distributed in exactly the same way each time we train
# have a mean of zero in weight initialization
syn0=2*np.random.random((3,1))-1 #weight matrix (3,1)
for i in range(10000):
    #forward propagation
    l0=X
    #同时处理整个训练集，有4个不同的l0行
    l1=nonlin(np.dot(l0,syn0))
    #The final matrix generated is thus the number of rows of the first matrix and the number of columns of the second matrix.
    #对于本例：一列四行，和y相同
    l1_error=y-l1
    l1_delta=l1_error*nonlin(l1,True)
    #两个4x1的矩阵按元素乘
    syn0+=np.dot(l0.T,l1_delta) #update weights
print("训练后的输出：")
print(l1)
print("学习到的权重：")
print(syn0)

