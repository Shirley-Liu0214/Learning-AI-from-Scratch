# there is a one-to-one relationship between a combination of inputs->add an another layer
import numpy as np
def nonlin(x,deriv=False):
    if (deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
x=np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y=np.array([[0],[1],[1],[0]])
np.random.seed(1)
syn0=2*np.random.random((3,4))-1
syn1=2*np.random.random((4,1))-1
for j in range(10000):
    I0=x
    I1=nonlin(np.dot(I0,syn0))
    I2=nonlin(np.dot(I1,syn1))
    I2_error=y-I2
    I2_delta=I2_error*nonlin(I2,deriv=True)
    I1_error=I2_delta.dot(syn1.T)
    I1_delta=I1_error*nonlin(I1,deriv=True)
    syn0+=I0.T.dot(I1_delta)
    syn1+=I1.T.dot(I2_delta)
print(syn0)
print(syn1)
IO=nonlin(x.dot(syn0))
IO=nonlin(IO.dot(syn1))

print(IO)

