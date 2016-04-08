import numpy as np

C1=np.zeros([5,5])
C1[0:3,0:3]=1
C1=C1.reshape([25,1])

C2=np.zeros([5,5])
C2[1:4,1:4]=1
C2=C2.reshape([25,1])

Y=np.zeros([5*5,500])
for i in range(0,500):
    Y[:,i:i+1] = C1*np.abs(np.sin(np.pi*(i-1)/50))*1000 + C2*np.abs(np.sin(np.pi*(i-1)/19))*1000 + 25*np.array(np.random.randn(25,1));

np.save('testdata_small.npy',Y.T)
