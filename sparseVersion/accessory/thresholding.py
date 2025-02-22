import scipy.io 
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

# from params import *

dirr = '/Users/sima/Documents/MATLAB/oberlaender'
dictt = scipy.io.loadmat('%s/MatStandardL4ssJuly.mat'%dirr)

keys = list(dictt.keys())[3:]

# for ik, k in enumerate(keys):
#     print('')
#     print(k)
#     print(np.shape(dictt[k]))
    

amtx = dictt['SqA']
# cmtx = dictt['SqC']


x = amtx.flatten()
print(len(np.where(x)[0])/len(x))

z = x[np.where(x)[0]]

# plt.figure()
# plt.hist(z, 100)
# plt.show()



PE = np.array([0.18368079, 0.28728917])# Prob E-to-E, Prob I-to-E
PI = np.array([0.25259917, 0.21585833])# Prob I-to-I, Prob E-to-I
NI, NE, Next = [ 680, 2931,  311]
N = NI + NE

j1 = 1
h1=1.5
g1 = 3
gAll=1.25


K = NE*PE[0]

cm = 250 #[pA]
taum=10 #[ms]

tauref=1
gl=cm/taum #[nS]c
thresh=15
IC = thresh*gl # [pA] Threshold Current


ww = 1/np.sqrt(K);
jj = np.array([[-g1, h1], [-j1 * g1, 1]])
J = gAll*IC*(jj)*ww*taum;




rndmtx = np.random.rand(N, N)


diffrand = 0.15
# diffrand = 0
# diffrand = -0.1
W = diffrand+rndmtx < amtx
W = W.astype(int)

print(np.sum(W)/N/(N-1))



# Initialize the matrix
JMat = np.zeros((N,N))

globalind = [np.arange(NI), np.arange(NI, N)]
# Loop over tpost and tpre
for tpost in range(2):
    for tpre in range(2):
        JMat[np.ix_(globalind[tpost],globalind[tpre])] = J[tpost, tpre] * W[np.ix_(globalind[tpost],globalind[tpre])]
 
 
 
        
from matplotlib.colors import ListedColormap, BoundaryNorm
cmap = ListedColormap(['#0000ff', '#ffffff', '#ff4d4d', '#cc0000'])  
boundaries = [-610, -10, 10, 250, 350]  # For example: data values in the range [0, 1), [1, 2), ...
norm = BoundaryNorm(boundaries, cmap.N)  

plt.close('all')
fig = plt.figure()
az = fig.add_subplot(111)
cav = az.matshow(JMat, cmap = cmap, norm=norm)
cbar = plt.colorbar(cav, ticks = [-310.,    0.,  130.,  300.])
cbar.set_ticklabels([-606, 0, 202, 303])
plt.show()  

ab = JMat!=0
np.sum(ab,0)
receivingEnd = np.sum(ab,1)
providingEnd = np.sum(ab,0)

plt.figure(figsize=(10,4))
plt.subplot(121)
plt.hist(receivingEnd[NI:],100, color='r')
plt.hist(receivingEnd[:NI],100, color='b')
plt.title('receiving degree (dendritic end)')
# plt.legend(['exc', 'inh'])

plt.subplot(122)
plt.hist(providingEnd[NI:],100, color='r')
plt.hist(providingEnd[:NI],100, color='b')
plt.title('providing degree (axonal end)')
plt.legend(['exc', 'inh'])
plt.show()
