import numpy as np
import os
from os import listdir
import cv2
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random

#%% Path
dic = {'train':['./train_0', './train_1'], 'valid':['./valid_0']}

#%% Func
def _txt(dic):
    P = dic['train']
    F = []
    Emp = []
    Fn = 0
    En = 0
    for p in P:
        fL = listdir(p)
        for nf in fL:
            if nf.split('.')[-1]=='txt':
                f = np.loadtxt(os.path.join(p, nf))
                if len(f)!=0:
                    if f.ndim==1:
                        f = f[np.newaxis, :]
                    F.append(f)
                    Fn = Fn + 1
                else:
                    Emp.append(f)
                    En = En + 1
    F = np.vstack(F)
    return F, Emp, Fn, En

def _openImg(p, x):
    IMG = []
    for i in x:
        f = str(i) + '.jpg'
        img = cv2.imread(os.path.join(p, f))
        img_n = cv2.resize(img, (64, 96))
        IMG.append(img_n[np.newaxis, :, :, :])
    # IMG = np.vstack(IMG)
    return IMG


def _img(dic):
    P = dic['train']
    IL = []
    IM = []
    IR = []
    for p in P:
        n = 0
        fL = listdir(p)
        for nf in fL:
            if nf.split('.')[-1]=='jpg':
                n = n + 1 
                # img_tra = cv2.imread(os.path.join(p, nf))
                # I.append(img_tra[np.newaxis, :, :, :])
        M = _openImg(p, np.arange(0, n, 3))
        L = _openImg(p, np.arange(1, n, 3))
        R = _openImg(p, np.arange(2, n, 3))
        IM = IM + M
        IL = IL + L
        IR = IR + R
 
    return IM, IL, IR

def _rand(X, n):
    nX = random.sample(X, n)
    return nX

def _cal(F):
    B = np.zeros(5)
    for i in range(5):
        A = F[F[:,0]==i]
        B[i] = len(A)
    return B

def createLabels(data):
    for item in data:
        height = item.get_height()
        plt.text(
            item.get_x()+item.get_width()/2., 
            height*1, 
            '%d' % int(height),
            ha = "center",
            va = "bottom",
        )    

#%%
'''
F, Emp, Fn, En = _txt(dic)
class_num = _cal(F)
'''
n = 5
IM, IL, IR = _img(dic)
nIM, nIL, nIR = _rand(IM, n), _rand(IL, n), _rand(IR, n)
X = np.vstack(nIM + nIL + nIR).reshape(n*3, -1)

#%% t-SNE
print('=====t-SNE=====')
tsne = TSNE(n_components=2, init='random', random_state=5, perplexity=30)
X_tsne = tsne.fit_transform(X)

#%% Plt
'''
fig, ax = plt.subplots(1,1)
classes = [
    'TYPE_UNKNOWN',
    'TYPE_VEHICLE',
    'TYPE_PEDESTRIAN',
    'TYPE_SIGN',
    'TYPE_CYCLIST'
]
A = plt.bar(classes, class_num)
createLabels(A)
ax.set_ylabel('num')
ax.set_title('Number of class')
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig('./img/class.png')
# plt.show()

fig, ax = plt.subplots(1,1)
x = ['Empty', 'Exist']
F_num = np.array([En, Fn])
B = plt.bar(x, F_num)
createLabels(B)
ax.set_ylabel('num')
ax.set_title('Number of empty')
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig('./img/empty.png')
# plt.show()

fig, ax = plt.subplots(1,1)
x = ['Empty', 'Exist']
F_num = np.array([len(Emp), len(F)])
B = plt.bar(x, F_num)
createLabels(B)
ax.set_ylabel('num')
ax.set_title('Number of empty class')
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig('./img/empty_class.png')
# plt.show()
'''

fig, ax = plt.subplots(1,1)
colors = ['deepskyblue', 'crimson', 'lawngreen']
M_plt = ax.scatter(X[:n, 0], X[:n, 1], c=colors[0], alpha=0.5)
L_plt = ax.scatter(X[n:n*2, 0], X[n:n*2, 1], c=colors[1], alpha=0.5)
R_plt = ax.scatter(X[n*2:n*3, 0], X[n*2:n*3, 1], c=colors[2], alpha=0.5)
plt.legend((M_plt, L_plt, R_plt), ('Center', 'Left', 'Right'), scatterpoints=1, loc='lower left', ncol=3)
plt.title('t-SNE of Direction')
plt.tight_layout()
plt.savefig('./img/tSNE.png')

