import numpy as np
import os
from os import listdir
import matplotlib.pyplot as plt

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
F, Emp, Fn, En = _txt(dic)
class_num = _cal(F)

#%% Plt

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


