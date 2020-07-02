import numpy as np
import matplotlib.pyplot as plt
from constant_name import buttom
from pandas import DataFrame
def compute_confusion_matrix(pre,label):
    # names = np.unique(label)
    # ccm = confusion_matrix(y_true=label,y_pred=pre,
    #                        labels=names)
    # print(ccm)

    #如果传入是字母标签，也能计算
    _,pre_l = np.unique(pre,return_inverse=True)
    names,label_l = np.unique(label,return_inverse=True)
    l = len(names)
    cm = np.zeros((l,l))
    cm = list(cm)
    cm = [list(x) for x in cm]
    for i in range(len(label_l)):
        cm[label_l[i]][pre_l[i]] += 1

    if buttom['pre']:
        plt.matshow(cm)
        plt.colorbar()
        plt.show()
    # return cm
    for i in range(len(cm)):
        t=sum(cm[i])
        cm[i].append(t)
    p = []
    for i in range(len(cm[0])):
        t = [x[i] for x in cm]
        p.append(sum(t))

    print('y_true','\n')
    cou = 0
    for i in range(len(cm)-1,-1,-1):
        print('class:',i,':',end='')
        for item in cm[i]:
            print('%5d'%item,end='')
        # print('   ')
        cou += cm[i][i]
        print('  class_acc:','%.2f' %(cm[i][i]/cm[i][-1]))
        print('\n')


    print('predi_l ',':',end='')

    for item in p:
        print('%5d' % item, end='')
    cou = int(cou)
    print('  class_acc:','%.2f' %(cou/p[-1]))
    print('\n')
    return cm

def compute_confusion_matrix_ver2(pre,label):
    _,pre_l = np.unique(pre,return_inverse=True)
    names,label_l = np.unique(label,return_inverse=True)
    l = len(names)
    cm = np.zeros((l,l))
    cm = list(cm)
    cm = [list(x) for x in cm]
    for i in range(len(label_l)):
        cm[label_l[i]][pre_l[i]] += 1
    for i in range(len(cm)):
        t=sum(cm[i])
        cm[i].append(t)
    return cm

def draw_line(y_list,x_list = None):
    #y为对应横点的所有Y点
    if x_list==None:
        l = len(y_list)
        x_list = [i for i in range(1,l+1)]
    t_line = []
    for i in range(len(y_list[0])):
        t = [x[i] for x in y_list]
        t_line.append(t)

    for i in range(len(t_line)):
        plt.plot(x_list,t_line[i],label='acc_class'+str(i))
        plt.legend()
    plt.xlabel('num_epoch')
    plt.ylabel('class_acc')
    plt.show()
    # cmd = input('save_or_not:')
    # if cmd =='yes':
    #     plt.savefig()

def compute_output(f1,f2,ttag):
    import pandas as pd
    fi1 = pd.read_csv(f1,header=0,index_col=0)
    fi2 = pd.read_csv(f2,header=0,index_col=0)
    f1 = list(fi1['0'])
    f2 = list(fi2['0'])
    c = len(f1)
    index = [i for i in range(c)]
    if c==len(f2):
        for i in range(c):
            if f1[i] == 0:
                if f2[i] == 0:
                    index[i] = 0
                else:
                    index[i] = 1
            else:
                index[i] = 2
    fs = DataFrame(index)
    fs.index = [i for i in range(c)]
    fs.columns = [i for i in range(fs.shape[1])]
    fs.to_csv(r'outcome/final_'+ttag+'.csv')






if __name__ == '__main__':
    compute_output(r'D:\PROJECT\finall_at\outcome\o1\s1_0724_1132.csv',
                   r'D:\PROJECT\finall_at\outcome\o2\s2_0724_1525.csv')
    # y_true = [1,2,0,3,4,1,2,4,1,2,1,3,1,1]
    # y_pred = [1,2,0,3,4,1,2,4,2,1,3,1,1,4]
    # x = [i for i in range(14)]
    # draw_line([[1,2],[2,3],[1,2],[2,3],[1,2],[2,3],[1,2]])
    # # c = compute_confusion_matrix(y_pred,y_true)
