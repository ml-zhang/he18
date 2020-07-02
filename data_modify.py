import pandas as pd
import numpy.random as nr


# nr.seed(15)
# nr.shuffle() shuffle返回值是none，因为他是原地修改函数
# fi.iloc[1:2] 以行向量形式取出第i行
def direct_steam(timetag, ncge=2500, nlge=3000, nmge=3500, rand_seed=15):
    #
    nr.seed(rand_seed)
    ti = timetag
    fi = pd.read_csv(r'D:\PROJECT\finall_at\data\all_3\f_train_18000.csv', index_col=0, header=0)
    fi.index = [i for i in range(fi.shape[0])]
    fi.columns = [i for i in range(fi.shape[1])]

    fi1 = fi.iloc[:3000]
    fi1.index = [i for i in range(0, 3000)]
    fi2 = fi.iloc[3000:10000]
    fi2.index = [i for i in range(7000)]
    fi3 = fi.iloc[10000:18000]
    fi3.index = [i for i in range(8000)]

    cn = [i for i in range(0, 3000)]
    ln = [i for i in range(0, 7000)]
    mn = [i for i in range(0, 8000)]

    nr.shuffle(cn)
    test_c = cn[0:ncge]
    nr.shuffle(ln)
    test_l = ln[0:nlge]
    nr.shuffle(mn)
    test_m = mn[0:nmge]

    tc = fi1.iloc[test_c[0]:test_c[0] + 1]
    for i in test_c[1:]:
        c = fi1.iloc[i:i + 1]
        tc = pd.concat((tc, c))

    tc.index = [i for i in range(ncge)]

    tl = fi2.iloc[test_l[0]:test_l[0] + 1]
    for i in test_l[1:]:
        c = fi2.iloc[i:i + 1]
        tl = pd.concat((tl, c))
    tl.index = [i for i in range(ncge, nlge + ncge)]

    tm = fi3.iloc[test_m[0]:test_m[0] + 1]
    for i in test_m[1:]:
        c = fi3.iloc[i:i + 1]
        tm = pd.concat((tm, c))
    tm.index = [i for i in range(ncge + nlge, ncge + nlge + nmge)]

    print('start_concat')
    save_name_tr = r'D:\PROJECT\finall_at\data\all_3_mul\f_tr_' + ti + '.csv'
    save_name_te = r'D:\PROJECT\finall_at\data\all_3_mul\f_tr_label_' + ti + '.csv'
    f_tr = pd.concat((tc, tl, tm))
    f_tr.to_csv(save_name_tr)
    un = ncge * [0]
    up = nlge * [1]
    unp = nmge * [2]
    un.extend(up)
    un.extend(unp)
    nu = pd.DataFrame(un)
    nu.index = [i for i in range(ncge + nlge + nmge)]
    nu.columns = [0]
    nu.to_csv(save_name_te)
    return save_name_tr, save_name_te


def dm(path, gene, fi_name, cge_num, lge_num, mge_num, tcn, tln, tmn, raw_data=None):
    fi = pd.read_csv(path + gene + fi_name, index_col=0, header=0)
    fi_name = fi_name[:-4]
    fi.index = [i for i in range(fi.shape[0])]
    fi.columns = [i for i in range(fi.shape[1])]

    fi1 = fi.iloc[:cge_num]
    fi1.index = [i for i in range(0, cge_num)]
    fi2 = fi.iloc[cge_num:(cge_num + lge_num)]
    fi2.index = [i for i in range(lge_num)]
    fi3 = fi.iloc[cge_num + lge_num:(cge_num + lge_num + mge_num)]
    fi3.index = [i for i in range(mge_num)]

    cn = [i for i in range(cge_num)]
    ln = [i for i in range(lge_num)]
    mn = [i for i in range(mge_num)]

    nr.shuffle(cn)
    test_c = cn[0:tcn]
    nr.shuffle(ln)
    test_l = ln[0:tln]
    nr.shuffle(mn)
    test_m = mn[0:tmn]

    tc = fi1.iloc[test_c[0]:test_c[0] + 1]
    for i in test_c[1:]:
        c = fi1.iloc[i:i + 1]
        tc = pd.concat((tc, c))
    for i in test_c:
        fi1 = fi1.drop(i)
    fi1.index = [i for i in range(cge_num - tcn)]
    tc.index = [i for i in range(tcn)]

    tl = fi2.iloc[test_l[0]:test_l[0] + 1]
    for i in test_l[1:]:
        c = fi2.iloc[i:i + 1]
        tl = pd.concat((tl, c))
    for i in test_l:
        fi2 = fi2.drop(i)
    fi2.index = [i for i in range(fi1.shape[0], fi1.shape[0] + lge_num - tln)]
    tl.index = [i for i in range(tcn, tcn + tln)]

    tm = fi3.iloc[test_m[0]:test_m[0] + 1]
    for i in test_m[1:]:
        c = fi3.iloc[i:i + 1]
        tm = pd.concat((tm, c))
    for i in test_m:
        fi3 = fi3.drop(i)
    fi3.index = [i for i in range(fi2.shape[0], fi2.shape[0] + mge_num - tmn)]
    tm.index = [i for i in range(tcn + tln, tcn + tln + tmn)]

    print('start_concat')

    f_test_s2 = pd.concat((tc, tl))
    f_test = pd.concat((f_test_s2, tm))
    f_test.to_csv(path + gene + fi_name + '_te.csv')
    un = tcn * [0]
    unp = tln * [1]
    un.extend(unp)
    unp = tmn * [2]
    un.extend(unp)
    un = pd.DataFrame(un)
    un.index = [i for i in range(un.shape[0])]
    un.columns = [0]
    un.to_csv(path + gene + fi_name + '_te_l.csv')

    f_tr = pd.concat((fi1, fi2, fi3))
    f_tr.to_csv(path + gene + fi_name + '_tr.csv')
    un = (cge_num - tcn) * [0]
    unp = (lge_num - tln) * [1]
    un.extend(unp)
    unp = (mge_num - tmn) * [2]
    un.extend(unp)
    un = pd.DataFrame(un)
    un.index = [i for i in range(un.shape[0])]
    un.columns = [0]
    un.to_csv(path + gene + fi_name + '_tr_l.csv')


def dd():
    fi = pd.read_csv(r'D:\PROJECT\finall_at\data\all_3\f_train_18000.csv', index_col=0, header=0)
    fi1 = fi.iloc[0:3000]
    fi2 = fi.iloc[4500:7500]
    fi3 = fi.iloc[11000:14000]
    f = pd.concat((fi1, fi2, fi3))
    f.index = [i for i in range(9000)]
    f.to_csv(r'D:\PROJECT\finall_at\data\all_3\f_train_9000.csv')

    a = 3000 * [0]
    # d = 6000*[0]
    b = 3000 * [1]
    c = 3000 * [2]
    a.extend(b)
    a.extend(c)
    # a.extend(d)
    nu = a
    nu = pd.DataFrame(nu)
    nu.index = [i for i in range(9000)]
    nu.columns = [0]
    nu.to_csv(r'D:\PROJECT\finall_at\data\all_3\tr_9000_label.csv')


def dp(path, gene, fi_name):
    fi = pd.read_csv(path + gene + fi_name, index_col=0, header=0)
    print(fi.shape)
    fi_name = fi_name[:-4] + '_gene.csv'
    fi.index = [i for i in range(fi.shape[0])]
    fi.columns = [i for i in range(fi.shape[1])]
    fi.to_csv(path + gene + fi_name)


if __name__ == '__main__':
    nl = [('278gene/', '278_gene.csv', 'pre_278.csv'), ('1046gene/', '1046_gene.csv', 'pre_1046.csv'),
          ('allgene/', 'all_gene.csv', 'pre_all.csv'), ('3000gene/', '3000_gene.csv', 'pre_3000.csv')]
    i=3
    dm(path='D:/PROJECT/md/data/', gene=nl[i][0], fi_name=nl[i][1], cge_num=2872, lge_num=2735, mge_num=1551,
       tcn=535, tln=572, tmn=351)
    dp(path='D:/PROJECT/md/data/', gene=nl[i][0], fi_name=nl[i][2])
