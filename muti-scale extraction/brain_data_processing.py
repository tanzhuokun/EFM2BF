import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale, scale #minmax_scale 归一化
def _scaleSimMat(A):
    """Scale rows of similarity matrix"""
    A = A - np.diag(np.diag(A)) # np.diag(np.diag(A))  返回的是A对角线的矩阵
    A = A + np.diag(A.sum(axis=0) == 0)
    col = A.sum(axis=0)
    A = A.astype(np.float)/col[:, None]

    return A

def RWR(A, K=6, alpha=0.96):
    """Random Walk on graph"""
    A = _scaleSimMat(A)
    # Random surfing
    n = A.shape[0] #行长
    P0 = np.eye(n, dtype=float)
    P = P0.copy() #返回对象的浅复制，随着P0变化而变化
    M = np.zeros((n, n), dtype=float)
    for i in range(0, K):
        P = alpha*np.dot(P, A) + (1. - alpha)*P0
        M = M + P

    return M
# adj_path = ['adj_coex','adj_coo','adj_da']
# rwr_path = ['rwr_coex','rwr_coo','rwr_da']

# path = ['ppi_subnetwork_coexpression.npz']
# save_path =['coexpression.npz']

# 'ppi_subnetwork_cooccurence.npz',  # 2968
# 'ppi_subnetwork_fusion.npz',  # 6200
# 'ppi_subnetwork_neighborhood.npz'  # 4039
path = ['ppi_subnetwork_database.npz',#10610
        'ppi_subnetwork_coexpression.npz',#19025
        'BFC_based_network_filtered.npz',#6007
        'ppi_subnetwork_experimental.npz',#18317
        'ppi_subnetwork_cooccurence.npz',
        'ppi_subnetwork_fusion.npz',
        'ppi_subnetwork_neighborhood.npz'
 ]
save_path =['database_last.npz',
            'coexpression_last.npz',
            'BFC_filtered_last.npz',
            'experimental_last.npz',
            'cooccurence_last.npz',
            'fusion_last.npz',
            'neighborhood_last.npz'
            ]
res_path =['z0_best.npy',
            'z1_best.npy',
            'z2_best.npy',
            'z3_best.npy',
            'z4_best.npy',
            'z5_best.npy',
            'z6_best.npy'
           ]
res_path =['z_all_best_database.npy',
            'z_all_best_coexpression.npy',
            'z_all_best_bfc.npy',
            'z_all_best_experimental.npy',
            'z_all_best_cooccurence.npy',
            'z_all_best_fusion.npy',
            'z_all_best_neighborhood.npy'
           ]
intergration_path = ['ppi_subnetwork_database_gcn.npz',
                     'ppi_subnetwork_coexpression_gcn.npz',
                     'BFC_based_network_filtered_gcn.npz',
                     'ppi_subnetwork_experimental_gcn.npz',
                     'ppi_subnetwork_cooccurence_gcn.npz',
                     'BFC_based_network_fusion_gcn.npz',
                     'ppi_subnetwork_neighborhood_gcn.npz'
                     ]

fear_path =['database.npz',
            'coexpression.npz',
            'BFC_filtered.npz',
            'experimental.npz'
            'cooccurencet.npz',
            'fusion.npz',
            'neighborhood.npz']


def adj_rwr_get(file,save_path):
    # 消除因为np版本的报错,或者在load中加allow_pickle=True
    # np.load.__defaults__ = (None, True, True, 'ASCII')
    fear_all = np.load(file,allow_pickle=True)
    # np.load.__defaults__ = (None, False, True, 'ASCII')

    temp_vec = fear_all['corr']
    # 特征长度
    len = temp_vec.shape[0]
    # txt文件转为npy文件
    adj_vec = temp_vec>0  # 定义size  共表达  19025
    # 便于自连接
    emd_martrx = np.eye(len)
    adj_vec = adj_vec + emd_martrx
    # 把链接矩阵保存
    # fear_all['adj'] = adj_vec
    rwr_vec = RWR(temp_vec)
    # fear_all['rwr'] = rwr_vec
    print("特征shape：",fear_all['corr'].shape)
    np.savez(save_path, adj=adj_vec,rwr=rwr_vec,corr=fear_all['corr'], symbol=fear_all['symbol'])

def intergration(input1,input2,output):
    fear_all = np.load(input1,allow_pickle=True)
    fear_new = np.load(input2,allow_pickle=True)
    print(output," 的特征shape：", fear_new.shape)
    # np.savez(output, corr=fear_new, symbol=fear_all['symbol'])
    np.savez(output, features=fear_new, symbol=list(fear_all['symbol'].tolist()))

def fear_expand(temp_df,path,symbol):
    temp_rwr_df,temp_adj_df = temp_df,temp_df
    fear = np.load(path, allow_pickle=True)
    fear_rwr_df = pd.DataFrame(fear['rwr'], index=fear['symbol'], columns=fear['symbol'])
    fear_adj_df = pd.DataFrame(fear['adj'], index=fear['symbol'], columns=fear['symbol'])
    # fear_corr_df = pd.DataFrame(fear['corr'], index=fear['symbol'], columns=fear['symbol'])
    #### 时间太长
    # for i in fear['symbol']:
    #     for j in fear['symbol']:
    #         temp_rwr_df.loc[i][j] = fear_rwr_df.loc[i][j]
    #         temp_adj_df.loc[i][j] = fear_adj_df.loc[i][j]
            # temp_corr_df.loc[i][j] = fear_corr_df.loc[i][j]

    symbol_other = list(set(symbol) - set(fear['symbol']))
    # 增加多行

    new_index = fear_rwr_df.index.tolist() + symbol_other
    temp_rwr_df = fear_rwr_df.reindex(new_index)
    temp_adj_df = fear_adj_df.reindex(new_index)
    # print('\n增加多行：\n', df)



    # 增加多列
    new_col = fear_rwr_df.columns.tolist() + symbol_other
    temp_rwr_df = temp_rwr_df.reindex(columns=new_col)
    temp_adj_df = temp_adj_df.reindex(columns=new_col)


    # 填充
    temp_rwr_df.fillna(0.,inplace=True)
    temp_adj_df.fillna(0.,inplace=True)




    # np.savez(path.split('.')[0] + '_last.npz', adj=temp_adj_df, rwr=temp_rwr_df, corr=temp_corr_df, symbol=symbol)
    np.savez(path.split('.')[0] + '_last.npz', adj=temp_adj_df, rwr=temp_rwr_df, symbol=new_index)

if __name__ == '__main__':

    # # 查看数据,结果说明BFC网络和其他的网络之间都是具有相同的symbol，进行卷积的时候将网络没有的填充0，有的继续使用
    # fear1 = np.load('data/BFC_filtered.npz', allow_pickle=True)
    # fear2 = np.load('BFC_based_network_filtered_gcn.npz', allow_pickle=True)
    # fear22 = np.load('ppi_subnetwork_database_gcn.npz', allow_pickle=True)
    # fear1_s = fear1['symbol']
    # fear2_s = fear2['symbol']
    # # c = set(fear1_s) & set(fear2_s)
    # print()

    # # 将所有网络处理成同样大小
    # fear1 = np.load('data/BFC_filtered.npz', allow_pickle=True)
    # fear2 = np.load('data/coexpression.npz', allow_pickle=True)
    # fear3 = np.load('data/database.npz', allow_pickle=True)
    # fear4 = np.load('data/experimental.npz', allow_pickle=True)
    # # 并集
    # symbol = set(fear1['symbol']) | set(fear2['symbol']) | set(fear3['symbol']) | set(fear4['symbol'])
    #
    # print('四个网络并集大小：',len(symbol))
    # # 遍历每个网络
    # for path in fear_path:
    #     temp_vec = np.zeros((len(symbol), len(symbol)))
    #     temp_vec_df = pd.DataFrame(temp_vec, index=list(symbol), columns=list(symbol))
    #     fear_expand(temp_vec_df,'data/'+path,symbol)


    print()
    for i in range(len(res_path)):
        print("正在处理：", path[i])
        ### 获取初步数据 ###
        # adj_rwr_get('data/' + path[i],'训练出来的结果/' +save_path[i])
        #### 跑完gcn再整合成 ###
        intergration('训练出来的结果/' + save_path[i], '训练出来的结果/' +res_path[i], '训练出来的结果/' +intergration_path[i])
        print("处理完成！")



