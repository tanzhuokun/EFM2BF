from tqdm import tqdm
import numpy as np
import logging
import torch
from sklearn.decomposition import PCA
# utils.py 的文件里有一些通用的函数

def pcc(u, v, eps=1e-8):
    u, v = u - torch.mean(u, dim=-1, keepdims=True), v - torch.mean(v, dim=-1, keepdims=True)
    # torch.unsqueeze(input, dim, out=None) tensor (Tensor) – 输入张量； dim (int) – 插入维度的索引
    #https://zhuanlan.zhihu.com/p/86763381
    u, v = torch.unsqueeze(u, 1), torch.unsqueeze(v, 0)
    return torch.sum(u * v, dim=-1) / (torch.sqrt(torch.sum(u ** 2, dim=-1)) * torch.sqrt(torch.sum(v ** 2, dim=-1)) + eps)


def extractConstraints(representation):

    # Laplace  X的一瞥（通过自动编码器重构的基因的特征表示）
    # representation = representation + np.eye(representation.shape[0])# np.eye(representation.shape[0])生成对角线为1的矩阵
    # D = representation.sum(axis=1)
    # D_ = np.diag(np.power(D, -0.5)) #计算D的-0.5次方
    # representation = np.dot(np.dot(D_, representation), D_) #文中的（7）

    # 本身就是经过了gcn处理了，上述的拉普拉斯正则化处理，不需要了
    # PCA  https://www.cnblogs.com/pinard/p/6243025.html，https://blog.csdn.net/qq_20135597/article/details/95247381
    pca = PCA(n_components=600)#原来是600
    representation = pca.fit_transform(representation) #返回降维后的数据
    # 使用detach返回的tensor和原始的tensor共同一个内存，返回一个新的tensor，且requires_grad为false
    representation = torch.from_numpy(representation).float().cuda().detach()

    pcc_mat = np.zeros((representation.shape[0], representation.shape[0]), dtype='float')
    for i in tqdm(range(0, representation.shape[0], 10)):
        pcc_mat[i:i + 10] = pcc(representation[i:i + 10], representation).cpu().numpy()

    pcc_mat = np.abs(pcc_mat)
    # print("=====PCC值=======")
    # print(pcc_mat)

    return pcc_mat


def obtain_constraints(net_numbs, emb, symbols, topN, idx_layer):

    pcc_mats = []
    for idx_net in range(net_numbs):#IDX index
        pcc_mat = extractConstraints(emb[idx_net])
        pcc_mats.append(pcc_mat)

    must_links = []
    # threshold_list = [0.2,0.2,0.2,0.2]
    for i, pcc_mat in enumerate(pcc_mats):
        np.fill_diagonal(pcc_mat, 0) #numpy数组的对角线填充为参数中传递的值，将矩阵pcc_mat的对角线填充为0

        pcc_order = np.sort(pcc_mat.flatten())#扁平化，将多行转变为一行，并且pcc排序
        print('排序的数量:',len(pcc_order),'选取的数量:',topN[i])
        threshold_max = pcc_order[-topN[i]]
        # print("=====排序后的PCC值=======")
        # print(pcc_order,pcc_order[-topN[i]])
        # threshold_max = threshold_list[i]
        # # # 设置
        # #
        # if pcc_order[-topN[i]] < threshold_list[i]:
        #     threshold_max = pcc_order[-topN[i]]
        # print(threshold_max)
        must_link = (pcc_mat >= threshold_max) #.astype('float')
        must_links.append(must_link)
    print("=====必要约束=======")
    print(must_links)
    #############################################################################################################
    for i in range(net_numbs):
        for j in range(i):
            #np.intersect1d返回两个数组中共同的元素（注意：是排序后的）； return_indices=True返回输出及其对应的索引
            # xy返回的是相同的值的矩阵，x_indx, y_inds返回的分别是相同值在symbols[i], symbols[j]的索引（已排序的）
            xy, x_indx, y_inds = np.intersect1d(symbols[i], symbols[j], return_indices=True)
            tmp = must_links[i][x_indx][:, x_indx] + must_links[j][y_inds][:, y_inds]
            # np.ix_输入两个数组，产生笛卡尔积的映射关系 ，能把两个一维数组 转换为 一个用于选取方形区域的索引器
            # https://www.cnblogs.com/yqxg/p/10615300.html
            must_links[i][np.ix_(x_indx, x_indx)] = tmp 
            must_links[j][np.ix_(y_inds, y_inds)] = tmp
        
        logging.info('### Network {}: Number of Must link: {}'.format(i, must_links[i].sum()))
    #############################################################################################################
    return must_links
