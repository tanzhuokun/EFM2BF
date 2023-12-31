import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler,minmax_scale

import scipy.sparse as sp
import numpy as np
import os
import time


from preprocessing import *

import args
import model
from layer import BPl_decode,dot_product_decode




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# gpu不够
# device = torch.device("cpu")
torch.cuda.empty_cache()
#设置试图大小
n_view = 4


# load data
if args.data_type == 'brain':

    adj_weight0 = np.load('./data/BFC_filtered_last.npz')['adj']
    adj_weight1 = np.load('./data/coexpression_last.npz')['adj']
    adj_weight2 = np.load('./data/database_last.npz')['adj']
    adj_weight3 = np.load('./data/experimental_last.npz')['adj']

    features0 = np.load('./data/BFC_filtered_last.npz')['rwr']
    features1 = np.load('./data/coexpression_last.npz')['rwr']
    features2 = np.load('./data/database_last.npz')['rwr']
    features3 = np.load('./data/experimental_last.npz')['rwr']


else:
    print('wrong data type')






adj_weight = [adj_weight0,adj_weight1,adj_weight2,adj_weight3]
features_rwr = [features0,features1,features2,features3]
adj_all=[]
adj_label_all=[]
adj_norm_all=[]
pos_weight=[]
norm_all=[]
weight_mask=[]
weight_tensor_all=[]
features_all=[]
N = adj_weight0.shape[0]
# cnt=1

for i in range(n_view):
    # adj[0]=1
    adj = sp.coo_matrix(np.where(adj_weight[i]>0,1,0))
    adj_all.append(adj)
    pos_weight = float(N * N - adj.sum()) / adj.sum()
    norm = N * N / float((N * N - adj.sum()) * 2)
    norm_all.append(norm)

    adj_label = sparse_to_tuple(adj + sp.eye(adj.shape[0]))
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),torch.FloatTensor(adj_label[1]),torch.Size(adj_label[2]))
    adj_label = adj_label.to(device)
    adj_label_all.append(adj_label)

    adj_norm = preprocess_graph(adj_weight[i])
    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),torch.FloatTensor(adj_norm[1]),torch.Size(adj_norm[2]))
    adj_norm = adj_norm.to(device)
    adj_norm_all.append(adj_norm)

    # features = sp.coo_matrix(minmax_scale(features[i], axis=0))
    features = sp.coo_matrix(features_rwr[i])
    features = sparse_to_tuple(features.tocoo())
    features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),torch.FloatTensor(features[1]),torch.Size(features[2]))
    features = features.to(device)
    features_all.append(features)

    weight_mask = adj_label.to_dense().view(-1) == 1  # view就是reshape view（-1）就是给拉成1行
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight  #num(0)/num(1)
    weight_tensor = weight_tensor.to(device)
    weight_tensor_all.append(weight_tensor)
    # print(cnt)
    # cnt+=1





# model.MGEGFP(adj_norm_all, 6).to(device)
Model = model.MGEGFP(adj_norm_all, 4).to(device)



# for name, param in Model.named_parameters():
#     if param.requires_grad:
#         print(name)


LocalityConstrains = nn.CosineEmbeddingLoss()
criterion = nn.MSELoss(reduction='mean')
optimizer = Adam(Model.parameters(),
                 lr=args.learning_rate,
                 # eps=1e-08,
                 weight_decay=0)




best = 1e9
best_t = 0
# loss_all = []
# epoch_all = []
cnt_wait = 0





for epoch in range(args.num_epoch):

    optimizer.zero_grad()
    t = time.time()



    # MGEGFP
    z_global,z_out = Model(features_all,device)


    # reconstruction loss
    loss_A = 0.
    for i in range(n_view):
        loss_temp = norm_all[i]* F.binary_cross_entropy(dot_product_decode(z_global[i]).view(-1),
                                                adj_label_all[i].to_dense().view(-1),
                                                weight=weight_tensor_all[i])
        loss_A+=loss_temp

    loss_A = loss_A/n_view
    loss = loss_A




    loss.backward()
    optimizer.step()




    if loss < best:
        best = loss
        best_t = epoch + 1
        cnt_wait = 0

        torch.save(Model.state_dict(), 'MGEGFP_best.pkl')


    else:
        cnt_wait += 1

    if cnt_wait == args.patience:
        print('Early stopping!')
        break

    # print("Epoch:", '%04d' % (epoch + 1),
    #       "train_loss=", "{:.5f}".format(loss.item()),
    #       "loss_A=", "{:.5f}".format(loss_A.item()),
    #       "loss_cos=", "{:.5f}".format(loss_cos.item()),
    #       "time=", "{:.5f}".format(time.time() - t))
    print("Epoch:", '%04d' % (epoch + 1),
          "train_loss=", "{:.5f}".format(loss.item()),
          "loss_A=", "{:.5f}".format(loss_A.item()),

          "time=", "{:.5f}".format(time.time() - t))





print('load best model'+str(best_t))

Model.load_state_dict(torch.load('MGEGFP_best.pkl'))

z_global_best,z_out_best = Model(features_all,device)


np.save('z0_best.npy', z_out_best[0].cpu().detach().numpy())
np.save('z1_best.npy', z_out_best[1].cpu().detach().numpy())
np.save('z2_best.npy', z_out_best[2].cpu().detach().numpy())
np.save('z3_best.npy', z_out_best[3].cpu().detach().numpy())


z0_best = z_out_best[0].cpu().detach().numpy()
z1_best = z_out_best[1].cpu().detach().numpy()
z2_best = z_out_best[2].cpu().detach().numpy()
z3_best = z_out_best[3].cpu().detach().numpy()


z_all_best = np.hstack((z0_best,z1_best,z2_best,z3_best))



np.save('z_all_best.npy', z_all_best)



