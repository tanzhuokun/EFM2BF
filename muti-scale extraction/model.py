import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from layer import *
import time
import args



class MGEGFP(nn.Module):
	def __init__(self, adj_all, nview):
		super(MGEGFP, self).__init__()
		self.nview = nview
		self.adj_0 = adj_all[0]
		self.adj_1 = adj_all[1]
		self.adj_2 = adj_all[2]
		self.adj_3 = adj_all[3]




		self.base_gcns0 = GraphConvSparse(args.input_dim, args.hidden_dim, adj_all[0])
		self.base_gcns1 = GraphConvSparse(args.input_dim, args.hidden_dim, adj_all[1])
		self.base_gcns2 = GraphConvSparse(args.input_dim, args.hidden_dim, adj_all[2])
		self.base_gcns3 = GraphConvSparse(args.input_dim, args.hidden_dim, adj_all[3])




		self.out_gcns0 = GraphConvSparse(args.hidden_dim, args.hidden_dim, adj_all[0])
		self.out_gcns1 = GraphConvSparse(args.hidden_dim, args.hidden_dim, adj_all[1])
		self.out_gcns2 = GraphConvSparse(args.hidden_dim, args.hidden_dim, adj_all[2])
		self.out_gcns3 = GraphConvSparse(args.hidden_dim, args.hidden_dim, adj_all[3])



		self.final_gcns0 = GraphConvSparse(args.hidden_dim, args.hidden_dim, adj_all[0], activation=lambda x:x)
		self.final_gcns1 = GraphConvSparse(args.hidden_dim, args.hidden_dim, adj_all[1], activation=lambda x:x)
		self.final_gcns2 = GraphConvSparse(args.hidden_dim, args.hidden_dim, adj_all[2], activation=lambda x:x)
		self.final_gcns3 = GraphConvSparse(args.hidden_dim, args.hidden_dim, adj_all[3], activation=lambda x:x)



		self.common_gcn0_0 = GraphConvSparse_(args.input_dim, args.common_dim)
		self.common_gcn0_1 = GraphConvSparse_(args.input_dim, args.common_dim)
		self.common_gcn0_2 = GraphConvSparse_(args.input_dim, args.common_dim)
		self.common_gcn0_3 = GraphConvSparse_(args.input_dim, args.common_dim)

		self.common_gcn1 = GraphConvSparse_(args.common_dim, args.common_dim)
		self.common_gcn2 = GraphConvSparse_(args.common_dim, args.common_dim)





		self.gate0 = nn.ModuleList()
		self.gate1 = nn.ModuleList()
		self.gate2 = nn.ModuleList()
		self.gate3 = nn.ModuleList()


		for i in range(4):
			self.gate0.append(nn.Linear(args.hidden_dim + args.common_dim, 1, bias=True))
			self.gate1.append(nn.Linear(args.hidden_dim + args.common_dim, 1, bias=True))
			self.gate2.append(nn.Linear(args.hidden_dim + args.common_dim, 1, bias=True))
			self.gate3.append(nn.Linear(args.hidden_dim + args.common_dim, 1, bias=True))





	def forward(self, x_all,device):


		hidden0 = self.base_gcns0(x_all[0])
		hidden1 = self.base_gcns1(x_all[1])
		hidden2 = self.base_gcns2(x_all[2])
		hidden3 = self.base_gcns3(x_all[3])

		mid0 = self.out_gcns0(hidden0)
		mid1 = self.out_gcns1(hidden1)
		mid2 = self.out_gcns2(hidden2)
		mid3 = self.out_gcns3(hidden3)


		final0 = self.final_gcns0(mid0)
		final1 = self.final_gcns1(mid1)
		final2 = self.final_gcns2(mid2)
		final3 = self.final_gcns3(mid3)


		if args.layer_agg == 'mean':
			z_0 = torch.mean(torch.stack((hidden0,mid0,final0),dim=0),dim=0)
			z_1 = torch.mean(torch.stack((hidden1,mid1,final1),dim=0),dim=0)
			z_2 = torch.mean(torch.stack((hidden2,mid2,final2),dim=0),dim=0)
			z_3 = torch.mean(torch.stack((hidden3,mid3,final3),dim=0),dim=0)


		elif args.layer_agg == 'max':
			z_0 = torch.max(torch.stack((hidden0,mid0,final0),dim=0),dim=0)[0]
			z_1 = torch.max(torch.stack((hidden1,mid1,final1),dim=0),dim=0)[0]
			z_2 = torch.max(torch.stack((hidden2,mid2,final2),dim=0),dim=0)[0]
			z_3 = torch.max(torch.stack((hidden3,mid3,final3),dim=0),dim=0)[0]


		elif args.layer_agg == 'none':
			z_0 = final0
			z_1 = final1
			z_2 = final2
			z_3 = final3


		elif args.layer_agg == 'concat':
			z_0 = torch.cat([hidden0,mid0,final0],dim=1)
			z_1 = torch.cat([hidden1,mid1,final1],dim=1)
			z_2 = torch.cat([hidden2,mid2,final2],dim=1)
			z_3 = torch.cat([hidden3,mid3,final3],dim=1)

		else:
			print('wrong layer aggregation type')






		z_common0 = self.common_gcn0_0(x_all[0],self.adj_0)
		z_common0 = self.common_gcn1(z_common0,self.adj_0)
		z_common0 = self.common_gcn2(z_common0,self.adj_0)

		z_common1 = self.common_gcn0_1(x_all[1],self.adj_1)
		z_common1 = self.common_gcn1(z_common1,self.adj_1)
		z_common1 = self.common_gcn2(z_common1,self.adj_1)

		z_common2 = self.common_gcn0_2(x_all[2],self.adj_2)
		z_common2 = self.common_gcn1(z_common2,self.adj_2)
		z_common2 = self.common_gcn2(z_common2,self.adj_2)

		z_common3 = self.common_gcn0_3(x_all[3],self.adj_3)
		z_common3 = self.common_gcn1(z_common3,self.adj_3)
		z_common3 = self.common_gcn2(z_common3,self.adj_3)
		#

		z0 = torch.cat([z_common0, z_0],dim=1)
		z1 = torch.cat([z_common1, z_1],dim=1)
		z2 = torch.cat([z_common2, z_2],dim=1)
		z3 = torch.cat([z_common3, z_3],dim=1)


		z = [z0,z1,z2,z3]
		# z = [z0]

		score0 = torch.empty(22325, 4).to(device)
		score1 = torch.empty(22325, 4).to(device)
		score2 = torch.empty(22325, 4).to(device)
		score3 = torch.empty(22325, 4).to(device)



		if args.act_mg == 'relu':
			for i in range(4):
				score0[:,i] = F.relu(self.gate0[i](z[i]).squeeze())
				score1[:,i] = F.relu(self.gate1[i](z[i]).squeeze())
				score2[:,i] = F.relu(self.gate2[i](z[i]).squeeze())
				score3[:,i] = F.relu(self.gate3[i](z[i]).squeeze())


		elif args.act_mg == 'none':
			for i in range(4):
				score0[:,i] = self.gate0[i](z[i]).squeeze()
				score1[:,i] = self.gate1[i](z[i]).squeeze()
				score2[:,i] = self.gate2[i](z[i]).squeeze()
				score3[:,i] = self.gate3[i](z[i]).squeeze()

		elif args.act_mg == 'tanh':
			for i in range(4):
				score0[:,i] = F.tanh(self.gate0[i](z[i]).squeeze())
				score1[:,i] = F.tanh(self.gate1[i](z[i]).squeeze())
				score2[:,i] = F.tanh(self.gate2[i](z[i]).squeeze())
				score3[:,i] = F.tanh(self.gate3[i](z[i]).squeeze())


		elif args.act_mg == 'leakyrelu':
			for i in range(4):
				score0[:,i] = F.leaky_relu(self.gate0[i](z[i]).squeeze())
				score1[:,i] = F.leaky_relu(self.gate1[i](z[i]).squeeze())
				score2[:,i] = F.leaky_relu(self.gate2[i](z[i]).squeeze())
				score3[:,i] = F.leaky_relu(self.gate3[i](z[i]).squeeze())



		else:
			print('wrong activation type')



		score0 = torch.softmax(score0,dim=1)


		score0 = F.tanh(score0)
		score1 = F.tanh(score1)
		score2 = F.tanh(score2)
		score3 = F.tanh(score3)



		z_global_0 = (score0[:, 0].view(-1, 1)) * z0 + (score0[:, 1].view(-1, 1)) * z1 + (score0[:, 2].view(-1, 1)) * z2 + (score0[:, 3].view(-1, 1)) * z3
		z_global_1 = (score1[:, 0].view(-1, 1)) * z0 + (score1[:, 1].view(-1, 1)) * z1 + (score1[:, 2].view(-1, 1)) * z2 + (score1[:, 3].view(-1, 1)) * z3
		z_global_2 = (score2[:, 0].view(-1, 1)) * z0 + (score2[:, 1].view(-1, 1)) * z1 + (score2[:, 2].view(-1, 1)) * z2 + (score2[:, 3].view(-1, 1)) * z3
		z_global_3 = (score3[:, 0].view(-1, 1)) * z0 + (score3[:, 1].view(-1, 1)) * z1 + (score3[:, 2].view(-1, 1)) * z2 + (score3[:, 3].view(-1, 1)) * z3






		assert not torch.any(torch.isnan(z0))
		assert not torch.any(torch.isnan(z1))
		assert not torch.any(torch.isnan(z2))
		assert not torch.any(torch.isnan(z3))





		z_global=[z_global_0,z_global_1,z_global_2,z_global_3]
		z_out = [z0, z1, z2, z3]


		return z_global,z_out





##
