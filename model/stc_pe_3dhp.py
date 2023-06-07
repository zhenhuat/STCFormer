import torch
import torch.nn as nn
# from model.module.trans import Transformer as Transformer_s
# from model.module.trans_hypothesis import Transformer
import numpy as np
from einops import rearrange
from collections import OrderedDict
from torch.nn import functional as F
from torch.nn import init
import scipy.sparse as sp

from timm.models.layers import DropPath


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        layers, channel, d_hid, length = args.layers, args.channel, args.d_hid, args.frames
        self.num_joints_in, self.num_joints_out = args.n_joints, args.out_joints
        args.d_hid = 256
        isTrainning = args.train

        # dimension tranfer
        self.pose_emb = nn.Linear(2, args.d_hid, bias=False)
        self.gelu = nn.GELU()

        # self.flow_emb = nn.Linear(2, args.d_hid, bias=False)
        # self.gelu = nn.GELU()

        self.mlpmixer = MlpMixer(6, args.frames, 17, args.d_hid, isTrainning)

        self.pose_lift = nn.Linear(args.d_hid, 3, bias=False)

        # self.sequence_pos_encoder = PositionalEncoding(args.d_hid, 0.1)

        # self.tem_pool = nn.AdaptiveAvgPool1d(1)
        # self.lpm = LearnedPosMap2D(args.frames,18)

    def forward(self, x):
      	#x = x[:, :, :, :, 0].permute(0, 2, 3, 1).contiguous()  # B,T,J,2,1
        x = x[:, :, :, :, 0].permute(0, 2, 3, 1).contiguous()  # B,T,J,2,1
        #x = x.view(x.shape[0], x.shape[1], x.shape[2], -1)  # b,t,j,2

        b, t, j, c = x.shape

        #g = torch.zeros([b,t,1,c]).cuda()
        #x = torch.cat((x,g),-2)

        x = self.pose_emb(x)
        x = self.gelu(x)


        # x = x.reshape(b,t,j,c)

        x = self.mlpmixer(x)


        x = self.pose_lift(x)

        return x


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_mx_from_edges(num_pts, edges, sparse=False):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)
    # print(11,adj_mx)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    # adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    return adj_mx.sum(-1)


class ChebConv(nn.Module):
    """
    The ChebNet convolution operation.
    :param in_c: int, number of input channels.
    :param out_c: int, number of output channels.
    :param K: int, the order of Chebyshev Polynomial.
    """

    def __init__(self, in_c, out_c, K, bias=True, normalize=True):
        super(ChebConv, self).__init__()
        self.normalize = normalize

        self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c, out_c))  # [K+1, 1, in_c, out_c]
        init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
            init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

        self.K = K + 1

    def forward(self, inputs, graph):
        """
        :param inputs: the input data, [B, N, C]
        :param graph: the graph structure, [N, N]
        :return: convolution result, [B, N, D]
        """
        L = ChebConv.get_laplacian(graph, self.normalize)  # [N, N]
        mul_L = self.cheb_polynomial(L).unsqueeze(1)  # [K, 1, N, N]

        result = torch.matmul(mul_L, inputs)  # [K, B, N, C]

        result = torch.matmul(result, self.weight)  # [K, B, N, D]
        result = torch.sum(result, dim=0) + self.bias  # [B, N, D]

        return result

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.
        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)  # [N, N]
        multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.float)  # [K, N, N]
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)

        if self.K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if self.K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, self.K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k - 1]) - \
                                               multi_order_laplacian[k - 2]

        return multi_order_laplacian

    @staticmethod
    def get_laplacian(graph, normalize):
        """
        return the laplacian of the graph.
        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:

            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L






class cross_att(nn.Module):
    def __init__(self, d_time, d_joint, d_coor, isTrainning=False, head=4):
        super().__init__()

        self.qkv = nn.Linear(d_coor, d_coor * 3)
        self.head = head
        self.layer_norm = nn.LayerNorm(d_coor)
        # self.lpm_st_1 = LearnedPosMap2D(d_time, d_joint, gamma=4)
        self.scale = d_coor ** -0.5
        self.proj = nn.Linear(d_coor, d_coor)
        self.d_time = d_time
        self.d_joint = d_joint
        self.head = head

        # self.gate_s = nn.Conv2d(d_coor//2, d_coor//2, kernel_size=3, stride=1, padding=1,groups=d_coor//2)
        # self.gate_t = nn.Conv2d(d_coor//2, d_coor//2, kernel_size=3, stride=1, padding=1,groups=d_coor//2)

#         self.gate_s = MSLSP(d_time, d_joint, d_coor // 2)
        self.gate_t = nn.Conv2d(d_coor//2, d_coor//2, kernel_size=3, stride=1, padding=1,groups=d_coor//2)
        self.gate_s = nn.Conv2d(d_coor//2, d_coor//2, kernel_size=3, stride=1, padding=1,groups=d_coor//2)
        
        # self.gate_gs = ChebConv(d_coor//2, d_coor//2, K=2)
        #self.scf     = nn.Parameter(0.0001*torch.Tensor(1,1,d_coor//8))
        
        #self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c, out_c))  # [K+1, 1, in_c, out_c]
        #init.xavier_normal_(self.scf)
        
        
        self.body_edges = torch.tensor([[0, 1], [1, 2], [2, 3],
                                        [0, 4], [4, 5], [5, 6],
                                        [0, 7], [7, 8], [8, 9], [9, 10],
                                        [8, 11], [11, 12], [12, 13],
                                        [8, 14], [14, 15], [15, 16]], dtype=torch.long)
                                             #   [0,17],[1,17],[2,17],[3,17],[4,17],[5,17],[6,17],[7,17],[8,17],[9,17],
                                        #[10,17],[11,17],[12,17],[13,17],[14,17],[15,17],[16,17]
        # self.conv_2 = nn.Conv2d(d_coor, d_coor, kernel_size=5, stride=1, padding=2,groups=d_coor)
        self.graph = adj_mx_from_edges(d_joint, self.body_edges).long().cuda()
        self.emb =  nn.Embedding(20, d_coor//8, padding_idx=0)
        self.part = torch.tensor([0,0,1,1,1,2,2,2,3,3,3,4,4,4,0,0,0]).long().cuda()
        
        # self.gate_t = MSLSP(d_time, d_joint, d_coor//2)


        # self.lpm_s = LearnedPosMap2D(d_time,d_joint)
        # self.lpm_t = LearnedPosMap2D(d_time,d_joint)

        self.drop = DropPath(0.5)

    def forward(self, input):
        b, t, s, c = input.shape
#         print(self.scf)
#         exit()
        # input = input + self.lpm_st_1(input)
        h = input
        # print(input.shape)
        # exit()
        x = self.layer_norm(input)
        qkv = self.qkv(x)

        qkv = qkv.reshape(b, t, s, c, 3).permute(4, 0, 1, 2, 3)  # b,t,s,c
        # print(qkv.shape)

        qkv_s, qkv_t = qkv.chunk(2, 4)
        # print(qkv_s.shape,qkv_t.shape)

        q_s, k_s, v_s = qkv_s[0], qkv_s[1], qkv_s[2]  # b,t,s,c
        q_t, k_t, v_t = qkv_t[0], qkv_t[1], qkv_t[2]  # b,t,s,c

        # print(q_s.shape,q_t.shape)

        q_s = rearrange(q_s, 'b t s (h c) -> (b h t) s c', h=self.head)
        k_s = rearrange(k_s, 'b t s (h c) -> (b h t) c s ', h=self.head)

        q_t = rearrange(q_t, 'b  t s (h c) -> (b h s) t c', h=self.head)
        k_t = rearrange(k_t, 'b  t s (h c) -> (b h s) c t ', h=self.head)

        att_s = (q_s @ k_s) * self.scale  # b*h,s,s
        att_t = (q_t @ k_t) * self.scale  # b*h,s,s

        att_s = att_s.softmax(-1)
        att_t = att_t.softmax(-1)

        v_s = rearrange(v_s, 'b  t s c -> b c t s ')
        v_t = rearrange(v_t, 'b  t s c -> b c t s ')


#         
#         print(v_s.shape,self.graph.shape)
        lep_s = self.gate_s(v_s)
        lep_t = self.gate_t(v_t)
        v_s = rearrange(v_s, 'b c t s -> (b t ) s c')
        # sep_s = self.gate_gs(v_s,self.graph)
        sep_s = self.emb(self.part).unsqueeze(0)
        # print(sep_s.shape)
        
        # sep_s = rearrange(sep_s, '(b t) s (h c)   -> (b h t) s c ', t=t,h=self.head)

        lep_s = rearrange(lep_s, 'b (h c) t s  -> (b h t) s c ', h=self.head)
        lep_t = rearrange(lep_t, 'b (h c) t s  -> (b h s) t c ', h=self.head)
    
    
        v_s = rearrange(v_s, '(b t) s (h c)   -> (b h t) s c ', t=t,h=self.head)
#         v_s = rearrange(v_s, 'b (h c) t s  -> (b h t) s c ', h=self.head)
        v_t = rearrange(v_t, 'b (h c) t s  -> (b h s) t c ', h=self.head)
        #print(lep_s[55,:,:])
        #print(sep_s[55,:,:])
        #print(self.scf)
        #print(self.scf*sep_s[55,:,:])
        #exit()

        # v = torch.cat((v1, v2), -1)

        x_s = att_s @ v_s + lep_s + 0.0001*self.drop(sep_s)  # b*h,s,c//h
        x_t = att_t @ v_t + lep_t  # b*h,t,c//h
        # print(x_s.shape,x_t.shape)

        x_s = rearrange(x_s, '(b h t) s c -> b h t s c ', h=self.head, t=t)
        x_t = rearrange(x_t, '(b h s) t c -> b h t s c ', h=self.head, s=s)
        # print(x_s.shape,x_t.shape)
        x = torch.cat((x_s, x_t), -1)
        x = rearrange(x, 'b h t s c -> b  t s (h c) ')

        x = self.proj(x)
        # print(x.shape,h.shape)
        x = x + h
        return x


class MLP_3D(nn.Module):
    def __init__(self, d_time, d_joint, d_coor, isTrainning=False, ):
        super().__init__()

        self.d_time = d_time
        self.d_joint = d_joint
        self.d_coor = d_coor

        self.layer_norm1 = nn.LayerNorm(self.d_coor)
        self.layer_norm2 = nn.LayerNorm(self.d_coor)

        self.mlp1 = Mlp(self.d_coor, self.d_coor * 4, self.d_coor)

        self.cross_att = cross_att(d_time, d_joint, d_coor, isTrainning)
        self.drop = DropPath(0.0)

    def forward(self, input):
        b, t, s, c = input.shape

        x = self.cross_att(input)

        x = x + self.drop(self.mlp1(self.layer_norm1(x)))

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mlp_C(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.sig = nn.Sigmoid()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        b, t, s, c = x.shape
        # gate = self.avg(x.permute(0,3,1,2)).permute(0,2,3,1)
        gate = self.fc1(x)
        gate = self.act(gate)
        gate = self.drop(gate)
        gate = self.fc2(gate)
        gate = self.sig(gate)
        # gate = gate.expand(b,t,s,c)
        x = x * gate
        return x


class MlpMixer(nn.Module):
    def __init__(self, num_block, d_time, d_joint, d_coor, isTrainning=False, ):
        super(MlpMixer, self).__init__()

        self.num_block = num_block
        self.d_time = d_time
        self.d_joint = d_joint
        self.d_coor = d_coor

        self.mixerblocks = []
        for l in range(self.num_block):
            self.mixerblocks.append(MLP_3D(self.d_time, self.d_joint, self.d_coor, isTrainning))
        self.mixerblocks = nn.ModuleList(self.mixerblocks)

    def forward(self, input):
        # blocks layers
        for i in range(self.num_block):
            input = self.mixerblocks[i](input)
        # exit()

        return input


if __name__ == "__main__":
    inputs = torch.rand(64, 351, 34)  # [btz, channel, T, H, W]
    # inputs = torch.rand(1, 64, 4, 112, 112) #[btz, channel, T, H, W]
    net = Model()
    output = net(inputs)
    print(output.size())
    from thop import profile

    flops, params = profile(net, inputs=(inputs,))
    print(flops)
    print(params)
    """
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name,':',param.size())
    """
