from socket import gaierror
import numpy as np
import torch as th
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from dgl.nn.pytorch import GATv2Conv

def drop_node(feats, drop_rate, training):
    
    n = feats.shape[0]
    drop_rates = th.FloatTensor(np.ones(n) * drop_rate)
    
    if training:
            
        masks = th.bernoulli(1. - drop_rates).unsqueeze(1)
        feats = masks.to(feats.device) * feats
        
    else:
        feats = feats * (1. - drop_rate)

    return feats

class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, input_droprate, hidden_droprate, use_bn =False):
        super(MLP, self).__init__()
        
        self.layer1 = nn.Linear(nfeat, nhid, bias = True)
        self.layer2 = nn.Linear(nhid, nclass, bias = True)

        self.input_dropout = nn.Dropout(input_droprate)
        self.hidden_dropout = nn.Dropout(hidden_droprate)
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
    
    def reset_parameters(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        
    def forward(self, x):
         
        if self.use_bn: 
            x = self.bn1(x)
        x = self.input_dropout(x)
        x = F.relu(self.layer1(x))
        
        if self.use_bn:
            x = self.bn2(x)
        x = self.hidden_dropout(x)
        x = self.layer2(x)

        return x   
        

class GRANDConv(nn.Module):
    '''
    Parameters
    -----------
    graph: dgl.Graph
        The input graph
    feats: Tensor (n_nodes * feat_dim)
        Node features
    order: int 
        Propagation Steps
    '''
    def __init__(self, in_dim) -> None:
        super().__init__()
        self.atten_fc = nn.Linear(2 * in_dim, 1, bias=False)
        self.dropout = nn.Dropout(0)
        self.fc = nn.Linear(in_dim, in_dim, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
    
    def edge_attention(self, edges):
        z = th.cat([edges.src['z'] * edges.src['norm'], edges.dst['z'] * edges.dst['norm']], dim=1)
        a = self.atten_fc(z)
        return {'e': self.activation(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        alpha = self.dropout(alpha)
        h = th.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, graph, feats, order):
        with graph.local_scope():
            
            ''' Calculate Symmetric normalized adjacency matrix   \hat{A} '''
            degs = graph.in_degrees().float().clamp(min=1)
            norm = th.pow(degs, -0.5).to(feats.device).unsqueeze(1)

            graph.ndata['norm'] = norm
            graph.apply_edges(fn.u_mul_v('norm', 'norm', 'weight'))
            
            ''' Graph Conv '''
            x = feats
            y = 0+feats

            for i in range(order):
                graph.ndata['z'] = x
                graph.apply_edges(self.edge_attention)
                # graph.update_all(fn.u_mul_e('h', 'e', 'm'), fn.sum('m', 'h'))
                graph.update_all(self.message_func, self.reduce_func)
                x = graph.ndata.pop('h')
                y.add_(x)

        return y /(order + 1)

class GRAND(nn.Module):
    r"""

    Parameters
    -----------
    in_dim: int
        Input feature size. i.e, the number of dimensions of: math: `H^{(i)}`.
    hid_dim: int
        Hidden feature size.
    n_class: int
        Number of classes.
    S: int
        Number of Augmentation samples
    K: int
        Number of Propagation Steps
    node_dropout: float
        Dropout rate on node features.
    input_dropout: float
        Dropout rate of the input layer of a MLP
    hidden_dropout: float
        Dropout rate of the hidden layer of a MLP
    batchnorm: bool, optional
        If True, use batch normalization.

    """
    def __init__(self,
                 in_dim,
                 hid_dim,
                 n_class,
                 S = 1,
                 K = 3,
                 node_dropout=0.0,
                 input_droprate = 0.0, 
                 hidden_droprate = 0.0,
                 batchnorm=False,
                 is_ori=True):

        super(GRAND, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.S = S
        self.K = K
        self.n_class = n_class
        self.is_ori = is_ori
        
        self.mlp = MLP(in_dim, hid_dim, n_class, input_droprate, hidden_droprate, batchnorm)
        self.gat = GAT(2, in_dim, hid_dim, n_class, 8, feat_drop=input_droprate, attn_drop=hidden_droprate, neg_slope=0.2)
        self.mygat = MYGAT(1, in_dim, in_dim, in_dim, 1, feat_drop=0, attn_drop=0, neg_slope=0.2, is_v2=False)
        self.grand = GRANDConv(in_dim)
        
        self.dropout = node_dropout
        self.node_dropout = nn.Dropout(node_dropout)

    def forward(self, graph, feats, training = True):
        
        X = feats
        S = self.S
        if training: # Training Mode
            output_list = []
            for s in range(S):
                drop_feat = drop_node(X, self.dropout, True)  # Drop node
                feat = self.grand(graph, drop_feat, self.K)    # Graph Convolution
                # feat = self.mygat(graph, feat)
                if self.is_ori:
                    output_list.append(th.log_softmax(self.mlp(feat), dim=-1))  # Prediction
                # output_list.append(th.log_softmax(self.mygat(graph, self.mlp(feat)), dim=-1))  # Prediction
                # output_list.append(th.log_softmax(self.mlp(self.gat(graph, feat)), dim=-1))  # Prediction
                # output_list.append(th.log_softmax(self.gat(graph, self.mlp(feat)), dim=-1))  # Prediction
                else:
                    output_list.append(th.log_softmax(self.gat(graph, feat), dim=-1))  # Prediction
        
            return output_list
        else:   # Inference Mode
            drop_feat = drop_node(X, self.dropout, False)
            X =  self.grand(graph, drop_feat, self.K)
            # X = self.mygat(graph, X)
            
            if self.is_ori:
                return th.log_softmax(self.mlp(X), dim = -1)
            # return th.log_softmax(self.mygat(graph, self.mlp(X)), dim = -1)
            # return th.log_softmax(self.mlp(self.gat(graph, X)), dim = -1)
            # return th.log_softmax(self.gat(graph, self.mlp(X)), dim = -1)
            else:
                return th.log_softmax(self.gat(graph, X), dim = -1)


class GAT(nn.Module):
    def __init__(self, num_layer, in_dim, hid_dim, out_dim, num_head, feat_drop=0, attn_drop=0, neg_slope=0.2, activation=F.elu, share=True) -> None:
        super().__init__()
        self.num_layer = num_layer
        self.layers = nn.ModuleList()
        if num_layer > 1:
            self.layers.append(GATv2Conv(in_dim, hid_dim, num_head, feat_drop, attn_drop, neg_slope, activation=activation, share_weights=share))
            for i in range(num_layer - 2):
                self.layers.append(GATv2Conv(hid_dim * num_head, hid_dim, num_head, feat_drop, attn_drop, neg_slope, activation=activation, share_weights=share))
            self.layers.append(GATv2Conv(hid_dim * num_head, out_dim, 1, feat_drop, attn_drop, neg_slope, activation=None, share_weights=share))
        else:
            self.layers.append(GATv2Conv(in_dim, out_dim, num_head, feat_drop, attn_drop, neg_slope, activation=None, share_weights=share))

    def forward(self, g, h):
        for i in range(self.num_layer):
            h = self.layers[i](g, h)
            h = h.flatten(1) if i != self.num_layer - 1 else h.mean(1)

        return h


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, attn_drop=0, neg_slope=0.2):
        super().__init__()
        self.dropout = nn.Dropout(attn_drop)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.atten_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=neg_slope)
    
    def edge_attention(self, edges):
        z = th.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.atten_fc(z)
        return {'e': self.activation(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        alpha = self.dropout(alpha)
        h = th.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        # z = self.fc(h)
        z = h
        g.ndata['z'] = z

        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)

        return g.ndata.pop('h')


class GATLayerv2(nn.Module):
    def __init__(self, in_dim, out_dim, attn_drop=0, neg_slope=0.1):
        super().__init__()
        self.dropout = nn.Dropout(attn_drop)
        self.fc_left = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_right = nn.Linear(in_dim, out_dim, bias=False)
        self.atten_fc = nn.Linear(out_dim, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=neg_slope)
    
    def edge_attention(self, edges):
        s = self.activation(edges.src['el'] + edges.dst['er'])
        e = self.atten_fc(s)

        return {'e': e}

    def message_func(self, edges):
        return {'el': edges.src['el'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        alpha = self.dropout(alpha)
        h = th.sum(alpha * nodes.mailbox['el'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        # el = self.fc_left(h)
        # er = self.fc_right(h)
        g.ndata['el'] = h
        g.ndata['er'] = h

        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)

        return g.ndata.pop('h')



class MutiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_head, attn_drop=0, neg_slope=0.1, merge='cat', is_v2=False) -> None:
        super().__init__()
        self.heads = nn.ModuleList()
        self.merge = merge
        for i in range(num_head):
            if is_v2 == True:
                self.heads.append(GATLayerv2(in_dim, out_dim, attn_drop, neg_slope))
            else:
                self.heads.append(GATLayer(in_dim, out_dim, attn_drop, neg_slope))
        
    def forward(self, g, h):
        out_heads = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == 'cat':
            return th.cat(out_heads, dim=1)
        else:
            return th.mean(th.stack(out_heads), dim=0)


class MYGAT(nn.Module):
    def __init__(self, num_layer, in_dim, hid_dim, out_dim, num_head, feat_drop=0, attn_drop=0, neg_slope=0.2, activation=F.relu, is_v2=True) -> None:
        super().__init__()
        self.num_layer = num_layer
        self.activation = activation
        self.layers = nn.ModuleList()
        if num_layer > 1:
            self.layers.append(MutiHeadGATLayer(in_dim , hid_dim, num_head, attn_drop, neg_slope, is_v2))
            for i in range(num_layer - 2):
                self.layers.append(MutiHeadGATLayer(hid_dim * num_head, hid_dim, num_head, attn_drop, neg_slope, is_v2))
            self.layers.append(MutiHeadGATLayer(hid_dim * num_head, out_dim, 1, attn_drop, neg_slope, is_v2))
        else:
            self.layers.append(MutiHeadGATLayer(in_dim, out_dim, num_head, attn_drop, neg_slope, is_v2))

        self.dropout = nn.Dropout(feat_drop)

    def forward(self, g, h):
        for i in range(self.num_layer - 1):
            h = self.dropout(h)
            h = self.layers[i](g, h)
            h = self.activation(h)

        h = self.dropout(h)
        h = self.layers[-1](g, h)
        return h
