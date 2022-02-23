import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import MultiLayerFullNeighborSampler
from dgl.utils import expand_as_pair
import dgl.function as fn
class overAll(nn.Module):
    def __init__(self, node_size, node_hidden,
                 rel_size,
                 g,
                 g_r,
                 n_r,
                 dropout_rate=0., depth=2,
                 device='cpu'
                 ):
        super(overAll, self).__init__()
        self.encoder = MRAEA(depth=depth, device=device)
        self.device = device
        self.dropout_rate = dropout_rate
        self.depth = depth
        self.ent_emb = self.init_emb(node_size, node_hidden, init_func='uniform')
        self.rel_emb = self.init_emb(rel_size, node_hidden, init_func='uniform')

        # new adding
        self.g = g
        self.g_r = g_r
        self.n_r = n_r

    @staticmethod
    def init_emb(*size, init_func='xavier'):
        # TODO BIAS
        entities_emb = nn.Parameter(torch.randn(size))
        if init_func == 'xavier':
            torch.nn.init.xavier_normal_(entities_emb)
        elif init_func == 'zero':
            torch.nn.init.zeros_(entities_emb)
        elif init_func == 'uniform':
            torch.nn.init.uniform_(entities_emb, -.05, .05)
        else:
            raise NotImplementedError
        return entities_emb

    def forward(self, blocks):
        src_nodes = blocks[0].srcdata[dgl.NID]
        dst_nodes = blocks[-1].dstdata[dgl.NID]
        # torch.equal(src_nodes[:len(dst_nodes)], dst_nodes
        g = self.g
        # g.ndata['ent_emb'] = self.ent_emb
        temp_block = MultiLayerFullNeighborSampler(1).sample_blocks(g, src_nodes)[0]
        temp_block = temp_block.to(self.device)
        temp_block.srcdata['ent_emb'] = self.ent_emb[temp_block.srcdata[dgl.NID]]
        temp_block.update_all(fn.copy_u('ent_emb', 'm'), fn.mean('m', 'feature'))
        ent_feature = temp_block.dstdata['feature']

        n_r = self.n_r
        n_r.nodes['relation'].data['rel_emb'] = self.rel_emb
        r_temp_block = MultiLayerFullNeighborSampler(1).sample_blocks(n_r, {'entity': src_nodes.type(torch.int32)})[0]
        r_temp_block.update_all(fn.copy_u('rel_emb', 'm'), fn.mean('m', 'r_neigh'), etype='link')
        rel_feature = r_temp_block.dstnodes['entity'].data['r_neigh']

        feature = torch.cat([ent_feature, rel_feature], dim=1)
        out_feature = self.encoder(blocks, self.g_r, feature, self.rel_emb)
        # out_feature_rel = self.r_encoder(blocks, self.g_r, rel_feature, self.rel_emb)
        # out_feature = torch.cat((out_feature_ent, out_feature_rel), dim=-1)

        out_feature = F.dropout(out_feature, p=self.dropout_rate, training=self.training)
        return out_feature

class MRAEA(nn.Module):
    def __init__(self,
                 use_bias=False,
                 depth=1,
                 activation=F.relu,
                 device='cpu',
                 attn_heads_reduction='mean',
                 attn_heads=2
                 ):
        super(MRAEA, self).__init__()

        self.activation = activation
        self.depth = depth
        self.use_bias = use_bias
        self.attn_kernels = []
        self.biases = []
        
        # adding
        self.attn_heads_reduction = attn_heads_reduction
        self.attn_heads = attn_heads

        dim = 100

        node_F = dim
        rel_F = dim
        self.ent_F = node_F + rel_F
        ent_F = self.ent_F

        # different from RREA & DUAL
        # using the same kernel for each layer(hop)
        for head in range(self.attn_heads):
            # todo attn_kernel seems different
            if self.use_bias:
                bias = overAll.init_emb(ent_F, 1, init_func='xavier')
                self.biases.append(bias)
            attn_kernel_self = overAll.init_emb(ent_F, 1)
            attn_kernel_neighs = overAll.init_emb(ent_F, 1)
            attn_kernel_rels = overAll.init_emb(rel_F, 1)
            attn_kernel = [attn_kernel_self, attn_kernel_neighs, attn_kernel_rels]
            self.attn_kernels.append([x.to(device) for x in attn_kernel])


    def forward(self, blocks, g_r: dgl.heterograph, features, rel_emb):
        outputs = []
        features = self.activation(features)
        # append dst feature
        dst_nodes = blocks[-1].dstdata[dgl.NID]
        outputs.append(features[:len(dst_nodes)])
        for l in range(self.depth):
            features_list = []
            for head in range(self.attn_heads):
                attention_kernel = self.attn_kernels[head]
                # compute attention for rel

                eid = blocks[l].edata[dgl.EID]
                g_r.nodes['relation'].data['features'] = torch.matmul(rel_emb, attention_kernel[2])
                temp_block = MultiLayerFullNeighborSampler(1).sample_blocks(g_r, {'index': eid})[0]
                temp_block.update_all(fn.copy_u('features', 'm'), fn.mean('m', 'att'), etype='in')
                attn_for_rels = temp_block.dstnodes['index'].data['att']


                # blocks[l].srcdata['features'] = features
                # blocks[l].apply_edges(fn.copy_u('features', 'neighs'))
                # neighs = blocks[l].edata['neighs']
    
                # add self
                src, trg = blocks[l].edges()
                selfs = features[trg]
                neighs = features[src]


                attn_for_neighs = torch.matmul(neighs, attention_kernel[1])

                attn_for_selfs = torch.matmul(selfs, attention_kernel[0])
                att = attn_for_rels + attn_for_selfs + attn_for_neighs
                att = F.leaky_relu(att)


                from dgl.nn.functional import edge_softmax
                att = edge_softmax(blocks[l], att, norm_by='dst')
                new_feature = neighs * att
                blocks[l].edata['feat'] = new_feature
                blocks[l].update_all(fn.copy_e('feat', 'm'), fn.sum('m', 'layer'+str(l)))
                dst_features = blocks[l].dstdata['layer' + str(l)]
    


                features_list.append(dst_features)
            if self.attn_heads_reduction == 'concat':
                features = torch.cat(features_list, dim=1)
            else:
                features = torch.mean(torch.stack(features_list), dim=0)
            features = self.activation(features)
            dst_nodes = blocks[-1].dstdata[dgl.NID]
            outputs.append(features[:len(dst_nodes)])
        
        # deal with layer
        outputs = torch.cat(outputs, dim=1)
        return outputs
