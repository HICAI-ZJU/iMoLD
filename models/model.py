import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_add, scatter_mean
import numpy as np

from GOOD.networks.models.GINs import GINFeatExtractor
from GOOD.networks.models.GINvirtualnode import vGINFeatExtractor

from vector_quantize_pytorch import VectorQuantize
# from .vq_update import VectorQuantize

from .gnnconv import GNN_node


class Separator(nn.Module):
    def __init__(self, args, config):
        super(Separator, self).__init__()
        if args.dataset.startswith('GOOD'):
            # GOOD
            if config.model.model_name == 'GIN':
                self.r_gnn = GINFeatExtractor(config, without_readout=True)
            else:
                self.r_gnn = vGINFeatExtractor(config, without_readout=True)
            emb_d = config.model.dim_hidden
        else:
            self.r_gnn = GNN_node(num_layer=args.layer, emb_dim=args.emb_dim,
                                  drop_ratio=args.dropout, gnn_type=args.gnn_type)
            emb_d = args.emb_dim

        self.separator = nn.Sequential(nn.Linear(emb_d, emb_d * 2),
                                       nn.BatchNorm1d(emb_d * 2),
                                       nn.ReLU(),
                                       nn.Linear(emb_d * 2, emb_d),
                                       nn.Sigmoid())
        self.args = args

    def forward(self, data):
        if self.args.dataset.startswith('GOOD'):
            # DrugOOD
            node_feat = self.r_gnn(data=data)
        else:
            # GOOD
            node_feat = self.r_gnn(data)
        score = self.separator(node_feat)  # [n, d]

        # reg on score

        pos_score_on_node = score.mean(1)  # [n]
        pos_score_on_batch = scatter_add(pos_score_on_node, data.batch, dim=0)  # [B]
        neg_score_on_batch = scatter_add((1 - pos_score_on_node), data.batch, dim=0)  # [B]
        return score, pos_score_on_batch + 1e-8, neg_score_on_batch + 1e-8


class DiscreteEncoder(nn.Module):
    def __init__(self, args, config):
        super(DiscreteEncoder, self).__init__()
        self.args = args
        self.config = config
        if args.dataset.startswith('GOOD'):
            emb_dim = config.model.dim_hidden
            if config.model.model_name == 'GIN':
                self.gnn = GINFeatExtractor(config, without_readout=True)
            else:
                self.gnn = vGINFeatExtractor(config, without_readout=True)
            self.classifier = nn.Sequential(*(
                [nn.Linear(emb_dim, config.dataset.num_classes)]
            ))
        else:
            emb_dim = args.emb_dim
            self.gnn = GNN_node(num_layer=args.layer, emb_dim=args.emb_dim,
                                drop_ratio=args.dropout, gnn_type=args.gnn_type)
            self.classifier = nn.Sequential(nn.Linear(emb_dim, emb_dim * 2),
                                            nn.BatchNorm1d(emb_dim * 2),
                                            nn.ReLU(),
                                            nn.Dropout(),
                                            nn.Linear(emb_dim * 2, 1))

        self.pool = global_mean_pool

        self.vq = VectorQuantize(dim=emb_dim,
                                 codebook_size=args.num_e,
                                 commitment_weight=args.commitment_weight,
                                 decay=0.9)

        self.mix_proj = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim),
                                      nn.BatchNorm1d(emb_dim),
                                      nn.ReLU(),
                                      nn.Dropout(),
                                      nn.Linear(emb_dim, emb_dim))

        self.simsiam_proj = nn.Sequential(nn.Linear(emb_dim, emb_dim * 2),
                                          nn.BatchNorm1d(emb_dim * 2),
                                          nn.ReLU(),
                                          nn.Linear(emb_dim * 2, emb_dim))

    def vector_quantize(self, f, vq_model):
        v_f, indices, v_loss = vq_model(f)

        return v_f, v_loss

    def forward(self, data, score):
        if self.args.dataset.startswith('GOOD'):
            # DrugOOD
            node_feat = self.gnn(data=data)
        else:
            # GOOD
            node_feat = self.gnn(data)

        node_v_feat, cmt_loss = self.vector_quantize(node_feat.unsqueeze(0), self.vq)
        node_v_feat = node_v_feat.squeeze(0)
        node_res_feat = node_feat + node_v_feat
        c_node_feat = node_res_feat * score
        s_node_feat = node_res_feat * (1 - score)

        c_graph_feat = self.pool(c_node_feat, data.batch)
        s_graph_feat = self.pool(s_node_feat, data.batch)

        c_logit = self.classifier(c_graph_feat)

        return c_logit, c_graph_feat, s_graph_feat, cmt_loss


class MyModel(nn.Module):
    def __init__(self, args, config):
        super(MyModel, self).__init__()
        self.args = args
        self.config = config

        self.separator = Separator(args, config)
        self.encoder = DiscreteEncoder(args, config)

    def forward(self, data):
        score, pos_score, neg_score = self.separator(data)
        c_logit, c_graph_feat, s_graph_feat, cmt_loss = self.encoder(data, score)
        # reg on score
        loss_reg = torch.abs(pos_score / (pos_score + neg_score) - self.args.gamma * torch.ones_like(pos_score)).mean()
        return c_logit, c_graph_feat, s_graph_feat, cmt_loss, loss_reg

    def mix_cs_proj(self, c_f: torch.Tensor, s_f: torch.Tensor):
        n = c_f.size(0)
        perm = np.random.permutation(n)
        mix_f = torch.cat([c_f, s_f[perm]], dim=-1)
        proj_mix_f = self.encoder.mix_proj(mix_f)
        return proj_mix_f

