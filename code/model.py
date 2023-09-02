
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义KCDN模型
class KCDN(nn.Module):
    # 初始化模型
    def __init__(self, args, n_entity, n_relation):
        super(KCDN, self).__init__()
        self._parse_args(args, n_entity, n_relation)
        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim)
        # 定义注意力机制
        self.attention = nn.Sequential(
                nn.Linear(self.dim*2, self.dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.dim, self.dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.dim, 1, bias=False),
                nn.Sigmoid(),
                )
        self._init_weight()
                
    # 前向传播函数
    def forward(
        self,
        items: torch.LongTensor,
        user_triple_set: list,
        item_triple_set: list,
    ):

        item_embeddings = []

        # 获取初始的物品嵌入
        item_emb_origin = self.entity_emb(items)
        item_embeddings.append(item_emb_origin)

        for i in range(self.n_layer):
            # 获取实体、关系和目标的嵌入
            h_emb = self.entity_emb(item_triple_set[0][i])
            r_emb = self.relation_emb(item_triple_set[1][i])
            t_emb = self.entity_emb(item_triple_set[2][i])
            # 根据注意力机制计算物品嵌入
            item_emb_i = self._sim_attention(h_emb, r_emb, t_emb, item_emb_origin)
            item_embeddings.append(item_emb_i)

        user_embeddings = []

        # 获取初始的用户嵌入
        user_emb_0 = self.entity_emb(user_triple_set[0][0])
        user_embeddings.append(user_emb_0.mean(dim=1))
        
        for i in range(self.n_layer):
            # 获取实体、关系和目标的嵌入
            h_emb = self.entity_emb(user_triple_set[0][i])
            r_emb = self.relation_emb(user_triple_set[1][i])
            t_emb = self.entity_emb(user_triple_set[2][i])
            # 根据注意力机制计算用户嵌入
            user_emb_i = self._sim_attention(h_emb, r_emb, t_emb, user_emb_0)
            user_embeddings.append(user_emb_i)

        # 使用聚合器聚合用户和物品嵌入
        e_u = user_embeddings[0]
        e_v = item_embeddings[0]

        if self.agg == 'concat':
            if len(user_embeddings) != len(item_embeddings):
                raise Exception("Concat aggregator needs same length for user and item embedding")
            for i in range(1, len(user_embeddings)):
                e_u = torch.cat((user_embeddings[i], e_u),dim=-1)
            for i in range(1, len(item_embeddings)):
                e_v = torch.cat((item_embeddings[i], e_v),dim=-1)
        elif self.agg == 'sum':
            for i in range(1, len(user_embeddings)):
                e_u = e_u + user_embeddings[i]
            for i in range(1, len(item_embeddings)):
                e_v = e_v + item_embeddings[i]
        elif self.agg == 'pool':
            for i in range(1, len(user_embeddings)):
                e_u = torch.max(e_u, user_embeddings[i])
            for i in range(1, len(item_embeddings)):
                e_v = torch.max(e_v, item_embeddings[i])
        else:
            raise Exception("Unknown aggregator: " + self.agg)
            
        # 计算得分
        scores = (e_v * e_u).sum(dim=1)
        scores = torch.sigmoid(scores)
        return scores
    
    # 解析参数
    def _parse_args(self, args, n_entity, n_relation):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_layer = args.n_layer
        self.agg = args.agg
        
    # 初始化权重
    def _init_weight(self):
        # 初始化嵌入权重
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        # 初始化注意力机制权重
        for layer in self.attention:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    # 注意力机制
    def _sim_attention(self, h_emb, r_emb, t_emb, i_emb):
        i_emb = i_emb.unsqueeze(dim=1)
        sim_weights = (h_emb * i_emb).sum(dim=-1)
        sim_weights = F.softmax(sim_weights, dim=-1)
        emb_i = torch.mul(sim_weights.unsqueeze(-1), t_emb)
        emb_i = emb_i.sum(dim=1)
        return emb_i
