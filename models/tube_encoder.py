from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptTensor
from torch_geometric.typing import Size
from torch_geometric.utils import softmax
from torch_geometric.utils import subgraph

from models import MultipleInputEmbedding
from models import SingleInputEmbedding
from utils import DistanceDropEdge
from utils import TemporalData
from utils import init_weights
from models.local_encoder import ALEncoder2,ALEncoder
from models import TemporalEncoder
from models import AAEncoder
from models.local_encoder import AttentionLayer
from torch_cluster import radius_graph
from utils import merge_edges
from models.fourier_embedding import FourierEmbedding

class MapEncoder(nn.Module): 
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int=8,
                 head_dim: int=16,
                 dropout: float=0.1,
                 num_layers: int=1,
                 num_freq_bands:int=64,
                 local_radius: float=50): 
        super(MapEncoder, self).__init__()
        self.local_radius= local_radius
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.int_pl_emb = nn.Embedding(2, embed_dim)
        self.turn_pl_emb = nn.Embedding(3, embed_dim)
        self.traffic_pl_emb = nn.Embedding(2, embed_dim)
        self.pl2pl_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=embed_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.x_pl_emb = FourierEmbedding(input_dim=2, hidden_dim=embed_dim, num_freq_bands=num_freq_bands)

        self.r_pl2pl_emb = FourierEmbedding(input_dim=3, hidden_dim=embed_dim,
                                            num_freq_bands=num_freq_bands)
        if 0: 
            self.type_pl2pl_emb =  nn.Embedding(5, embed_dim)
        self.apply(init_weights)

    def forward(self, data: TemporalData) -> torch.Tensor:
        pos_pl = data['lane_vectors'][:, :2].contiguous()
        #if data['rotate_mat'] is not None: 
        #    pos_pl = torch.bmm(pos_pl.unsqueeze(0), data['rotate_mat'][0,:,:].unsqueeze(0)).squeeze(0)
        x_pl_categorical_embs = [self.int_pl_emb(data['is_intersections'].long()),
                                 self.turn_pl_emb(data['turn_directions'].long()),
                                 self.traffic_pl_emb(data['traffic_controls'].long())]
        x_pl = self.x_pl_emb(continuous_inputs=pos_pl, categorical_embs=x_pl_categorical_embs) #? pos needed
        
        lane_batch = torch.cumsum((data['lane_ids']==0).long(), dim=0)-1
        edge_index_pl2pl_radius = radius_graph(x=pos_pl[:, :2], r=self.local_radius,
                                               batch=lane_batch,
                                               loop=False, max_num_neighbors=200) #50
        if 0: 
            edge_index_pl2pl = data['lane_lane_edge_index']
            rel_pos_pl2pl = pos_pl[edge_index_pl2pl[0]] - pos_pl[edge_index_pl2pl[1]]        
            type_pl2pl = data['lane_lane_edge_type']
            type_pl2pl_radius = type_pl2pl.new_zeros(edge_index_pl2pl_radius.size(1), dtype=torch.uint8)
            edge_index_pl2pl, type_pl2pl = merge_edges(edge_indices=[edge_index_pl2pl_radius, edge_index_pl2pl],
                                                    edge_attrs=[type_pl2pl_radius, type_pl2pl], reduce='max')
            rel_pos_pl2pl = pos_pl[edge_index_pl2pl[0]] - pos_pl[edge_index_pl2pl[1]]
            r_pl2pl = torch.stack([rel_pos_pl2pl[:,0],rel_pos_pl2pl[:,1],
                torch.norm(rel_pos_pl2pl[:, :2], p=2, dim=-1)], dim=-1)
            r_pl2pl = self.r_pl2pl_emb(continuous_inputs=r_pl2pl, categorical_embs=[self.type_pl2pl_emb(type_pl2pl.long())])
            for i in range(self.num_layers):
                x_pl = self.pl2pl_layers[i](x_pl, r_pl2pl, edge_index_pl2pl)
        else: 
            rel_pos_pl2pl = pos_pl[edge_index_pl2pl_radius[0]] - pos_pl[edge_index_pl2pl_radius[1]]  
            r_pl2pl = torch.stack([rel_pos_pl2pl[:,0],rel_pos_pl2pl[:,1],
                 torch.norm(rel_pos_pl2pl[:, :2], p=2, dim=-1)], dim=-1)
            r_pl2pl = self.r_pl2pl_emb(continuous_inputs=r_pl2pl, categorical_embs=None)
            for i in range(self.num_layers):
                x_pl = self.pl2pl_layers[i](x_pl, r_pl2pl, edge_index_pl2pl_radius)

        

        return x_pl 
        
 
class PatchEncoder(nn.Module):
    def __init__(self,
                 historical_steps: int,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 patch_size: int, 
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 num_temporal_layers: int = 4,
                 local_radius: float = 50,
                 parallel: bool = False,
                 use_map_encoder: bool=False) -> None:
        super(PatchEncoder, self).__init__()
        self.historical_steps = historical_steps
        self.parallel = parallel
        self.patch_size = patch_size
        self.use_map_encoder = use_map_encoder
        self.drop_edge = DistanceDropEdge(local_radius)
        self.aa_encoder = AAEncoder(historical_steps=historical_steps,
                                    node_dim=node_dim,
                                    edge_dim=edge_dim,
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    dropout=dropout,
                                    parallel=parallel)
        if self.patch_size>1: 
            self.temporal_encoder = TemporalEncoder(historical_steps=patch_size, #historical_steps,
                                                embed_dim=embed_dim,
                                                num_heads=num_heads,
                                                dropout=dropout,
                                                num_layers=num_temporal_layers)
            if self.use_map_encoder: 
                self.al_encoder = ALEncoder2(node_dim=node_dim,
                                        edge_dim=edge_dim,
                                        embed_dim=embed_dim,
                                        num_heads=num_heads,
                                        dropout=dropout)
            else: 
                self.al_encoder = ALEncoder(node_dim=node_dim,
                                    edge_dim=edge_dim,
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    dropout=dropout)
            self.apply(init_weights)

    def forward(self, data: TemporalData) -> torch.Tensor:
        torch.cuda.empty_cache()
        edge_dict = dict()
        for t in range(self.historical_steps):
            edge_dict[f'edge_index_{t}'], _ = subgraph(subset=~data['padding_mask'][:, t], edge_index=data.edge_index)
            edge_dict[f'edge_attr_{t}'] = \
                    data['positions'][edge_dict[f'edge_index_{t}'][0], t] - data['positions'][edge_dict[f'edge_index_{t}'][1], t]
        if self.parallel:
            snapshots = [None] * self.historical_steps
            for t in range(self.historical_steps):
                edge_index, edge_attr = self.drop_edge(edge_dict[f'edge_index_{t}'], edge_dict[f'edge_attr_{t}'])
                snapshots[t] = Data(x=data.x[:, t], edge_index=edge_index, edge_attr=edge_attr,
                                        num_nodes=data.num_nodes)
            batch = Batch.from_data_list(snapshots)
            out = self.aa_encoder(x=batch.x, t=None, edge_index=batch.edge_index, edge_attr=batch.edge_attr,
                                    bos_mask=data['bos_mask'], rotate_mat=data['rotate_mat'])
            out = out.view(self.historical_steps, out.shape[0] // self.historical_steps, -1)
        else:
            out = [None] * self.historical_steps
            for t in range(self.historical_steps):
                edge_index, edge_attr = self.drop_edge(edge_dict[f'edge_index_{t}'], edge_dict[f'edge_attr_{t}'])
                out[t] = self.aa_encoder(x=data.x[:, t], t=t, edge_index=edge_index, edge_attr=edge_attr,
                                            bos_mask=data['bos_mask'][:, t], rotate_mat=data['rotate_mat'])
            out = torch.stack(out)  # [T, N, D]
        if self.patch_size >1: 
            out2 = [None]*(self.historical_steps//self.patch_size)
            edge_index, edge_attr = self.drop_edge(data['lane_actor_index'], data['lane_actor_vectors'])

            for t in range(0, self.historical_steps, self.patch_size): 
                out_patch = self.temporal_encoder(x=out[t:(t+self.patch_size)], 
                                                padding_mask=data['padding_mask'][:, t:(t+self.patch_size)])   
                if self.use_map_encoder:                   
                    out2[t//self.patch_size] = self.al_encoder(x=(data['map_enc'], out_patch), #x=(data['lane_vectors'], out_patch), 
                                                        edge_index=edge_index, edge_attr=edge_attr,
                                    #is_intersections=data['is_intersections'], turn_directions=data['turn_directions'],traffic_controls=data['traffic_controls'], 
                                    rotate_mat=data['rotate_mat'])
                else: 
                    out2[t//self.patch_size] = self.al_encoder(x=(data['lane_vectors'], out_patch), 
                                                        edge_index=edge_index, edge_attr=edge_attr,
                                    is_intersections=data['is_intersections'], turn_directions=data['turn_directions'],traffic_controls=data['traffic_controls'], 
                                    rotate_mat=data['rotate_mat'])
        else: 
            out2 = [None]*self.historical_steps
            for t in self.historical_steps: 
                out2[t] = out[t,:,:]
        del edge_dict, edge_index, edge_attr , out
        return out2
            
class FullyConnectedEncoder(nn.Module): 
    def __init__(self,
                 historical_steps: int,
                 embed_dim: int,
                 edge_dim: int,
                 patch_size: int,
                 num_modes: int = 6,
                 output_modes: bool=True,
                 num_heads: int = 8,
                 head_dim: int=16,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 rotate: bool = True) -> None:
        super(FullyConnectedEncoder, self).__init__()
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_modes = num_modes
        self.patch_size = patch_size
        self.output_modes = output_modes
        #if rotate:
        #    self.rel_embed = MultipleInputEmbedding(in_channels=[edge_dim, edge_dim], out_channel=embed_dim)
        #else:
        #    self.rel_embed = SingleInputEmbedding(in_channel=edge_dim, out_channel=embed_dim)
        
        self.fc_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=embed_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        #nn.ModuleList(
        #    [PairwiseInteractionLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        #     for _ in range(num_layers)])
        
        self.pos_embed = nn.Parameter(torch.Tensor(self.historical_steps//self.patch_size,1, embed_dim))
        nn.init.normal_(self.pos_embed, mean=0., std=.02)

        self.norm = nn.LayerNorm(embed_dim)
        if output_modes: 
            self.multihead_proj = nn.Linear(embed_dim, num_modes * embed_dim)
        else: 
            self.multihead_proj = nn.Linear(embed_dim, embed_dim)
        self.apply(init_weights)

    def forward(self, data, patch_embed): 
        #torch.cuda.empty_cache()
        patch_num = self.historical_steps//self.patch_size
        local_embed = torch.stack(patch_embed) #(num_patch, num_agent, dim) 
        inds = torch.cumsum(data['num_agent_nodes'],dim=0).cpu()
        local_embed_arrs = torch.tensor_split(local_embed, inds[:-1],dim=1)#(T, num_agent/bach, dim)*num_bach
        
        x = [v.reshape(-1,self.embed_dim) for v in local_embed_arrs]
        x = torch.vstack(x)
        

        #node_batch_ids = torch.cumsum(data['num_agent_nodes'],dim=0)
        #node_batch_ids = torch.cat([torch.tensor([0]).to(node_batch_ids.device), node_batch_ids], dim=0).long()
        #agent_batch_data =torch.empty((0,self.embed_dim)).float().to(node_batch_ids.device)
        #sample_data_shape = []
        #for bi in range(len(data['num_agent_nodes'])): 
        #    sample_data = torch.cat([patch_embed[i][node_batch_ids[bi]:node_batch_ids[bi+1],:] for i in range(patch_num)])
        #    sample_data = sample_data.reshape(patch_num, -1, self.embed_dim)
        #    sample_data = sample_data+self.pos_embed
        #    sample_data = sample_data.reshape(-1, self.embed_dim)
        #    agent_batch_data = torch.cat([agent_batch_data, sample_data],dim=0)
        #    sample_data_shape.append(sample_data.shape[0])
        for layer in self.fc_layers: 
            x = layer(x, None, data['patch_agent_agent_index'])
        
        #sample_data_shape = torch.tensor(sample_data_shape).to(node_batch_ids.device)
        #sample_data_shape_accum = torch.cumsum(sample_data_shape, dim=0)
        x = torch.tensor_split(x, inds[:-1]*patch_num,dim=0)
        x = [x[i][-data['num_agent_nodes'][i]:] for i  in range(len(x))]
        x = torch.vstack(x)
        
        #zld:todo
        #patch_edge_batch_ids = (data['patch_agent_agent_ids']==0).nonzero(as_tuple=True)[0]
        #patch_edge_batch_ids = torch.cat([patch_edge_batch_ids, 
        #                                  torch.tensor([len(data[f'patch_agent_agent_ids'])]).to(patch_edge_batch_ids.device)],
        #                                   dim=0)
        #agent_features = []
        #num_batch = len(data['num_agent_nodes'])
        #for bi in range(num_batch): 
        #    sample_data = torch.cat([patch_embed[i][node_batch_ids[bi]:node_batch_ids[bi+1],:] for i in range(patch_num)])
        #    sample_edge_index = data['patch_agent_agent_index'][:,patch_edge_batch_ids[bi]:patch_edge_batch_ids[bi+1]]
        #    sample_data = sample_data.reshape(patch_num, -1, self.embed_dim)
        #    sample_data = (sample_data + self.pos_embed).reshape(-1, self.embed_dim) 
        #    for layer in self.fc_layers: 
        #        sample_data = layer(sample_data, None, sample_edge_index)
        #    sample_data = sample_data.reshape(patch_num, -1, self.embed_dim)
        #    agent_features.append(sample_data[-1,:,:]) #find the feture in last timestamp
        #x = torch.cat(agent_features, dim=0)#[N,D], zld, to-debug 
        if self.output_modes: 
            x = self.multihead_proj(x).view(-1, self.num_modes, self.embed_dim)  # [N, F, D]
            x = x.transpose(0, 1)  # [F, N, D]
        else: 
            x = self.multihead_proj(x) #[N,D]

        #del sample_data, sample_edge_index, agent_features
        return x   

class PairwiseInteractionLayer(MessagePassing): 
    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 **kwargs) -> None:
        super(PairwiseInteractionLayer, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.lin_q_node = nn.Linear(embed_dim, embed_dim)
        self.lin_k_node = nn.Linear(embed_dim, embed_dim)
        #self.lin_k_edge = nn.Linear(embed_dim, embed_dim)
        self.lin_v_node = nn.Linear(embed_dim, embed_dim)
        #self.lin_v_edge = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))

        self.apply(init_weights)
    def forward(self,
                x: torch.Tensor,
                edge_index: Adj,
                size: Size = None) -> torch.Tensor:
        x = x + self._mha_block(self.norm1(x), edge_index, size)
        x = x + self._ff_block(self.norm2(x))
        return x

    def message(self,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:
        query = self.lin_q_node(x_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key_node = self.lin_k_node(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value_node = self.lin_v_node(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * (key_node )).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return (value_node ) * alpha.unsqueeze(-1)

    def update(self,
               inputs: torch.Tensor,
               x: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x))
        return inputs + gate * (self.lin_self(x) - inputs)

    def _mha_block(self,
                   x: torch.Tensor,
                   edge_index: Adj,
                   size: Size) -> torch.Tensor:
        x = self.out_proj(self.propagate(edge_index=edge_index, x=x, edge_attr=None,size=size))
        return self.proj_drop(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
       