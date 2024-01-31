from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_weights
from utils import DistanceDropEdge
from models import ALEncoder
from models.mlp_layers import MLPLayer
from models.local_encoder import AttentionLayer
from models.fourier_embedding import FourierEmbedding

from torch_geometric.data import Batch
from torch_cluster import radius
from torch_cluster import radius_graph
from torch_geometric.utils import dense_to_sparse
import math 
import pickle

class TubeDecoder(nn.Module):

    def __init__(self,
                 channels: int,
                 historical_steps: int,
                 future_steps: int,
                 num_modes: int,
                 output_head: bool,
                 num_heads: int=8,
                 dropout: float = 0.1,
                 local_radius: int=50,
                 num_layers: int = 2,
                 uncertain: bool = True,
                 head_dim: int=16,
                 min_scale: float = 1e-3,
                 use_map_encoder: bool=False) -> None:
        super(TubeDecoder, self).__init__()
        self.input_size = channels
        self.num_historical_steps = historical_steps
        self.hidden_dim = channels
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.uncertain = uncertain
        self.min_scale = min_scale
        self.num_recurrent_steps = 3
        self.num_layers = num_layers
        num_freq_bands = 64
        local_radius=50
        self.local_radius=local_radius
        self.output_head = output_head
        self.use_map_encoder = use_map_encoder

        self.use_map_decoder=False
        if self.use_map_decoder: 
            self.m2lane_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=channels, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
            )
            self.m2lane_refine_attn_layers =  nn.ModuleList(
            [AttentionLayer(hidden_dim=channels, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
            )
            self.r_pl2m_emb = FourierEmbedding(input_dim=1, hidden_dim=channels,
                                           num_freq_bands=num_freq_bands)
        

        self.mode_emb = nn.Embedding(num_modes, channels)
        self.traj_emb = nn.GRU(input_size=channels, hidden_size=channels, num_layers=1, bias=True,
                               batch_first=False, dropout=0.0, bidirectional=False)
        self.traj_emb_h0 = nn.Parameter(torch.zeros(1, channels))
        self.y_emb = FourierEmbedding(input_dim=2 + output_head, hidden_dim=channels,
                                      num_freq_bands=num_freq_bands)
        self.r_a2m_emb = FourierEmbedding(input_dim=1, hidden_dim=channels,
                                           num_freq_bands=num_freq_bands)
        
        self.a2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=channels, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.m2m_propose_attn_layer = AttentionLayer(hidden_dim=channels, num_heads=num_heads, head_dim=head_dim,
                                                     dropout=dropout, bipartite=False, has_pos_emb=False)
        
        self.a2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=channels, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.m2m_refine_attn_layer = AttentionLayer(hidden_dim=channels, num_heads=num_heads, head_dim=head_dim,
                                                     dropout=dropout, bipartite=False, has_pos_emb=False)

        self.to_loc_propose_pos = MLPLayer(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim,
                                           output_dim=future_steps * 2 // self.num_recurrent_steps)
        self.to_scale_propose_pos = MLPLayer(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim,
                                             output_dim=future_steps * 2 // self.num_recurrent_steps)
        self.to_loc_refine_pos = MLPLayer(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim,
                                          output_dim=future_steps * 2)
        self.to_scale_refine_pos = MLPLayer(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim,
                                            output_dim=future_steps * 2)
        if output_head:
            self.to_loc_propose_head = MLPLayer(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim,
                                                output_dim=future_steps // self.num_recurrent_steps)
            self.to_conc_propose_head = MLPLayer(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim,
                                                 output_dim=future_steps // self.num_recurrent_steps)
            self.to_loc_refine_head = MLPLayer(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, output_dim=future_steps)
            self.to_conc_refine_head = MLPLayer(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim,
                                                output_dim=future_steps)
        else:
            self.to_loc_propose_head = None
            self.to_conc_propose_head = None
            self.to_loc_refine_head = None
            self.to_conc_refine_head = None
        self.to_pi = MLPLayer(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, output_dim=1)
        
        self.int_pl_emb = nn.Embedding(2, self.hidden_dim)
        self.turn_pl_emb = nn.Embedding(3, self.hidden_dim)
        self.traffic_pl_emb = nn.Embedding(2, self.hidden_dim)
        self.pl_pos_emb = MLPLayer(input_dim=2, hidden_dim=self.hidden_dim, output_dim=self.hidden_dim)
        self.pl_emb = MLPLayer(input_dim=self.hidden_dim*2, hidden_dim=self.hidden_dim, 
                               output_dim=self.hidden_dim)
        
        self.apply(init_weights)
    def forward(self,
                data: torch.Tensor, 
                agent_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        #torch.cuda.empty_cache()

        pos_m = data['positions'][:, self.num_historical_steps - 1, :2]
        agent_batch = data['batch']
        lane_batch   = torch.cumsum((data['lane_ids']==0).long(), dim=0)-1

        m = self.mode_emb.weight.repeat(agent_embed.size(0), 1)
        valid_mask = ~data['padding_mask'][:, :]
        if self.use_map_encoder: 
            x_pl  =data['map_enc'].repeat(self.num_modes, 1)
        else: 
            
            x_pl_embeds = self.pl_pos_emb(data['lane_vectors'])
            accum_emb = self.int_pl_emb(data['is_intersections'].long())+\
                        self.turn_pl_emb(data['turn_directions'].long())+\
                        self.traffic_pl_emb(data['traffic_controls'].long())
            x_pl = self.pl_emb(torch.cat([x_pl_embeds, accum_emb], dim=1))
            x_pl = x_pl.repeat(self.num_modes, 1)
            
        
        x_a = agent_embed.repeat(self.num_modes, 1)
        mask_src = valid_mask[:, :self.num_historical_steps].contiguous()
        #mask_src[:, :self.num_historical_steps - self.num_t2m_steps] = False
        mask_dst = valid_mask[:,self.num_historical_steps:].any(dim=-1, keepdim=True).repeat(1, self.num_modes)

        #pl2m 
        if self.use_map_decoder: 
            pos_pl = data['lane_vectors'][:, :2]
            edge_index_pl2m = radius(
                x=pos_m[:, :2],
                y=pos_pl[:, :2],
                r=self.local_radius,
                batch_x=agent_batch if isinstance(data, Batch) else None,
                batch_y=lane_batch if isinstance(data, Batch) else None,
                max_num_neighbors=200) #50
            edge_index_pl2m = edge_index_pl2m[:, mask_dst[edge_index_pl2m[1], 0]]
            rel_pos_pl2m = pos_pl[edge_index_pl2m[0]] - pos_m[edge_index_pl2m[1]]
            r_pl2m = torch.stack([torch.norm(rel_pos_pl2m[:, :2], p=2, dim=-1)], dim=-1)
            r_pl2m = self.r_pl2m_emb(continuous_inputs=r_pl2m, categorical_embs=None)
            edge_index_pl2m = torch.cat([edge_index_pl2m + i * edge_index_pl2m.new_tensor(
                [[data['lane_vectors'].shape[0]], [data['x'].shape[0]]]) for i in range(self.num_modes)], dim=1)
            r_pl2m = r_pl2m.repeat(self.num_modes, 1)
        
        #a2m 
        edge_index_a2m = radius_graph(
            x=pos_m[:, :2],
            r=self.local_radius,
            batch=agent_batch if isinstance(data, Batch) else None,
            loop=False,
            max_num_neighbors=50)#50
        edge_index_a2m = edge_index_a2m[:, mask_src[:, -1][edge_index_a2m[0]] & mask_dst[edge_index_a2m[1], 0]]
        rel_pos_a2m = pos_m[edge_index_a2m[0]] - pos_m[edge_index_a2m[1]]
        r_a2m = torch.stack([torch.norm(rel_pos_a2m[:, :2], p=2, dim=-1)], dim=-1)
        r_a2m = self.r_a2m_emb(continuous_inputs=r_a2m, categorical_embs=None)
        edge_index_a2m = torch.cat(
            [edge_index_a2m + i * edge_index_a2m.new_tensor([data['x'].shape[0]]) for i in
             range(self.num_modes)], dim=1)
        r_a2m = r_a2m.repeat(self.num_modes, 1)
        
        #m2m 
        edge_index_m2m = dense_to_sparse(mask_dst.unsqueeze(2) & mask_dst.unsqueeze(1))[0]


        locs_propose_pos= [None] * self.num_recurrent_steps
        scales_propose_pos= [None] * self.num_recurrent_steps
        locs_propose_head = [None] * self.num_recurrent_steps
        concs_propose_head = [None] * self.num_recurrent_steps
        for t in range(self.num_recurrent_steps):
            for i in range(self.num_layers):
                m = m.reshape(-1, self.hidden_dim)
                if self.use_map_decoder: 
                    m = self.m2lane_propose_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
                    m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                m = self.a2m_propose_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
                m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            m = self.m2m_propose_attn_layer(m, None, edge_index_m2m)
            m = m.reshape(-1, self.num_modes, self.hidden_dim)
            locs_propose_pos[t] = self.to_loc_propose_pos(m)
            scales_propose_pos[t] = self.to_scale_propose_pos(m)
            if self.output_head:
                locs_propose_head[t] = self.to_loc_propose_head(m)
                concs_propose_head[t] = self.to_conc_propose_head(m)

        loc_propose_pos = torch.cumsum(
            torch.cat(locs_propose_pos, dim=-1).view(-1, self.num_modes, self.future_steps,2),
            dim=-2)
        scale_propose_pos = torch.cumsum(
            F.elu_(
                torch.cat(scales_propose_pos, dim=-1).view(-1, self.num_modes, self.future_steps, 2),#self.output_dim),
                alpha=1.0) +
            1.0,
            dim=-2) + 0.1
        if self.output_head:
            loc_propose_head = torch.cumsum(torch.tanh(torch.cat(locs_propose_head, dim=-1).unsqueeze(-1)) * math.pi,
                                            dim=-2)
            conc_propose_head = 1.0 / (torch.cumsum(F.elu_(torch.cat(concs_propose_head, dim=-1).unsqueeze(-1)) + 1.0,
                                                    dim=-2) + 0.02)
            m = self.y_emb(torch.cat([loc_propose_pos.detach(),loc_propose_head.detach()], dim=-1).view(-1, 2+1))#self.output_dim + 1))
        else:
            loc_propose_head = loc_propose_pos.new_zeros((loc_propose_pos.size(0), self.num_modes,
                                                          self.future_steps, 1))
            conc_propose_head = scale_propose_pos.new_zeros((scale_propose_pos.size(0), self.num_modes,
                                                             self.future_steps, 1))
            m = self.y_emb(loc_propose_pos.detach().view(-1, 2))#self.output_dim))
        m = m.reshape(-1, self.future_steps, self.hidden_dim).transpose(0, 1)
        m = self.traj_emb(m, self.traj_emb_h0.unsqueeze(1).repeat(1, m.size(1), 1))[1].squeeze(0)
        for i in range(self.num_layers):
            if self.use_map_decoder: 
                m = self.m2lane_refine_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
                m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            m = self.a2m_refine_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
            m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
        m = self.m2m_refine_attn_layer(m, None, edge_index_m2m)
        m = m.reshape(-1, self.num_modes, self.hidden_dim)
        loc_refine_pos = self.to_loc_refine_pos(m).view(-1, self.num_modes, self.future_steps,2)# self.output_dim)
        loc_refine_pos = loc_refine_pos + loc_propose_pos.detach()
        scale_refine_pos = F.elu_(
            self.to_scale_refine_pos(m).view(-1, self.num_modes, self.future_steps, 2),#self.output_dim),
            alpha=1.0) + 1.0 + 0.1
        if self.output_head:
            loc_refine_head = torch.tanh(self.to_loc_refine_head(m).unsqueeze(-1)) * math.pi
            loc_refine_head = loc_refine_head + loc_propose_head.detach()
            conc_refine_head = 1.0 / (F.elu_(self.to_conc_refine_head(m).unsqueeze(-1)) + 1.0 + 0.02)
        else:
            loc_refine_head = loc_refine_pos.new_zeros((loc_refine_pos.size(0), self.num_modes, self.future_steps,
                                                        1))
            conc_refine_head = scale_refine_pos.new_zeros((scale_refine_pos.size(0), self.num_modes,
                                                           self.future_steps, 1))
        pi = self.to_pi(m).squeeze(-1)

        return {
            'loc_propose_pos': loc_propose_pos,
            'scale_propose_pos': scale_propose_pos,
            'loc_propose_head': loc_propose_head,
            'conc_propose_head': conc_propose_head,
            'loc_refine_pos': loc_refine_pos,
            'scale_refine_pos': scale_refine_pos,
            'loc_refine_head': loc_refine_head,
            'conc_refine_head': conc_refine_head,
            'pi': pi,
        }
        
    

