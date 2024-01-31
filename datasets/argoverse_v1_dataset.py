# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from itertools import permutations
from itertools import product, chain
from typing import Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from argoverse.map_representation.map_api import ArgoverseMap
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from tqdm import tqdm

from utils import TemporalData
from argoverse.utils.centerline_utils import (
   # filter_candidate_centerlines,
    get_centerlines_most_aligned_with_trajectory,
    remove_overlapping_lane_seq,
)


_polygon_to_polygon_types = ['NONE', 'PRED', 'SUCC', 'LEFT', 'RIGHT']

class ArgoverseV1Dataset(Dataset):

    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 local_radius: float = 50,
                 patch_size: int=10) -> None:
        self._split = split
        self._local_radius = local_radius
        self._patch_size = patch_size
        self._url = f'https://s3.amazonaws.com/argoai-argoverse/forecasting_{split}_v1.1.tar.gz'
        if split == 'sample':
            self._directory = 'forecasting_sample'
        elif split == 'train':
            self._directory = 'train'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'test_obs'
        else:
            raise ValueError(split + ' is not valid')
        self.root = root
        self._raw_file_names = os.listdir(self.raw_dir)
        self._processed_file_names = [os.path.splitext(f)[0] + '.pt' for f in self.raw_file_names]
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]
        super(ArgoverseV1Dataset, self).__init__(root, transform=transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'data')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self._directory, f'processed_{self._patch_size}_new')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def process(self) -> None:
        am = ArgoverseMap()
       
        for raw_path in tqdm(self.raw_paths):
            dst_file = os.path.join(self.processed_dir, os.path.basename(raw_path)[:-4]+ '.pt')
            if not Path(dst_file).exists(): 
                kwargs = process_argoverse(self._split, raw_path, am, self._local_radius, self._patch_size)
                data = TemporalData(**kwargs)
                torch.save(data, os.path.join(self.processed_dir, str(kwargs['seq_id']) + '.pt'))

    def len(self) -> int:
        return len(self._raw_file_names)

    def get(self, idx) -> Data:
        #print(self.processed_paths[idx])
        try:
           temp= torch.load(self.processed_paths[idx])
        except Exception as e:
           print(self.processed_paths[idx]) 
           am = ArgoverseMap() 
           filename = os.path.basename(self.processed_paths[idx])
           raw_path = os.path.join('/mnt/disk1/lidanzha/data/argoverse/av1', self._split, 'data', filename[:-3]+'.csv')
           print(raw_path)
           kwargs = process_argoverse(self._split, raw_path, am, self._local_radius, self._patch_size)
           data = TemporalData(**kwargs)
           torch.save(data, self.processed_paths[idx])
           temp= torch.load(self.processed_paths[idx])
                
           
        temp_dict = temp.to_dict()
        if 'num_agent_nodes' not in temp_dict: 
            
            temp.__setitem__('num_agent_nodes', temp.__getitem__('num_nodes'))
        if 'patch_agent_agent_index' not in temp_dict:
            if f'patch_index_{self._patch_size}' not in temp_dict:
                process_path  = self.processed_paths[idx]
                am = ArgoverseMap()
                kwargs = process_argoverse(self._split, os.path.join(self.raw_dir, os.path.basename(process_path[:-3]+'.csv')),
                                           am ,  self._local_radius, self._patch_size)
                data = TemporalData(**kwargs)
                torch.save(data, process_path)
                temp= torch.load(self.processed_paths[idx])
            else: 
                temp.__setitem__('patch_agent_agent_index', temp.__getitem__(f'patch_index_{self._patch_size}'))
                temp.__setitem__('patch_agent_agent_ids', temp.__getitem__(f'patch_index_{self._patch_size}_ids')) 
                temp.__delitem__('num_nodes')   
                temp.__delitem__(f'patch_index_{self._patch_size}')
                temp.__delitem__(f'patch_index_{self._patch_size}_ids')       
        if 'lane_lane_edge_index' in temp_dict: 
            temp.__delitem__('lane_lane_edge_index')
            temp.__delitem__('lane_lane_edge_ids')
            temp.__delitem__('lane_lane_edge_type')
        return temp

def process_argoverse(split: str,
                      raw_path: str,
                      am: ArgoverseMap,
                      radius: float,
                      patch_size: int) -> Dict:
    df = pd.read_csv(raw_path)

    # filter out actors that are unseen during the historical time steps
    timestamps = list(np.sort(df['TIMESTAMP'].unique()))
    historical_timestamps = timestamps[: 20]
    historical_df = df[df['TIMESTAMP'].isin(historical_timestamps)]
    actor_ids = list(historical_df['TRACK_ID'].unique())
    df = df[df['TRACK_ID'].isin(actor_ids)]
    num_nodes = len(actor_ids)

    av_df = df[df['OBJECT_TYPE'] == 'AV'].iloc
    av_index = actor_ids.index(av_df[0]['TRACK_ID'])
    agent_df = df[df['OBJECT_TYPE'] == 'AGENT'].iloc
    agent_index = actor_ids.index(agent_df[0]['TRACK_ID'])
    city = df['CITY_NAME'].values[0]

    # make the scene centered at AV
    origin = torch.tensor([av_df[19]['X'], av_df[19]['Y']], dtype=torch.float)
    av_heading_vector = origin - torch.tensor([av_df[18]['X'], av_df[18]['Y']], dtype=torch.float)
    theta = torch.atan2(av_heading_vector[1], av_heading_vector[0])
    rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                               [torch.sin(theta), torch.cos(theta)]])

    # initialization
    x = torch.zeros(num_nodes, 50, 2, dtype=torch.float)
    edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
    padding_mask = torch.ones(num_nodes, 50, dtype=torch.bool)
    bos_mask = torch.zeros(num_nodes, 20, dtype=torch.bool)
    rotate_angles = torch.zeros(num_nodes, dtype=torch.float)
    x_raw = torch.zeros(num_nodes, 50, 2, dtype=torch.float)

    for actor_id, actor_df in df.groupby('TRACK_ID'):
        node_idx = actor_ids.index(actor_id)
        node_steps = [timestamps.index(timestamp) for timestamp in actor_df['TIMESTAMP']]
        padding_mask[node_idx, node_steps] = False
        if padding_mask[node_idx, 19]:  # make no predictions for actors that are unseen at the current time step
            padding_mask[node_idx, 20:] = True
        xy = torch.from_numpy(np.stack([actor_df['X'].values, actor_df['Y'].values], axis=-1)).float()
        x[node_idx, node_steps] = torch.matmul(xy - origin, rotate_mat)
        x_raw[node_idx, node_steps] = xy
        node_historical_steps = list(filter(lambda node_step: node_step < 20, node_steps))
        if len(node_historical_steps) > 1:  # calculate the heading of the actor (approximately)
            heading_vector = x[node_idx, node_historical_steps[-1]] - x[node_idx, node_historical_steps[-2]]
            rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])
        else:  # make no predictions for the actor if the number of valid time steps is less than 2
            padding_mask[node_idx, 20:] = True

    # bos_mask is True if time step t is valid and time step t-1 is invalid
    bos_mask[:, 0] = ~padding_mask[:, 0]
    bos_mask[:, 1: 20] = padding_mask[:, : 19] & ~padding_mask[:, 1: 20]

    positions = x.clone()
    x[:, 20:] = torch.where((padding_mask[:, 19].unsqueeze(-1) | padding_mask[:, 20:]).unsqueeze(-1),
                            torch.zeros(num_nodes, 30, 2),
                            x[:, 20:] - x[:, 19].unsqueeze(-2))
    x[:, 1: 20] = torch.where((padding_mask[:, : 19] | padding_mask[:, 1: 20]).unsqueeze(-1),
                              torch.zeros(num_nodes, 19, 2),
                              x[:, 1: 20] - x[:, : 19])
    x[:, 0] = torch.zeros(num_nodes, 2)

    # get lane features at the current time step
    df_19 = df[df['TIMESTAMP'] == timestamps[19]]
    node_inds_19 = [actor_ids.index(actor_id) for actor_id in df_19['TRACK_ID']]
    node_positions_19 = torch.from_numpy(np.stack([df_19['X'].values, df_19['Y'].values], axis=-1)).float()
    (lane_vectors, is_intersections, turn_directions, traffic_controls, 
     lane_actor_index, lane_actor_vectors,
     lane_lane_index, lane_lane_type, lane_lane_ids) = get_lane_features(am, node_inds_19, node_positions_19, origin, rotate_mat, 
                                                                         city, radius, x_raw,~padding_mask[:, :20])
    lane_lane_num = lane_lane_ids.shape[0]
    lane_num = lane_vectors.shape[0]
    lane_ids = torch.LongTensor(torch.arange(0, lane_num)).contiguous()
    y = None if split == 'test' else x[:, 20:]
    seq_id = os.path.splitext(os.path.basename(raw_path))[0]

    #patch ids: 
    patch_num = 20//patch_size 
    if patch_num<=1: #not used patch
        patch_index, patch_index_ids=None, None
    else:
        index_per_patch = [None]*patch_num
        for i in range(patch_num): 
            index_per_patch[i] = edge_index.clone()+num_nodes*i
            
        cross_patch_index = set()
        for t1 in range(0, patch_num-1): 
            for t2 in range(t1+1, patch_num): 
                list1 = list(torch.unique(index_per_patch[t1]))#edge_index.clone()+t1*edge_index.new_tensor([[2], [edge_index.shape[1]]])
                list2 = list(torch.unique(index_per_patch[t2]))#edge_index.clone()+t2*edge_index.new_tensor([[2], [edge_index.shape[1]]])
                for r in chain(product(list1, list2), product(list2,list1)):
                    cross_patch_index.add((r[0],r[1]))
        cross_patch_index = torch.tensor(list(cross_patch_index)).t().contiguous()
        patch_index = torch.cat(index_per_patch, dim=1)
        patch_index = torch.cat([patch_index,cross_patch_index], dim=1).contiguous()
        patch_index_ids = torch.LongTensor(torch.arange(0, patch_index.shape[1])).contiguous()
    return {
        'x': x[:, : 20],  # [N, 20, 2]
        'positions': positions,  # [N, 50, 2]
        'edge_index': edge_index,  # [2, N x N - 1]
        'y': y,  # [N, 30, 2]
        #'num_nodes': num_nodes, #=N
        'num_agent_nodes':num_nodes,
        'padding_mask': padding_mask,  # [N, 50]
        'bos_mask': bos_mask,  # [N, 20]
        'rotate_angles': rotate_angles,  # [N]
        'lane_vectors': lane_vectors,  # [L, 2]
        'lane_ids': lane_ids, #[L], 0,1,...,L-1
        'num_lanes': lane_num, #=L
        #'lane_lane_edge_index':lane_lane_index,
        #'lane_lane_edge_type':lane_lane_type,
        #'lane_lane_edge_ids':lane_lane_ids,#[0,#edge]
        'num_lane_lane': lane_lane_num,
        'is_intersections': is_intersections,  # [L]
        'turn_directions': turn_directions,  # [L]
        'traffic_controls': traffic_controls,  # [L]
        'lane_actor_index': lane_actor_index,  # [2, E_{A-L}]
        'lane_actor_vectors': lane_actor_vectors,  # [E_{A-L}, 2]
        #f'patch_index_{patch_size}':patch_index, 
        #f'patch_index_{patch_size}_ids':patch_index_ids,
        'patch_agent_agent_index':patch_index,
        'patch_agent_agent_ids':patch_index_ids, 
        'seq_id': int(seq_id),
        'av_index': av_index,
        'agent_index': agent_index,
        'city': city,
        'origin': origin.unsqueeze(0),
        'theta': theta,
    }


def safe_list_index(ls, elem):
    try:
        return ls.index(elem)
    except ValueError:
        return None

def get_lane_features(am: ArgoverseMap,
                      node_inds: List[int],
                      node_positions: torch.Tensor,
                      origin: torch.Tensor,
                      rotate_mat: torch.Tensor,
                      city: str,
                      radius: float, 
                      raw_traj: torch.Tensor, 
                      valid_traj:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                              torch.Tensor]:
    lane_positions, lane_vectors, is_intersections, turn_directions, traffic_controls = [], [], [], [], []
    lane_ids = set()
    for node_position in node_positions:
        lane_ids.update(am.get_lane_ids_in_xy_bbox(node_position[0], node_position[1], city, radius))
    lane_ids = prune_lanes(am, lane_ids, raw_traj, valid_traj, city)
    node_positions = torch.matmul(node_positions - origin, rotate_mat).float()
    for lane_id in lane_ids:
        lane_centerline = torch.from_numpy(am.get_lane_segment_centerline(lane_id, city)[:, : 2]).float()
        lane_centerline = torch.matmul(lane_centerline - origin, rotate_mat)
        is_intersection = am.lane_is_in_intersection(lane_id, city)
        turn_direction = am.get_lane_turn_direction(lane_id, city)
        traffic_control = am.lane_has_traffic_control_measure(lane_id, city)
        lane_positions.append(lane_centerline[:-1])
        lane_vectors.append(lane_centerline[1:] - lane_centerline[:-1])
        count = len(lane_centerline) - 1
        is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8))
        if turn_direction == 'NONE':
            turn_direction = 0
        elif turn_direction == 'LEFT':
            turn_direction = 1
        elif turn_direction == 'RIGHT':
            turn_direction = 2
        else:
            raise ValueError('turn direction is not valid')
        turn_directions.append(turn_direction * torch.ones(count, dtype=torch.uint8))
        traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8))
    lane_positions = torch.cat(lane_positions, dim=0)
    lane_vectors = torch.cat(lane_vectors, dim=0)
    is_intersections = torch.cat(is_intersections, dim=0)
    turn_directions = torch.cat(turn_directions, dim=0)
    traffic_controls = torch.cat(traffic_controls, dim=0)

    lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), node_inds))).t().contiguous()
    lane_actor_vectors = \
        lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_vectors.size(0), 1)
    mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius
    lane_actor_index = lane_actor_index[:, mask]
    lane_actor_vectors = lane_actor_vectors[mask]
    
    polygon_to_polygon_edge_index = []
    polygon_to_polygon_type = []
    lane_ids = list(lane_ids)
    for lane_id in lane_ids: # lane_segment in map_api.get_scenario_lane_segments():
        lane_segment_idx = lane_ids.index(lane_id)
        pred_inds = []
        pred_lane_ids = am.get_lane_segment_predecessor_ids(lane_id, city)
        succ_lane_ids = am.get_lane_segment_successor_ids(lane_id, city)  
        l_lane_id = am.city_lane_centerlines_dict[city][lane_id].l_neighbor_id
        r_lane_id = am.city_lane_centerlines_dict[city][lane_id].r_neighbor_id 

        if pred_lane_ids is not None:    
            for pred in pred_lane_ids:
                pred_idx = safe_list_index(lane_ids, pred)
                if pred_idx is not None:
                    pred_inds.append(pred_idx)
        if len(pred_inds) != 0:
            polygon_to_polygon_edge_index.append(
                torch.stack([torch.tensor(pred_inds, dtype=torch.long),
                                 torch.full((len(pred_inds),), lane_segment_idx, dtype=torch.long)], dim=0))
            polygon_to_polygon_type.append(
                    torch.full((len(pred_inds),), _polygon_to_polygon_types.index('PRED'), dtype=torch.uint8))
            
        succ_inds = []
        if succ_lane_ids is not None: 
            for succ in succ_lane_ids:
                succ_idx = safe_list_index(lane_ids, succ)
                if succ_idx is not None:
                    succ_inds.append(succ_idx)
        if len(succ_inds) != 0:
            polygon_to_polygon_edge_index.append(
                    torch.stack([torch.tensor(succ_inds, dtype=torch.long),
                                 torch.full((len(succ_inds),), lane_segment_idx, dtype=torch.long)], dim=0))
            polygon_to_polygon_type.append(
                    torch.full((len(succ_inds),), _polygon_to_polygon_types.index('SUCC'), dtype=torch.uint8))    
        if l_lane_id is not None:
            left_idx = safe_list_index(lane_ids, l_lane_id)
            if left_idx is not None:
                polygon_to_polygon_edge_index.append(
                        torch.tensor([[left_idx], [lane_segment_idx]], dtype=torch.long))
                polygon_to_polygon_type.append(
                        torch.tensor([_polygon_to_polygon_types.index('LEFT')], dtype=torch.uint8))
        if r_lane_id is not None:
            right_idx = safe_list_index(lane_ids, r_lane_id)
            if right_idx is not None:
                polygon_to_polygon_edge_index.append(
                        torch.tensor([[right_idx], [lane_segment_idx]], dtype=torch.long))
                polygon_to_polygon_type.append(
                        torch.tensor([_polygon_to_polygon_types.index('RIGHT')], dtype=torch.uint8))   
    if len(polygon_to_polygon_edge_index) != 0:
            polygon_to_polygon_edge_index = torch.cat(polygon_to_polygon_edge_index, dim=1)
            polygon_to_polygon_type = torch.cat(polygon_to_polygon_type, dim=0)
            polygon_to_polygon_ids = torch.LongTensor(torch.arange(polygon_to_polygon_edge_index.shape[1])).contiguous()
    else:
            polygon_to_polygon_edge_index = torch.tensor([[], []], dtype=torch.long)
            polygon_to_polygon_type = torch.tensor([], dtype=torch.uint8)
            polygon_to_polygon_ids = torch.LongTensor([]).contiguous()
            


    return lane_vectors, is_intersections, turn_directions, traffic_controls, \
            lane_actor_index, lane_actor_vectors,\
            polygon_to_polygon_edge_index,polygon_to_polygon_type, polygon_to_polygon_ids

#ref def get_candidate_centerlines_for_traj(
def prune_lanes(am, raw_lane_ids, xys,
                valid_traj:torch.Tensor,
                city_name: str,
                viz: bool = False,
                max_search_radius: float = 50.0):
    all_candidate_lanes = set()

    for oi in range(xys.shape[0]): 
        if (not valid_traj[oi,-1]) or(not valid_traj[oi,0]): 
            continue 
        manhattan_threshold = 2.5
        xy = xys[oi,:20,:]
        curr_lane_candidates = am.get_lane_ids_in_xy_bbox(xy[-1,0], xy[-1, 1], city_name, manhattan_threshold)
            # Keep expanding the bubble until at least 1 lane is found
        while len(curr_lane_candidates) < 1 and manhattan_threshold < max_search_radius:
            manhattan_threshold *= 2
            curr_lane_candidates = am.get_lane_ids_in_xy_bbox(xy[-1, 0], xy[-1, 1], city_name, manhattan_threshold)

        if len(curr_lane_candidates) <= 0: 
            continue 
            
        # Set dfs threshold
        displacement = np.sqrt((xy[0, 0] - xy[-1, 0]) ** 2 + (xy[0, 1] - xy[-1, 1]) ** 2)
        dfs_threshold = displacement * 2.0
        #print(dfs_threshold)
        # DFS to get all successor and predecessor candidates
        obs_pred_lanes: List[List[int]] = []
        for lane in curr_lane_candidates:
            candidates_future = am.dfs(lane, city_name, 0, dfs_threshold)
            candidates_past = am.dfs(lane, city_name, 0, dfs_threshold, True)

            # Merge past and future
            for past_lane_seq in candidates_past:
                for future_lane_seq in candidates_future:
                    assert past_lane_seq[-1] == future_lane_seq[0], "Incorrect DFS for candidate lanes past and future"
                    obs_pred_lanes.append(past_lane_seq + future_lane_seq[1:])

        # Removing overlapping lanes
        obs_pred_lanes = remove_overlapping_lane_seq(obs_pred_lanes)

        # Remove unnecessary extended predecessors
        obs_pred_lanes = am.remove_extended_predecessors(obs_pred_lanes, xy, city_name)

        # Getting candidate centerlines
        candidate_cl = am.get_cl_from_lane_seq(obs_pred_lanes, city_name)

        # Reduce the number of candidates based on distance travelled along the centerline
        #candidate_centerlines = filter_candidate_centerlines(xy, candidate_cl)
        candidate_centerlines = candidate_cl 
        # If no candidate found using above criteria, take the onces along with travel is the maximum
        if len(candidate_centerlines) < 1:
            candidate_centerlines = get_centerlines_most_aligned_with_trajectory(xy, candidate_cl)
        if 0:
            import matplotlib.pyplot as plt 
            from argoverse.utils.mpl_plotting_utils import visualize_centerline

            plt.figure(0, figsize=(8, 7))
            for centerline_coords in candidate_centerlines:
                visualize_centerline(centerline_coords)
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "-",
                color="#d33e4c",
                alpha=1,
                linewidth=1,
                zorder=15,
            )

            final_x = xy[-1, 0]
            final_y = xy[-1, 1]

            #plt.plot(final_x, final_y, "o", color="#d33e4c", alpha=1, markersize=7, zorder=15)
            plt.plot(xy[:,0], xy[:,1], "o-", color="#d33e4c", alpha=1, markersize=7, zorder=15)
            plt.xlabel("Map X")
            plt.ylabel("Map Y")
            plt.axis("off")
            plt.title("Number of candidates = {}".format(len(candidate_centerlines)))
            plt.show()
        
        for centerline_coords in candidate_centerlines:
            for centerline_coord in centerline_coords: 
                nearby_lane_ids = am.get_lane_ids_in_xy_bbox(centerline_coord[0], centerline_coord[1], city_name, query_search_range_manhattan=5.)
                all_candidate_lanes.update(nearby_lane_ids)
    all_candidate_lanes = list(all_candidate_lanes)
    valid_lanes = set()
    for lane_id in raw_lane_ids: 
        if lane_id in all_candidate_lanes:
            valid_lanes.add(lane_id)
    #print('before:', len(raw_lane_ids),'  after:',
    return valid_lanes
