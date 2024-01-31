import torch 
import os 
from utils import TemporalData
from tqdm import tqdm
#from multiprocessing.pool import ThreadPool as Pool
from pathlib import Path
import threading 


patch_size = 1
root = '/mnt/disk1/lidanzha/data/argoverse/av1/'
#pool_size=500
thread_num=32
#import pudb;pudb.set_trace()
def process_file(pathnames, filenames, start, end): 
    for i in range(start, end): 
        filename = filenames[i]
        pathname = pathnames[i]
        print('processing:', pathname, filename)
        if filename in ['pre_filter.pt', 'pre_transform.pt']: 
            return 
        
        dst_folder = pathname+'_new'
        os.makedirs(dst_folder, exist_ok=True)
        saver_file = os.path.join(dst_folder, filename)
        if Path(saver_file).exists(): #exist
            return
        data = torch.load(os.path.join(pathname, filename))
        
    # data_dict = data.to_dict()
        data2 ={
            'x':data['x'], 
            'positions':data['positions'],
            'edge_index':data['edge_index'], #edge_attrs=data['edge_attrs'], 
            'y':data['y'], 
            'num_agent_nodes':data['x'].shape[0], 
            'padding_mask':data['padding_mask'], 
            'bos_mask':data['bos_mask'],
            'rotate_angles':data['rotate_angles'], 
            'lane_vectors':data['lane_vectors'],
            'lane_ids':data['lane_ids'], 
            'num_lanes':data['num_lanes'], 
            #'lane_lane_edge_ids':data['lane_lane_edge_ids'], 
            'num_lane_lane':data['num_lane_lane'],
            #'lane_lane_edge_index':data['lane_lane_edge_index'], 
            #'lane_lane_edge_type':data['lane_lane_edge_type'], 
            'is_intersections':data['is_intersections'], 
            'turn_directions':data['turn_directions'],
            'traffic_controls':data['traffic_controls'], 
            'lane_actor_index':data['lane_actor_index'], 
            'lane_actor_vectors':data['lane_actor_vectors'], 
            'seq_id':data['seq_id'], 
            #'patch_agent_agent_ids':data[f'patch_index_{patch_size}_ids'], 
            #'patch_agent_agent_index':data[f'patch_index_{patch_size}'],
            'av_index': data['av_index'],
            'agent_index':data['agent_index'],
            'city': data['city'],
            'origin': data['origin'],
            'theta': data['theta'],
        }
        data_new = TemporalData(**data2)
        torch.save(data_new, os.path.join(dst_folder, filename))


if 1: 
    #pool = Pool(pool_size)
    #split ='train'
    #process_dir = os.path.join(root, split, f'processed_{patch_size}')
    #process_files = os.listdir(process_dir)
    #for file in process_files: 
    #   pool.apply_async(process_file, (process_dir,file))
    #pool.close()
    #pool.join()
    dirs, files=[],[]
    for split in ['train']:#, 'val']:
        process_dir = os.path.join(root, split, f'processed_{patch_size}')
        process_files = os.listdir(process_dir)
        for file in tqdm(process_files): 
            if file not in ['pre_filter.pt', 'pre_transform.pt']: 
                dirs.append(process_dir)
                files.append(file)
    num_file = len(files)
    num_sample = int(num_file/thread_num+0.5)
    for n in range(0, num_file, num_sample): 
        stop = n+num_sample if n+num_sample <=num_file else num_file
        threading.Thread(target=process_file, args = (dirs, files, n, stop)).start()

else: 
    for split in [ 'val','train']: 
        process_dir = os.path.join(root, split, f'processed_{patch_size}')
        process_files = os.listdir(process_dir)
        for file in tqdm(process_files): 
            if file not in ['pre_filter.pt', 'pre_transform.pt']: 
                #print('processing:', file)
                process_file(process_dir, file)
#print('done!')
