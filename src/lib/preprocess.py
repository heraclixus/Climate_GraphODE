import numpy as np

total_step = 1095 # total number of time steps
unit_len = 73 # chunk a sequence into subsequences of length unit_len (reason: predicting 1095 time steps at once is too much work)
n_sub_seq = total_step // unit_len

def construct_grid( H, W):
    '''
    H: number of latitude (32)
    W: number of longitude (64)
    '''
    adj = np.zeros((H*W,H*W)) # 
    for r in range(H):
        for c in range(W):
            t = r * W + c 
            up = (r+1) * W + c 
            down = (r-1) * W + c 
            left = r * W + (c-1)
            right = r * W + (c+1)
            
            neighbor = [up, down, left, right]
            for s in neighbor: 
                if s >= 0 and s <= (H*W-1):
                    adj[t,s] = 1
                    adj[s,t] = 1
    return adj 
                     
def preprocess( data_type):
    '''
    Output:
        geopotential.npy: [#seq, #node, #timestep, #feature] (#feature=1)
        edges.npy: [#seq, #node, #node]
        times.npy: [#seq, #node, #timesteps] 
    '''
    if data_type == 'train':
        start_year = 2014
        end_year = 2015
    elif data_type == 'val':
        start_year = 2016
        end_year = 2016
    else:
        start_year = 2017
        end_year = 2017
    print('Generating graphs and time series')
    time_series = []
    edges = []
    time_obs = [] # observed timestamps (required for LGODE)
    for year in range(start_year, end_year+1):
        print('processing', year)
        for i in range(8):
            data = np.load(f"../../data/geopotential_6.526deg_np/{data_type}/{year}_{i}.npz")
            H, W = data['geopotential'].shape[2], data['geopotential'].shape[3]
            geopotential = data['geopotential'].squeeze(1).reshape(total_step,-1)  
            num_nodes = geopotential.shape[-1] 
            geopotential = geopotential.reshape(n_sub_seq, unit_len, num_nodes) # slice
            geopotential = geopotential.transpose(0,2,1)
            geopotential = np.expand_dims(geopotential, axis=3) # expand a dimension for feature
            time_series.append(geopotential)  
            adj =  construct_grid(H, W)
            adj = np.tile(adj, (n_sub_seq,1))
            adj = adj.reshape(n_sub_seq,num_nodes,num_nodes) 
            edges.append(adj)
            time_obs.append(np.ones((n_sub_seq, num_nodes, unit_len))) 

    time_series = np.concatenate(time_series, axis=0)  # train: 4440 x 73 x 2048 x 1
    edges = np.concatenate(edges, axis=0)
    time_obs = np.concatenate(time_obs, axis=0)  
    print('time_series', time_series.shape)
    print('edges', edges.shape)
    print('time_obs', time_obs.shape)
    return time_series, edges, time_obs 
preprocess('train')
