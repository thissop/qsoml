def train_test_split(X: tuple, test_prop: float = 0.1):
    import numpy as np

    idx = np.arange(len(X[0]))  
    np.random.shuffle(idx)

    test_size = int(len(X[0]) * test_prop)  
    test_idx, train_idx = idx[:test_size], idx[test_size:]

    split_data = [np.array(data)[train_idx] for data in X] + [np.array(data)[test_idx] for data in X]
    
    return tuple(split_data)

def load_data(data_dir:str):
    import pandas as pd 
    import os 
    import numpy as np
    
    z_df = pd.read_csv(os.path.join(data_dir, 'zkey.csv'))
    names, zs = z_df['name'].to_numpy(), z_df['z'].to_numpy()

    zs_sorted = []

    Y = []

    for file in os.listdir(data_dir): 
        if 'csv' in file and 'spec' in file: 
            spectrum_name = file.split('.')[0]
            zs_sorted.append(zs[np.argwhere(names==spectrum_name)].flatten())
            spectrum_df = pd.read_csv(os.path.join(data_dir, file))
            y = spectrum_df['y'].to_numpy()
            Y.append(y/np.median(y))
    
    y_train, y_test, z_train, z_test = train_test_split((Y, zs_sorted))

    return y_train, y_test, z_train, z_test

data_dir = 'data/csv-batch'
y_train, y_test, z_train, z_test = load_data(data_dir)