

def copy_downloaded(quasar_key:str, old_dir:str, new_dir:str): 
    import os 
    from tqdm import tqdm 
    import pandas as pd 
    import shutil
    import numpy as np

    df = pd.read_csv(quasar_key)

    zs, plates, mjds, fiberids = (df[i].to_numpy()[0:50] for i in list(df))

    for plate, mjd, fiberid in tqdm(zip(plates, mjds, fiberids)): 
        fiberid = (4-len(str(fiberid)))*'0'+str(fiberid)
        file_name = f'spec-{plate}-{mjd}-{fiberid}.fits'
        
        old_path = os.path.join(old_dir, file_name)
        new_path = os.path.join(new_dir, file_name)

        if os.path.exists(old_path):
            shutil.copy(old_path, new_path)

quasar_key = 'data/quasar_key.csv'
old_dir = '/Users/tkiker/Documents/GitHub/AGN-UMAP/data/sdss_spectra/'
new_dir = 'data/small-batch'
copy_downloaded(quasar_key, old_dir, new_dir)