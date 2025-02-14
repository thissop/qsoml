def copy_downloaded(quasar_key:str, old_dir:str, new_dir:str, z_range:tuple): 
    import os 
    from tqdm import tqdm 
    import pandas as pd 
    import shutil
    import numpy as np
    from astropy.io import fits

    df = pd.read_csv(quasar_key)

    zs, plates, mjds, fiberids = (df[i].to_numpy() for i in list(df))
    
    i = 0
    count = 10000
    for plate, mjd, fiberid in tqdm(zip(plates, mjds, fiberids)): 

        if i >= count: 
            break
        else: 
            fiberid = (4-len(str(fiberid)))*'0'+str(fiberid)
            file_name = f'spec-{plate}-{mjd}-{fiberid}.fits'
            
            old_path = os.path.join(old_dir, file_name)
            new_path = os.path.join(new_dir, file_name)
            if os.path.exists(old_path): 
                with fits.open(old_path) as hdul:
                    z = hdul[2].data['z'][0]
                    if z >= z_range[0] and z<=z_range[1] and i<count:
                        shutil.copy(old_path, new_path)
                        i+=1

quasar_key = 'data/quasar_key.csv'
old_dir = '/Users/tkiker/Documents/GitHub/AGN-UMAP/data/sdss_spectra/'
new_dir = 'data/small-batch'
#copy_downloaded(quasar_key, old_dir, new_dir, z_range=[1.5, 2.2])

def check_data_range(data_dir:str):
    import os 
    import numpy as np
    from astropy.io import fits
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import smplotlib 

    data_min = []
    data_max = []

    for file in tqdm(os.listdir(data_dir)):
        if file.endswith('.fits'):
            with fits.open(os.path.join(data_dir, file)) as hdul:
                data = hdul[1].data
                data_min.append(10**np.min(data['loglam']))
                data_max.append(10**np.max(data['loglam']))

                spectrum_length = len(data['flux'])

    plt.hist(data_min)
    plt.gca().ticklabel_format(style='plain', axis='both')
    plt.show()

    return np.max(data_min), np.min(data_max), spectrum_length

#print(check_data_range('data/small-batch'))

def process_data(old_data_dir:str, new_data_dir:str):
    import os 
    from astropy.io import fits 
    import numpy as np
    from tqdm import tqdm 
    import matplotlib.pyplot as plt 
    import smplotlib 
    import pandas as pd
    from scipy import interpolate

    count = 0


    zs = []
    spectrum_names = []

    for file in tqdm(os.listdir(old_data_dir)): 
        file_path = os.path.join(old_data_dir, file)


        with fits.open(file_path) as hdul: 
            data = hdul[1].data
            x = 10**data['loglam']
            model = data['model']
            flux = data['flux']
            z = hdul[2].data['z'][0]

            data_min = np.min(x)

            if data_min >= 3590 and data_min <= 3600 and count<=50: 
                f = interpolate.interp1d(x, model)
                x = np.linspace(3600, 10300, 4500)
                y = f(x)

                df = pd.DataFrame()
                df['x'] = x 
                df['y'] = y 

                zs.append(z)
                spectrum_names.append(file.split('.')[0])

                df.to_csv(os.path.join(new_data_dir, file.replace('fits', 'csv')), index=False)
                count += 1

            if count>=50: 
                break

                # Interpolate 
                
                # In ML Loading: 
                # Do Sky Lines
                # Divide by Median 

    df = pd.DataFrame()
    df['name'] = spectrum_names
    df['z'] = zs
    df.to_csv(os.path.join(new_dir, 'zkey.csv'), index=False)

old_dir = 'data/small-batch'
new_dir = 'data/csv-batch'

process_data(old_dir, new_dir)



            
