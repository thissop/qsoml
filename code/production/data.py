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

def process_data(old_data_dir:str, new_data_dir:str, max_count:int=10000):
    import os 
    from astropy.io import fits 
    import numpy as np
    from tqdm import tqdm 
    import pandas as pd
    from scipy import interpolate

    count = 0

    zs = []
    spectrum_names = []

    for file in tqdm(os.listdir(old_data_dir)): 
        file_path = os.path.join(old_data_dir, file)

        try: 
            with fits.open(file_path) as hdul: 
                data = hdul[1].data
                x = 10**data['loglam']
                model = data['model']
                flux = data['flux']
                z = hdul[2].data['z'][0]

                data_min = np.min(x)

                if data_min >= 3590 and data_min <= 3600 and count<=max_count: 
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

                if count>=max_count: 
                    break

                    # Interpolate 
                    
                    # In ML Loading: 
                    # Do Sky Lines
                    # Divide by Median 
        except: 
            continue 

    print('count')

    df = pd.DataFrame()
    df['name'] = spectrum_names
    df['z'] = zs
    df.to_csv(os.path.join(new_dir, 'zkey.csv'), index=False)

#old_dir = 'data/small-batch'
#new_dir = 'data/csv-batch'

#process_data(old_dir, new_dir)

import os
from astropy.io import fits
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy import interpolate

def process_data_with_ivar(fits_dir: str, csv_dir: str, out_dir: str):
    import os
    from astropy.io import fits
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from scipy import interpolate
    from emission_lines import emissionlines  # <- import emission lines

    window_width = 10  # +/- 10 Å masking window

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv') and 'spec' in f]
    target_basenames = [f.replace('.csv', '') for f in csv_files]

    zs = []
    spectrum_names = []

    os.makedirs(out_dir, exist_ok=True)

    for basename in tqdm(target_basenames):
        fits_path = os.path.join(fits_dir, basename + '.fits')
        csv_path = os.path.join(out_dir, basename + '.csv')

        if not os.path.exists(fits_path):
            print(f"Warning: FITS file not found for {basename}")
            continue

        try:
            with fits.open(fits_path) as hdul:
                data = hdul[1].data
                x_fits = 10 ** data['loglam']
                model = data['flux']
                ivar = data['ivar']
                z = hdul[2].data['z'][0]

                if np.min(x_fits) < 3590 or np.min(x_fits) > 3600:
                    continue

                x_new = np.linspace(3600, 10300, 4500)
                f_model = interpolate.interp1d(x_fits, model, bounds_error=False, fill_value=0.0)
                f_ivar = interpolate.interp1d(x_fits, ivar, bounds_error=False, fill_value=0.0)

                y_new = f_model(x_new)
                ivar_new = f_ivar(x_new)

                # Mask ivar around redshifted emission lines
                for rest_line in emissionlines:
                    obs_line = rest_line * (1 + z)
                    mask = np.abs(x_new - obs_line) <= window_width
                    ivar_new[mask] = 0.0

                df = pd.DataFrame({'x': x_new, 'y': y_new, 'ivar': ivar_new})
                df.to_csv(csv_path, index=False)

                zs.append(z)
                spectrum_names.append(basename)

        except Exception as e:
            print(f"Error processing {basename}: {e}")
            continue

    # Save zkey
    zkey_df = pd.DataFrame({'name': spectrum_names, 'z': zs})
    zkey_df.to_csv(os.path.join(out_dir, 'zkey.csv'), index=False)

    print(f"\nProcessed {len(spectrum_names)} spectra.")

# ==== USAGE ====

fits_dir = '/Users/tkiker/Documents/GitHub/qsoml/data/small-batch'
csv_dir = '/Users/tkiker/Documents/GitHub/qsoml/data/csv-batch'
out_dir = '/Users/tkiker/Documents/GitHub/qsoml/data/csv-batch'  # You want to overwrite

process_data_with_ivar(fits_dir, csv_dir, out_dir)

import os
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from emission_lines import emissionlines

def plot_spectrum_with_ivar(csv_path, fits_dir):
    import smplotlib 
    # Get basename without extension
    basename = os.path.basename(csv_path).replace('.csv', '')
    fits_path = os.path.join(fits_dir, basename + '.fits')

    # Load redshift z from fits file
    with fits.open(fits_path) as hdul:
        z = hdul[2].data['z'][0]

    # Load the CSV
    df = pd.read_csv(csv_path)

    # Create figure
    fig, axs = plt.subplots(2, 1, figsize=(15, 5), sharex=True, gridspec_kw={'hspace': 0.1})

    # Top plot: x vs y
    axs[0].plot(df['x'], df['y'], color='black')
    axs[0].set_ylabel('Flux')
    axs[0].set_title(f"{basename}  (z = {z:.3f})")

    # Bottom plot: x vs ivar
    axs[1].plot(df['x'], df['ivar'], color='blue')
    axs[1].set_xlabel('Wavelength (Å)')
    axs[1].set_ylabel('IVAR')

    # Add redshifted emission lines
    for rest_wave, label in emissionlines.items():
        obs_wave = rest_wave * (1 + z)
        if df['x'].min() <= obs_wave <= df['x'].max():
            for ax in axs:
                ax.axvline(obs_wave, color='red', linestyle='--', alpha=0.4)
            axs[0].text(obs_wave, axs[0].get_ylim()[1]*0.9, label, color='red',
                        fontsize=7, rotation=90, ha='right', va='top')

    plt.tight_layout()
    plt.savefig(f'/Users/tkiker/Documents/GitHub/qsoml/results/plots/{basename}-ivar-lines.png', dpi=300)

# ==== Example usage ====
csv_sample = '/Users/tkiker/Documents/GitHub/qsoml/data/csv-batch/spec-7746-58074-0638.csv'
fits_dir = '/Users/tkiker/Documents/GitHub/qsoml/data/small-batch'

plot_spectrum_with_ivar(csv_sample, fits_dir)