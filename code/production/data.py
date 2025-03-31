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
#plot_spectrum_with_ivar(csv_sample, fits_dir)

def process_fits_to_csv(
    fits_dir: str,
    out_dir: str,
    z_range: list = [1.5, 2.2],
    wave_grid=(3600, 10300),
    num_bins=4500,
    mask_width=5  # ±5 Å telluric mask
):
    import os
    from astropy.io import fits
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from scipy import interpolate

    # Telluric lines list (SDSS standard)
    telluric_lines = [
        5577.338, 6300.304, 6363.776, 6867.200, 6870.000, 6884.000, 6900.000,
        7200.000, 7250.000, 7300.000, 7320.000, 7340.000, 7360.000, 7380.000,
        7400.000, 7420.000, 7440.000, 7460.000, 7480.000, 7500.000, 7510.000,
        7605.000, 7620.000, 7640.000, 7660.000, 7680.000, 7700.000, 7720.000,
        7740.000, 7760.000, 7780.000, 7800.000, 7820.000, 7840.000, 7860.000,
        7880.000, 7900.000, 8000.000, 8020.000, 8040.000, 8060.000, 8080.000,
        8100.000, 8120.000, 8140.000, 8160.000, 8180.000, 8200.000, 8220.000
    ]

    os.makedirs(out_dir, exist_ok=True)
    x_target = np.linspace(wave_grid[0], wave_grid[1], num_bins)

    processed = 0
    skipped = 0
    redshifts = []
    names = []

    fits_files = [f for f in os.listdir(fits_dir) if f.endswith('.fits')]

    for fname in tqdm(fits_files):
        try:
            path = os.path.join(fits_dir, fname)
            with fits.open(path) as hdul:
                data = hdul[1].data
                z = hdul[2].data['z'][0]
                if not (z_range[0] <= z <= z_range[1]):
                    skipped += 1
                    continue

                x_raw = 10 ** data['loglam']
                flux = data['flux']
                ivar = data['ivar']

                # Require full coverage of target grid
                if x_raw[0] > wave_grid[0] or x_raw[-1] < wave_grid[1]:
                    skipped += 1
                    continue

                # Interpolate flux and ivar
                f_flux = interpolate.interp1d(x_raw, flux, bounds_error=False, fill_value=0.0)
                f_ivar = interpolate.interp1d(x_raw, ivar, bounds_error=False, fill_value=0.0)
                y_interp = f_flux(x_target)
                ivar_interp = f_ivar(x_target)

                # Mask bad pixels
                bad_pixel_mask = ivar_interp == 0.0

                # Mask telluric regions
                telluric_mask = np.zeros_like(x_target, dtype=bool)
                for line in telluric_lines:
                    telluric_mask |= (np.abs(x_target - line) <= mask_width)

                # Combine masks
                full_mask = bad_pixel_mask | telluric_mask
                ivar_interp[full_mask] = 0.0

                # Track masked fraction
                masked_frac = 100 * np.sum(ivar_interp == 0) / len(ivar_interp)
                if processed % 50 == 0:
                    print(f"{fname}: Masked {masked_frac:.2f}% of spectrum")

                # Save to CSV
                out_name = fname.replace('.fits', '.csv')
                df = pd.DataFrame({'x': x_target, 'y': y_interp, 'ivar': ivar_interp})
                df.to_csv(os.path.join(out_dir, out_name), index=False)

                redshifts.append(z)
                names.append(fname.replace('.fits', ''))
                processed += 1

        except Exception as e:
            print(f"Skipping {fname} due to error: {e}")
            skipped += 1
            continue

    # Save zkey
    df_key = pd.DataFrame({'name': names, 'z': redshifts})
    df_key.to_csv(os.path.join(out_dir, 'new_zkey.csv'), index=False)

    print(f"\n✅ Done. Processed: {processed} | Skipped: {skipped} | Total: {processed + skipped}")

fits_dir = '/Users/tkiker/Documents/GitHub/qsoml/data/small-batch'
csv_dir = 'data/csv-batch'
process_fits_to_csv(fits_dir, csv_dir)