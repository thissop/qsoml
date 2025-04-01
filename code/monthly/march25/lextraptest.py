import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import smplotlib 

observed_range = [3600, 10300]
z1 = 2.14  # spec-3586-55181-0850
z2 = 1.60  # spec-4219-55480-0194

file1 = "/Users/tkiker/Documents/GitHub/qsoml/data/csv-batch/spec-3586-55181-0850.csv"
file2 = "/Users/tkiker/Documents/GitHub/qsoml/data/csv-batch/spec-4219-55480-0194.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

wave_obs1 = df1['x'].values
wave_obs2 = df2['x'].values
wave_rest1 = wave_obs1 / (1 + z1)
wave_rest2 = wave_obs2 / (1 + z2)

rest_min = observed_range[0] / (1 + 2.2)  # z_max
rest_max = observed_range[1] / (1 + 1.5)  # z_min
rest_grid = np.linspace(rest_min, rest_max, len(wave_obs1))  # match length

spec1_interp = np.interp(rest_grid, wave_rest1, df1['y'].values, left=0, right=0)
spec2_interp = np.interp(rest_grid, wave_rest2, df2['y'].values, left=0, right=0)

obs_mask1 = ((rest_grid * (1 + z1)) >= observed_range[0]) & ((rest_grid * (1 + z1)) <= observed_range[1])
obs_mask2 = ((rest_grid * (1 + z2)) >= observed_range[0]) & ((rest_grid * (1 + z2)) <= observed_range[1])

spec1_full = spec1_interp.copy()
spec2_full = spec2_interp.copy()
spec1_full[~obs_mask1] = np.random.normal(0, 0.1, size=(~obs_mask1).sum())
spec1_full[~obs_mask1] = spec1_full[~obs_mask1]+np.exp(-0.001*rest_grid[~obs_mask1])*25*np.cos(rest_grid[~obs_mask1]/50)
spec2_full[~obs_mask2] = np.random.normal(0, 0.1, size=(~obs_mask2).sum())

mask_1_unobs_2_obs = (~obs_mask1) & (obs_mask2)
mask_2_unobs_1_obs = (~obs_mask2) & (obs_mask1)

plt.figure(figsize=(12, 6))
plt.plot(rest_grid, spec1_full, label=f"Spec 1 (z={z1})")
plt.plot(rest_grid, spec2_full, label=f"Spec 2 (z={z2})")

plt.fill_between(rest_grid, -0.5, 0.5, where=mask_1_unobs_2_obs, color='red', alpha=0.2, label='Spec 1 unobs, Spec 2 obs')
plt.fill_between(rest_grid, -0.5, 0.5, where=mask_2_unobs_1_obs, color='blue', alpha=0.2, label='Spec 2 unobs, Spec 1 obs')

plt.xlabel("Rest-frame Wavelength")
plt.ylabel("Normalized Flux")
plt.title("Extrapolation Loss Regions Visualization")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

