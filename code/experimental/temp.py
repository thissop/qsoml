import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import smplotlib

# ---- Load Data ----
file_path = "/Users/tkiker/Documents/GitHub/qsoml/data/csv-batch/spec-3588-55184-0657.csv"
df = pd.read_csv(file_path)

# ---- Plot Flux & ivar ----
fig, ax1 = plt.subplots(figsize=(8, 4))

ax1.plot(df['x'], df['y'], color='black', lw=0.5, label='Flux')
ax1.set_xlabel("Observed Wavelength (Å)")
ax1.set_ylabel("Flux")
ax1.set_title("Observed Spectrum")

# Optional: plot ivar on secondary axis
if 'ivar' in df.columns:
    ax2 = ax1.twinx()
    ax2.plot(df['x'], df['ivar'], color='red', alpha=0.4, lw=0.5, label='ivar')
    ax2.set_ylabel("Inverse Variance")

# Optional: Overlay telluric lines
telluric_lines = [
    5577.338, 6300.304, 6363.776, 6867.200, 6870.000, 6884.000, 6900.000,
    7200.000, 7250.000, 7300.000, 7320.000, 7340.000, 7360.000, 7380.000,
    7400.000, 7420.000, 7440.000, 7460.000, 7480.000, 7500.000, 7510.000,
    7605.000, 7620.000, 7640.000, 7660.000, 7680.000, 7700.000, 7720.000,
    7740.000, 7760.000, 7780.000, 7800.000, 7820.000, 7840.000, 7860.000,
    7880.000, 7900.000, 8000.000, 8020.000, 8040.000, 8060.000, 8080.000,
    8100.000, 8120.000, 8140.000, 8160.000, 8180.000, 8200.000, 8220.000
]
for line in telluric_lines:
    ax1.axvline(line, color='blue', alpha=0.2, lw=0.7)

plt.tight_layout()
plt.show()
