import matplotlib.pyplot as plt 
import smplotlib 
import pandas as pd 

for i in range(3):

    data_df = pd.read_csv(f'/Users/tkiker/Documents/GitHub/qsoml/prediction_df_{i}.csv')

    wave_obs = data_df['wave_obs'].to_numpy()
    y_test = data_df['y_test'].to_numpy()
    y_pred = data_df['y_pred'].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 3)) 
    ax.plot(wave_obs, y_pred, label='Predicted', linewidth=1, color='red')
    ax.plot(wave_obs, y_test, label='Actual', linewidth=1, color='black')
    ax.set_xlabel('Observed Wavelength')
    ax.set_ylabel('Normalized Flux')
    ax.set_title(f'Spectrum Comparison')
    ax.legend()

    fig.tight_layout()
    plt.savefig(f'/Users/tkiker/Documents/GitHub/qsoml/results/plots/prediction-comparison_{i}.png')
    plt.clf()
    