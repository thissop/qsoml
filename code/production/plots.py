def plot_loss_components(history_df_path, plot_dir):
    import matplotlib.pyplot as plt 
    import os 
    import pandas as pd 

    history_df = pd.read_csv(history_df_path)
    epochs = history_df['epoch']

    plt.figure()
    plt.plot(epochs, history_df["L_fid"], label="Fidelity Loss")
    plt.plot(epochs, history_df["L_sim"], label="Similarity Loss")
    plt.plot(epochs, history_df["L_c"], label="Consistency Loss")
    plt.plot(epochs, history_df["L_extrap"], label="Extrapolation Loss")
    plt.plot(epochs, history_df["total_loss"], '--', label="Total Loss")
    plt.plot(epochs, history_df["val_loss"], ':', label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Components Over Epochs")
    plt.legend()
    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'history.png'))
    plt.close()

history_df_path = '/Users/tkiker/Documents/GitHub/qsoml/results/data-for-plots/history.csv'
plot_dir = '/Users/tkiker/Documents/GitHub/qsoml/results/plots'
plot_loss_components(history_df_path=history_df_path, plot_dir=plot_dir)

def plot_rest_frame_spectrum(restframe_df_path, plot_dir):
    import matplotlib.pyplot as plt
    import smplotlib 
    import pandas as pd 
    import os 
    import re

    # Prepare inputs
    restframe_df = pd.read_csv(restframe_df_path)
    wave_rest = restframe_df['x']
    rest_spectrum = restframe_df['y']

    # Plot
    plt.figure(figsize=(5, 3))
    plt.plot(wave_rest, rest_spectrum, label='Rest-Frame Reconstruction')
    plt.xlabel('Rest-Frame Wavelength (Å)')
    plt.ylabel('Flux')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{os.path.splitext(os.path.basename(restframe_df_path))[0]}.png'))
    plt.close()

restframe_df_paths = ['/Users/tkiker/Documents/GitHub/qsoml/results/data-for-plots/restframe_prediction_1.csv', 
                      '/Users/tkiker/Documents/GitHub/qsoml/results/data-for-plots/restframe_prediction_3.csv',
                      '/Users/tkiker/Documents/GitHub/qsoml/results/data-for-plots/restframe_prediction_10.csv', 
                      '/Users/tkiker/Documents/GitHub/qsoml/results/data-for-plots/restframe_prediction_21.csv']

for restframe_df_path in restframe_df_paths:
    plot_rest_frame_spectrum(restframe_df_path, plot_dir)

def plot_reconstruction(df_path, plot_dir): 
    import pandas as pd
    import matplotlib.pyplot as plt 
    import os  
    import smplotlib 

    df = pd.read_csv(df_path)
    x = df['x']
    y_real = df['y_test']
    y_predicted = df['y_predicted']

    fig, ax = plt.subplots(figsize=(5, 3))

    ax.plot(x, y_real, label='Flux', color='black')
    ax.plot(x, y_predicted, label='Reconstructed Flux', color='red')
    ax.set_xlabel('Observed Wavelength (Å)')
    ax.set_ylabel('Flux')
    ax.legend(fontsize='small')
    fig.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{os.path.splitext(os.path.basename(df_path))[0]}.png'))
    plt.close()

df_paths = ['/Users/tkiker/Documents/GitHub/qsoml/results/data-for-plots/reconstruction_1.csv',
            '/Users/tkiker/Documents/GitHub/qsoml/results/data-for-plots/reconstruction_3.csv',
            '/Users/tkiker/Documents/GitHub/qsoml/results/data-for-plots/reconstruction_10.csv',
            '/Users/tkiker/Documents/GitHub/qsoml/results/data-for-plots/reconstruction_21.csv']

for df_path in df_paths:
    plot_reconstruction(df_path, plot_dir)