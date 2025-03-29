def plot_loss_components(history_df_path, plot_dir):
    import matplotlib.pyplot as plt 
    import smplotlib 
    import os 
    import pandas as pd 

    history_df = pd.DataFrame(history_df_path)
    epochs = history_df['epoch']

    plt.figure()
    plt.plot(epochs, history_df["L_fid"], label="Fidelity Loss")
    plt.plot(epochs, history_df["L_sim"], label="Similarity Loss")
    plt.plot(epochs, history_df["L_c"], label="Consistency Loss")
    plt.plot(epochs, history_df["total_loss"], '--', label="Total Loss")
    plt.plot(epochs, history_df["val_loss"], ':', label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Components Over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'history.png'))

history_df_path = ''
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
    z = float(re.search(r'z=([\d\.]+)', restframe_df_path).group(1))

    # Plot
    plt.figure()
    plt.plot(wave_rest, rest_spectrum, label='Rest-Frame Reconstruction')
    plt.xlabel('Rest-Frame Wavelength (Å)')
    plt.ylabel('Flux')
    plt.title(f'Rest-Frame Spectrum (z = {z:.2f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{os.path.splitext(os.path.basename(restframe_df_path))[0]}.png'))
    plt.close()

restframe_df_paths = []
for restframe_df_path in restframe_df_paths: 
    plot_rest_frame_spectrum(restframe_df_path, plot_dir)