import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, PReLU, Input, Reshape, Lambda
from tensorflow.keras.models import Model

def train_test_split(X: tuple, test_prop: float = 0.1):
    import numpy as np

    n_samples = len(X[0])  # Get sample size from the first element
    idx = np.arange(n_samples)  
    np.random.shuffle(idx)

    test_size = int(n_samples * test_prop)  
    test_idx, train_idx = idx[:test_size], idx[test_size:]

    # Ensure all splits match the shape of X[0]
    split_data = []
    for data in X:
        data = np.array(data)
        if len(data.shape) == 1:  # Handle scalars (z values)
            split_data.append(data[train_idx].reshape(-1, 1))  # Ensure correct shape (N,1)
            split_data.append(data[test_idx].reshape(-1, 1))
        else:  # Handle spectra
            split_data.append(data[train_idx])
            split_data.append(data[test_idx])

    return tuple(split_data)

def load_data(data_dir: str):
    import pandas as pd
    import os
    import numpy as np

    z_df = pd.read_csv(os.path.join(data_dir, 'zkey.csv'))
    z_map = dict(zip(z_df['name'], z_df['z'].astype(np.float32)))  # Use a dictionary for fast lookup

    Y, zs_sorted = [], []

    for file in os.listdir(data_dir):
        if file.endswith('.csv') and 'spec' in file:
            name = file.split('.')[0]
            if name in z_map:  # Ensure the spectrum name is valid
                zs_sorted.append(z_map[name])  # Store scalar, not array

                spectrum_df = pd.read_csv(os.path.join(data_dir, file))
                y = spectrum_df['y'].to_numpy(dtype=np.float32)  # Ensure float32
                Y.append(y / np.median(y))  # Normalize

    # Convert lists to numpy arrays
    Y = np.array(Y, dtype=np.float32)
    zs_sorted = np.array(zs_sorted, dtype=np.float32)  # Ensure 1D array

    y_train, y_test, z_train, z_test = train_test_split((Y, zs_sorted))

    return y_train, y_test, z_train, z_test  # z_train is already reshaped correctly

data_dir = '/burg/home/tjk2147/src/GitHub/qsoml/data/csv-batch'
y_train, y_test, z_train, z_test = load_data(data_dir)

observed_range = [3600, 10300]
z_range = [1.5, 2.2]

wave_rest_min = observed_range[0] / (1 + z_range[1])  # Compute min rest-frame wavelength
wave_rest_max = observed_range[1] / (1 + z_range[0]) 

obs_length = len(y_train[0])
upsample_factor = 2
rest_length = upsample_factor*obs_length 

wave_rest = tf.cast(tf.linspace(tf.constant(wave_rest_min, dtype=tf.float32), 
                                tf.constant(wave_rest_max, dtype=tf.float32), 
                                num=rest_length), dtype=tf.float32)

wave_obs = tf.cast(tf.linspace(tf.constant(observed_range[0], dtype=tf.float32), 
                               tf.constant(observed_range[1], dtype=tf.float32), 
                               num=obs_length), dtype=tf.float32)


### LOSS FUNCTION RELATED ###

def similarity_loss(latent_vectors, spectra, w_prime, k0=0.5, k1=10):
    """
    Computes the similarity loss term:
    L_sim = 1/N^2 * sum(sigmoid(k1 * (S_ij - k0))) + sum(sigmoid(-k1 * (S_ij - k0)))

    where:
    S_ij = (||s_i - s_j||^2) / S - (||x'_i - x'_j||^2) / M
    
    NOTE FOR MEMORY EFFICIENCY: 

    The similarity_loss function computes all pairwise distances, leading to an O(N²) complexity. This can become very slow for large batch sizes.
    
    Fix: Replace the explicit broadcasting with matrix multiplication-based distance computation:

    def pairwise_squared_distances(X):
        #Efficient computation of pairwise squared distances using dot-product tricks.
        #Uses ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * (x_i . x_j)
        X_sq = tf.reduce_sum(tf.square(X), axis=-1, keepdims=True)  # ||x_i||^2
        pairwise_sq_dists = X_sq + tf.transpose(X_sq) - 2 * tf.matmul(X, X, transpose_b=True)
        return pairwise_sq_dists

    """

    def pairwise_squared_distances(X):
        #Efficient computation of pairwise squared distances using dot-product tricks.
        #Uses ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * (x_i . x_j)
        X_sq = tf.reduce_sum(tf.square(X), axis=-1, keepdims=True)  # ||x_i||^2
        pairwise_sq_dists = X_sq + tf.transpose(X_sq) - 2 * tf.matmul(X, X, transpose_b=True)
        return pairwise_sq_dists

    batch_size = tf.shape(latent_vectors)[0]

    latent_dists = pairwise_squared_distances(latent_vectors)

    # Compute pairwise spectral distances
    spectral_dists = pairwise_squared_distances(spectra)

    # Compute pairwise latent distances
    #latent_dists = tf.reduce_sum(tf.square(tf.expand_dims(latent_vectors, 1) - tf.expand_dims(latent_vectors, 0)), axis=-1)  

    # Compute pairwise spectral distances
    #spectral_dists = tf.reduce_sum(w_prime * tf.square(tf.expand_dims(spectra, 1) - tf.expand_dims(spectra, 0)), axis=-1)

    # Compute S_ij
    S_ij = (latent_dists / tf.cast(tf.shape(latent_vectors)[-1], tf.float32)) - (spectral_dists / tf.cast(tf.shape(spectra)[-1], tf.float32))

    # Compute similarity loss using double sigmoid
    sim_loss = tf.reduce_mean(tf.sigmoid(k1 * (S_ij - k0))) + tf.reduce_mean(tf.sigmoid(-k1 * (S_ij - k0)))

    return sim_loss

def consistency_loss(latent_vectors, latent_augmented, sigma_s=0.1):
    """
    Computes the consistency loss term:
    L_c = 1/N * sum(sigmoid((||s_i - s_aug,i||^2) / (sigma_s^2 * S)) - 0.5)

    where:
    - s_i: original latent vector
    - s_aug,i: latent vector of augmented (redshifted) spectrum
    """

    batch_size = tf.shape(latent_vectors)[0]
    
    # Compute pairwise distances in latent space
    latent_dists = tf.reduce_sum(tf.square(latent_vectors - latent_augmented), axis=-1)

    # Compute consistency loss
    cons_loss = tf.reduce_mean(tf.sigmoid(latent_dists / (sigma_s**2 * tf.cast(tf.shape(latent_vectors)[-1], tf.float32)))) - 0.5

    return cons_loss

def custom_loss(y_true, y_pred, encoder, w_prime=None):
    """
    Custom loss function that dynamically handles training and validation cases.
    If w_prime is None, assume it should be computed dynamically.
    """
    
    # Fidelity loss (Mean Squared Error)
    fid_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # If w_prime is not provided, assume a filler of ones
    if w_prime is None:
        w_prime = tf.ones_like(y_true, dtype=tf.float32)
    
    # Compute latent representations
    latent_vectors = encoder(y_true)
    latent_augmented = encoder(y_pred)

    # Compute similarity loss
    sim_loss = similarity_loss(latent_vectors, y_true, w_prime)
    
    # Compute consistency loss
    cons_loss = consistency_loss(latent_vectors, latent_augmented)

    # Total loss
    total_loss = fid_loss + sim_loss + cons_loss
    return total_loss

### LOSS FUNCTION RELATED ###

def build_encoder(input_shape):
    input_layer = Input(shape=input_shape)

    # Convolutional Layers
    x = Conv1D(filters=128, kernel_size=5, padding='valid')(input_layer)
    x = PReLU()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=256, kernel_size=11, padding='valid')(x)
    x = PReLU()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=512, kernel_size=21, padding='valid')(x)
    x = PReLU()(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Flatten the output from Conv layers
    x = Flatten()(x)

    # Fully Connected Layers
    x = Dense(256)(x)
    x = PReLU()(x)
    x = Dense(128)(x)
    x = PReLU()(x)
    x = Dense(64)(x)
    x = PReLU()(x)

    # Latent Space
    latent_space = Dense(10, name='latent_space')(x)

    return Model(input_layer, latent_space, name='encoder')

def transform_spectrum(inputs):
    rest_spectrum, z = inputs  

    # Debugging
    #tf.print("Rest Spectrum Shape Before Processing:", tf.shape(rest_spectrum))
    #tf.print("Redshift Shape:", tf.shape(z))

    # Ensure correct dtypes
    rest_spectrum = tf.cast(rest_spectrum, dtype=tf.float32)
    z = tf.cast(z, dtype=tf.float32)
    z = tf.reshape(z, (-1, 1))  

    # Redshift transformation
    wave_redshifted = tf.expand_dims(wave_rest, axis=0) * (1 + z)
    #wave_redshifted = tf.clip_by_value(wave_redshifted, wave_obs[0], wave_obs[-1])
    wave_redshifted = tf.sort(wave_redshifted, axis=-1)

    # Expand observed wave grid
    batch_size = tf.shape(rest_spectrum)[0]  
    wave_obs_expanded = tf.tile(tf.expand_dims(wave_obs, axis=0), [batch_size, 1])

    # Squeeze rest_spectrum to expected shape
    rest_spectrum = tf.squeeze(rest_spectrum, axis=-1)  

    # Debugging
    #tf.print("Wave Redshifted Shape:", tf.shape(wave_redshifted))
    #tf.print("Wave Obs Expanded Shape:", tf.shape(wave_obs_expanded))
    #tf.print("Rest Spectrum Shape Before Interpolation:", tf.shape(rest_spectrum))

    # Fix: Ensure proper shape alignment before interpolation
    # New correct redshift transformation:
    batch_size = tf.shape(rest_spectrum)[0]

    # Keep wave_rest fixed (no batch-dependence)
    wave_rest_fixed = wave_rest  # shape: [rest_length]

    # Shift observed wavelengths into rest-frame (batch-dependent)
    wave_obs_shifted = tf.expand_dims(wave_obs, axis=0) / (1 + z)  # shape: [batch_size, obs_length]

    # Ensure monotonic increasing order
    wave_obs_shifted = tf.sort(wave_obs_shifted, axis=-1)

    # Prepare rest_spectrum for interpolation
    rest_spectrum = tf.reshape(rest_spectrum, [batch_size, rest_length])

    # Perform correct interpolation
    obs_spectrum = tfp.math.batch_interp_rectilinear_nd_grid(
        x=tf.expand_dims(wave_obs_shifted, axis=-1),  # [batch_size, obs_length, 1]
        x_grid_points=(wave_rest_fixed,),             # [rest_length] fixed grid
        y_ref=rest_spectrum,                          # [batch_size, rest_length]
        axis=1
    )

    # Final reshape
    obs_spectrum = tf.reshape(obs_spectrum, [batch_size, obs_length, 1])

    # Debugging final output shape
    #tf.print("Final Transform Spectrum Output Shape:", tf.shape(obs_spectrum))

    return obs_spectrum  

def build_decoder(latent_dim, rest_length):
    latent_input = Input(shape=(latent_dim,))
    z = Input(shape=(1,), name='z')

    # Fully Connected Layers
    x = Dense(64)(latent_input)
    x = PReLU()(x)
    x = Dense(256)(x)
    x = PReLU()(x)
    x = Dense(1024)(x)
    x = PReLU()(x)

    # generate rest-frame
    x = Dense(rest_length)(x)
    x = PReLU()(x)

    # Reshape BEFORE interpolation (batch_size, rest_length)
    x = Reshape((rest_length, 1))(x)  # ✅ Ensure proper shape for interpolation

    # Interpolate & downsample to observed-frame
    x = Lambda(transform_spectrum)([x, z])  # ✅ Downsample from rest_length → output_dim

    return Model([latent_input, z], x, name='decoder')

def build_autoencoder(input_shape, latent_dim:int=10):
    encoder = build_encoder(input_shape)
    decoder = build_decoder(latent_dim=latent_dim, rest_length=rest_length)

    input_layer = Input(shape=input_shape)
    latent_space = encoder(input_layer)
    
    z = Input(shape=(1,), name='z')
    reconstructed_output = decoder([latent_space, z])
    #print("Reconstructed Output Shape (symbolic):", reconstructed_output.shape)

    return Model([input_layer, z], reconstructed_output, name='autoencoder'), encoder

autoencoder, encoder = build_autoencoder(input_shape=(obs_length, 1), latent_dim=10)

autoencoder.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, encoder))
#autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

#print(f"y_train shape: {y_train.shape}, z_train shape: {z_train.shape}")
#print(f"y_test shape: {y_test.shape}, z_test shape: {z_test.shape}")

history = autoencoder.fit(
    [y_train, z_train],  # Ensure both inputs have the same batch size
    y_train,  # Target remains y_train
    epochs=10,
    shuffle=True,
    validation_data=([y_test, z_test], y_test)  # Validation set must match input format
)

autoencoder.save_weights('/burg/home/tjk2147/src/GitHub/qsoml/autoencoder_weights.h5')
#autoencoder.save('autoencoder_model')

import os 
import pandas as pd 

y_pred = autoencoder.predict([y_test, z_test])

history_df = pd.DataFrame()
history_df['loss'] = history.history['loss']
history_df['val_loss'] = history.history['val_loss']
#history_df['mse'] = history.history['mse']
#history_df['val_mse'] = history.history['val_mse']

history_df.to_csv('/burg/home/tjk2147/src/GitHub/qsoml/history_df.csv', index=False)


for i in range(3):
    prediction_df = pd.DataFrame()
    prediction_df['wave_obs'] = wave_obs.numpy()
    prediction_df['y_test'] = y_test[i].flatten()
    prediction_df['y_pred'] = y_pred[i].flatten()

    prediction_df.to_csv(f'/burg/home/tjk2147/src/GitHub/qsoml/prediction_df_{i}.csv', index=False)

"""
import os 
import matplotlib.pyplot as plt 

plot_dir = '/burg/home/tjk2147/src/GitHub/qsoml/figures'

# Generate predictions
y_pred = autoencoder.predict([y_test, z_test])

# Epoch vs. Loss
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch vs. Loss')
plt.legend()
plt.savefig(os.path.join(plot_dir, 'losshistory.png'))

# Epoch vs. MSE
plt.figure(figsize=(8, 4))
plt.plot(history.history['mse'], label='Train MSE')
plt.plot(history.history['val_mse'], label='Validation MSE', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Epoch vs. MSE')
plt.legend()
plt.savefig(os.path.join(plot_dir, 'msehistory.png'))

num_plots = 3

for i in range(num_plots):
    fig, ax = plt.subplots(figsize=(6, 3)) 
    ax.plot(wave_obs.numpy(), y_test[i].flatten(), label='Actual', linewidth=1, color='black')
    ax.plot(wave_obs.numpy(), y_pred[i].flatten(), label='Predicted', linewidth=1, color='red')
    ax.set_xlabel('Observed Wavelength')
    ax.set_ylabel('Normalized Flux')
    ax.set_title(f'Spectrum Comparison (Test Sample {i})')
    ax.legend()

    fig.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'actualpred-{i}.png'))
"""