import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import os
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, PReLU, Input, Reshape, Lambda
from tensorflow.keras.models import Model
#import matplotlib.pyplot as plt 

### --- Data Loading --- ###

def sample_delta_z(z_true, z_min=1.5, z_max=2.2, delta_z_max=0.5):
    z_true = z_true.flatten()
    z_aug = []
    for z_i in z_true:
        valid = False
        while not valid:
            delta_z = np.random.uniform(0.0, delta_z_max)
            z_new = z_i + delta_z
            if z_min <= z_new <= z_max:
                valid = True
        z_aug.append(z_new)
    return np.array(z_aug, dtype=np.float32).reshape(-1, 1)

def train_test_split(X: tuple, test_prop: float = 0.1):
    n_samples = len(X[0])
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    test_size = int(n_samples * test_prop)
    test_idx, train_idx = idx[:test_size], idx[test_size:]
    split_data = []
    for data in X:
        data = np.array(data)
        if len(data.shape) == 1:
            split_data.append(data[train_idx].reshape(-1, 1))
            split_data.append(data[test_idx].reshape(-1, 1))
        else:
            split_data.append(data[train_idx])
            split_data.append(data[test_idx])
    return tuple(split_data)

def load_data(data_dir: str, z_range:list, rest_norm_window:list=[3020, 3100]):
    r'''
    
    Rest Frame Range for Normalization: from COMPOSITE QUASAR SPECTRA FROM THE SLOAN DIGITAL SKY SURVEY (BERK 2001)
    https://iopscience.iop.org/article/10.1086/321167/meta

    '''
    import numpy as np
    import pandas as pd
    import os

    z_df = pd.read_csv(os.path.join(data_dir, 'zkey.csv'))
    z_map = dict(zip(z_df['name'], z_df['z'].astype(np.float32)))

    # --- Step 1: Get z range ---
    z_min = z_range[0]
    z_max = z_range[1]

    # --- Step 2: Define rest-frame clean window ---
    clean_rest_min = rest_norm_window[0]
    clean_rest_max = rest_norm_window[1]

    # --- Step 3: Compute observed-frame window ---
    obs_window_min = clean_rest_min * (1 + z_min)
    obs_window_max = clean_rest_max * (1 + z_max)

    # --- Step 4: Load spectra ---
    Y, zs_sorted, IVARS = [], [], []
    for file in os.listdir(data_dir):
        if file.endswith('.csv') and 'spec' in file:
            name = file.split('.')[0]
            if name in z_map:
                zs_sorted.append(z_map[name])
                spectrum_df = pd.read_csv(os.path.join(data_dir, file))
                y = spectrum_df['y'].to_numpy(dtype=np.float32)
                ivar = spectrum_df['ivar'].to_numpy(dtype=np.float32)
                x = spectrum_df['x'].to_numpy(dtype=np.float32)

                # Compute normalization factor
                mask = (x >= obs_window_min) & (x <= obs_window_max)
                if np.sum(mask) == 0:
                    norm_factor = np.median(y)  # fallback
                else:
                    norm_factor = np.median(y[mask])

                Y.append(y / norm_factor)
                IVARS.append(ivar)
    Y = np.array(Y, dtype=np.float32)
    IVARS = np.array(IVARS, dtype=np.float32)
    zs_sorted = np.array(zs_sorted, dtype=np.float32)

    y_train, y_test, z_train, z_test, ivar_train, ivar_test = train_test_split((Y, zs_sorted, IVARS))
    return y_train, y_test, z_train, z_test, ivar_train, ivar_test

### --- Model --- ###

def similarity_loss(latent_vectors, rest_spectra, ivar_batch):
    def pairwise_squared_distances(X):
        X_sq = tf.reduce_sum(tf.square(X), axis=-1, keepdims=True)
        pairwise_sq_dists = X_sq + tf.transpose(X_sq) - 2 * tf.matmul(X, X, transpose_b=True)
        return pairwise_sq_dists

    S = tf.cast(tf.shape(latent_vectors)[-1], tf.float32)
    M = tf.cast(tf.shape(rest_spectra)[-1], tf.float32)

    # Latent space pairwise distances (normalized by dim)
    latent_dists = pairwise_squared_distances(latent_vectors) / S

    # Spectral space pairwise distances (w'-weighted)
    spectral_dists = pairwise_squared_distances(rest_spectra)
    ivar_weights = tf.reduce_mean(ivar_batch, axis=-1)  # shape (N,)
    w_outer = tf.expand_dims(ivar_weights, 1) * tf.expand_dims(ivar_weights, 0)
    spectral_term = (w_outer * spectral_dists) / M

    # Z-score normalize both distance matrices
    latent_dists_norm = (latent_dists - tf.reduce_mean(latent_dists)) / (tf.math.reduce_std(latent_dists) + 1e-8)
    spectral_term_norm = (spectral_term - tf.reduce_mean(spectral_term)) / (tf.math.reduce_std(spectral_term) + 1e-8)

    # Similarity loss = mean squared error between normalized matrices
    sim_loss = tf.reduce_mean(tf.square(latent_dists_norm - spectral_term_norm))

    return sim_loss



### THIS VERSION OF SIMILIARITY_LOSS() WAS ONLY 0.5 ALWAYS!
"""def similarity_loss(latent_vectors, rest_spectra, ivar_batch, k0=2.5, k1=10.0):
    def pairwise_squared_distances(X):
        X_sq = tf.reduce_sum(tf.square(X), axis=-1, keepdims=True)
        pairwise_sq_dists = X_sq + tf.transpose(X_sq) - 2 * tf.matmul(X, X, transpose_b=True)
        return pairwise_sq_dists

    N = tf.shape(latent_vectors)[0]
    S = tf.cast(tf.shape(latent_vectors)[-1], tf.float32)
    M = tf.cast(tf.shape(rest_spectra)[-1], tf.float32)

    # Latent distance
    latent_dists = pairwise_squared_distances(latent_vectors) / S

    # Spectral distance (w' weighted)
    spectral_dists = pairwise_squared_distances(rest_spectra)
    ivar_weights = tf.reduce_mean(ivar_batch, axis=-1)  # shape (N,)
    w_outer = tf.expand_dims(ivar_weights, 1) * tf.expand_dims(ivar_weights, 0)  # shape (N, N)
    spectral_term = (w_outer * spectral_dists) / M

    # Similarity metric
    S_ij = latent_dists - spectral_term

    # Similarity loss (double sigmoid)
    sim_loss = tf.reduce_mean(tf.sigmoid(k1 * (S_ij - k0))) + tf.reduce_mean(tf.sigmoid(-k1 * (S_ij - k0)))
    sim_loss /= 2.0  # Optional, for scale consistency

    return sim_loss"""

def consistency_loss(latent_vectors, latent_augmented, sigma_s=0.1):
    latent_dists = tf.reduce_sum(tf.square(latent_vectors - latent_augmented), axis=-1)
    S = tf.cast(tf.shape(latent_vectors)[-1], tf.float32)
    L_c = tf.reduce_mean(tf.sigmoid(latent_dists / (sigma_s**2 * S))) - 0.5
    return L_c

def custom_loss(y_true, y_pred, latent_vectors, latent_augmented, rest_spectra, ivar_batch=None):
    fid_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    if ivar_batch is None: 
        ivar_batch = tf.ones_like(rest_spectra, dtype=tf.float32)
    
    ivar_weights = tf.reduce_mean(ivar_batch, axis=-1)
    w_prime = tf.expand_dims(ivar_weights, 1) * tf.expand_dims(ivar_weights, 0)
    sim_loss = similarity_loss(latent_vectors, rest_spectra, w_prime)

    cons_loss = consistency_loss(latent_vectors, latent_augmented)
    total_loss = fid_loss + sim_loss + cons_loss
    return fid_loss, sim_loss, cons_loss, total_loss

def build_encoder(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv1D(128, 5, padding='valid')(input_layer)
    x = PReLU()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(256, 11, padding='valid')(x)
    x = PReLU()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(512, 21, padding='valid')(x)
    x = PReLU()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = PReLU()(x)
    x = Dense(128)(x)
    x = PReLU()(x)
    x = Dense(64)(x)
    x = PReLU()(x)
    latent_space = Dense(10, name='latent_space')(x)
    return Model(input_layer, latent_space, name='encoder')

def transform_spectrum(inputs):
    rest_spectrum, z = inputs
    rest_spectrum = tf.cast(rest_spectrum, dtype=tf.float32)
    z = tf.cast(z, dtype=tf.float32)
    z = tf.reshape(z, (-1, 1))

    batch_size = tf.shape(rest_spectrum)[0]
    wave_obs_shifted = tf.expand_dims(wave_obs, axis=0) / (1 + z)

    rest_spectrum = tf.squeeze(rest_spectrum, axis=-1)
    rest_spectrum = tf.reshape(rest_spectrum, [batch_size, rest_length])

    obs_spectrum = tfp.math.batch_interp_rectilinear_nd_grid(
        x=tf.expand_dims(wave_obs_shifted, axis=-1),
        x_grid_points=(wave_rest,),
        y_ref=rest_spectrum,
        axis=1
    )

    obs_spectrum = tf.reshape(obs_spectrum, [batch_size, obs_length, 1])
    return obs_spectrum

def build_decoder(latent_dim, rest_length):
    latent_input = Input(shape=(latent_dim,))
    z = Input(shape=(1,), name='z')
    x = Dense(64)(latent_input)
    x = PReLU()(x)
    x = Dense(256)(x)
    x = PReLU()(x)
    x = Dense(1024)(x)
    x = PReLU()(x)
    x = Dense(rest_length)(x)
    x = PReLU()(x)

    # Rest Frame
    x = Reshape((rest_length, 1))(x)
    
    # Back to Observed Frame
    x = Lambda(transform_spectrum)([x, z])
    return Model([latent_input, z], x, name='decoder')

def build_autoencoder(input_shape, latent_dim=10):
    encoder = build_encoder(input_shape)
    decoder = build_decoder(latent_dim, rest_length)
    input_layer = Input(shape=input_shape)
    latent_space = encoder(input_layer)
    z = Input(shape=(1,), name='z')
    reconstructed_output = decoder([latent_space, z])
    autoencoder = Model([input_layer, z], reconstructed_output, name='autoencoder')
    return autoencoder, encoder, decoder

def precompute_z_aug(z_train, num_epochs, z_min=1.5, z_max=2.2, delta_z_max=0.5):
    z_aug_all = []
    for _ in range(num_epochs):
        z_aug_epoch = sample_delta_z(z_train, z_min=z_min, z_max=z_max, delta_z_max=delta_z_max)
        z_aug_all.append(z_aug_epoch)
    return np.stack(z_aug_all, axis=0)

### --- Main --- ###

observed_range = [3600, 10300]
z_range = [1.5, 2.2]

data_dir = '/burg/home/tjk2147/src/GitHub/qsoml/data/csv-batch'
y_train, y_test, z_train, z_test, ivar_train, ivar_test = load_data(data_dir, z_range=z_range)
print(f"Loaded data: {y_train.shape[0]} train samples, {y_test.shape[0]} val samples")

wave_rest_min = observed_range[0] / (1 + z_range[1])
wave_rest_max = observed_range[1] / (1 + z_range[0])

obs_length = len(y_train[0])
upsample_factor = 2
rest_length = upsample_factor * obs_length

wave_rest = tf.cast(tf.linspace(wave_rest_min, wave_rest_max, rest_length), dtype=tf.float32)
wave_obs = tf.cast(tf.linspace(observed_range[0], observed_range[1], obs_length), dtype=tf.float32)

autoencoder, encoder, decoder = build_autoencoder(input_shape=(obs_length, 1), latent_dim=10)

# Sub-model to extract restframe spectrum
decoder_fc = Model(decoder.input, decoder.layers[-2].output)

num_epochs = 20
batch_size = 128
z_aug_all = precompute_z_aug(z_train, num_epochs)

optimizer = tf.keras.optimizers.Adam()
history = {"loss": [], "val_loss": []}
history["fid_loss"] = []
history["sim_loss"] = []
history["cons_loss"] = []

@tf.function
def train_step(y_batch, z_batch, z_aug_batch, ivar_batch, batch_idx=0):
    y_batch = tf.expand_dims(y_batch, axis=-1)  # Ensure shape (batch, obs_length, 1)
    with tf.GradientTape() as tape:
        latent_vectors = encoder(y_batch, training=True)
        y_pred = decoder([latent_vectors, z_batch], training=True)
        rest_spectra = decoder_fc([latent_vectors, z_batch], training=True)
        latent_augmented = encoder(decoder([latent_vectors, z_aug_batch], training=True), training=True)
        fid_loss, sim_loss, cons_loss, total_loss = custom_loss(y_batch, y_pred, latent_vectors, latent_augmented, rest_spectra, ivar_batch)
    gradients = tape.gradient(total_loss, autoencoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
    return fid_loss, sim_loss, cons_loss, total_loss

@tf.function
def val_step(y_batch, z_batch, z_aug_batch, ivar_batch):
    y_batch = tf.expand_dims(y_batch, axis=-1)  # Ensure shape (batch, obs_length, 1)
    latent_vectors = encoder(y_batch, training=False)
    y_pred = decoder([latent_vectors, z_batch], training=False)
    rest_spectra = decoder_fc([latent_vectors, z_batch], training=False)
    latent_augmented = encoder(decoder([latent_vectors, z_aug_batch], training=False), training=False)
    _, _, _, loss = custom_loss(y_batch, y_pred, latent_vectors, latent_augmented, rest_spectra, ivar_batch)
    return loss

def sample_z_aug_for_val(z_val, delta_z_max=0.5, z_min=1.5, z_max=2.2):
    return sample_delta_z(z_val, z_min=z_min, z_max=z_max, delta_z_max=delta_z_max)

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    z_aug_epoch = z_aug_all[epoch]

    epoch_fid_loss = []
    epoch_sim_loss = []
    epoch_cons_loss = []
    epoch_total_loss = []

    indices = np.arange(len(y_train))
    np.random.shuffle(indices)
    y_train_shuffled = y_train[indices]
    z_train_shuffled = z_train[indices]
    z_aug_epoch_shuffled = z_aug_epoch[indices]
    ivar_train_shuffled = ivar_train[indices]

    for i in range(0, len(y_train), batch_size):
        y_batch = tf.convert_to_tensor(y_train_shuffled[i:i+batch_size], dtype=tf.float32)
        z_batch = tf.convert_to_tensor(z_train_shuffled[i:i+batch_size], dtype=tf.float32)
        z_aug_batch = tf.convert_to_tensor(z_aug_epoch_shuffled[i:i+batch_size], dtype=tf.float32)
        ivar_batch = tf.convert_to_tensor(ivar_train_shuffled[i:i+batch_size], dtype=tf.float32)

        fid_loss, sim_loss, cons_loss, total_loss = train_step(y_batch, z_batch, z_aug_batch, ivar_batch)

        tf.print(f"    Batch {i // batch_size + 1}: Sim Loss = {sim_loss}")

        epoch_fid_loss.append(fid_loss.numpy())
        epoch_sim_loss.append(sim_loss.numpy())
        epoch_cons_loss.append(cons_loss.numpy())
        epoch_total_loss.append(total_loss.numpy())

        if (i // batch_size + 1) % 25 == 0:
            print(f"  Batch {i//batch_size + 1}: Fid: {fid_loss.numpy():.4f}, Sim: {sim_loss.numpy():.4f}, Cons: {cons_loss.numpy():.4f}, Total: {total_loss.numpy():.4f}")

    # ---- Modified Validation Loop ----
    val_losses = []
    for i in range(0, len(y_test), batch_size):
        y_batch = tf.convert_to_tensor(y_test[i:i+batch_size], dtype=tf.float32)
        z_batch = tf.convert_to_tensor(z_test[i:i+batch_size], dtype=tf.float32)
        ivar_batch = tf.convert_to_tensor(ivar_test[i:i+batch_size], dtype=tf.float32)

        # Sample new z_aug_batch for validation
        z_aug_batch_np = sample_z_aug_for_val(z_batch.numpy())
        z_aug_batch = tf.convert_to_tensor(z_aug_batch_np, dtype=tf.float32)

        val_loss = val_step(y_batch, z_batch, z_aug_batch, ivar_batch)
        val_losses.append(val_loss.numpy())

    avg_val_loss = np.mean(val_losses)

    history["fid_loss"].append(np.mean(epoch_fid_loss))
    history["sim_loss"].append(np.mean(epoch_sim_loss))
    history["cons_loss"].append(np.mean(epoch_cons_loss))
    history["loss"].append(np.mean(epoch_total_loss))

    history["val_loss"].append(avg_val_loss)

    print(f"Epoch {epoch+1} Summary → Train Loss: {np.mean(epoch_total_loss):.4f} | Fid: {np.mean(epoch_fid_loss):.4f} | Sim: {np.mean(epoch_sim_loss):.4f} | Cons: {np.mean(epoch_cons_loss):.4f} | Val Loss: {avg_val_loss:.4f}")

### PLOT HISTORY OF DECOMPOSED TRAINING LOSS ### 

def save_loss_history(history, save_dir:str='/burg/home/tjk2147/src/GitHub/qsoml/results/data-for-plots'):
    history_df = pd.DataFrame()
    history_df['epoch'] = range(1, len(history["loss"]) + 1)
    history_df['L_fid'] = history["fid_loss"]
    history_df['L_sim'] = history["sim_loss"]
    history_df['L_c'] = history['cons_loss']
    history_df['total_loss'] = history['loss']
    history_df['val_loss'] = history['val_loss']

    history_df.to_csv(f"{save_dir}/history.csv", index=False)
 
save_loss_history(history)

# Optional save
# autoencoder.save_weights('spender_weights.h5')

### INVESTIGATE REST FRAME ###

def save_rest_frame_data(y_train, z_train, sample_idx, save_dir:str='/burg/home/tjk2147/src/GitHub/qsoml/results/data-for-plots'): 
    import os 
    # Prepare inputs

    y_sample = y_train[sample_idx]
    z_sample = z_train[sample_idx]


    y_input = tf.expand_dims(tf.expand_dims(y_sample, axis=0), axis=-1)
    z_input = tf.constant([[z_sample]], dtype=tf.float32)

    # Encode
    latent_vec = encoder(y_input, training=False)

    # Get rest-frame spectrum
    rest_spectrum = decoder_fc([latent_vec, z_input], training=False).numpy().squeeze()

    restframe_df = pd.DataFrame()
    restframe_df['x'] = wave_rest
    restframe_df['y'] = rest_spectrum

    restframe_df.to_csv(os.path.join(save_dir, f'restframe_prediction_{sample_idx}-z={z_sample:.2f}.csv'), index=False)

for i in [1, 10, 3, 100, 21]: 
    save_rest_frame_data(y_train, z_train, sample_idx=i)