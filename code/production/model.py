import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import os
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Reshape, Lambda, LeakyReLU
from tensorflow.keras.models import Model
import time 
from losses import custom_loss

### --- CONFIG --- ###

CONFIG = {
    "observed_range": [3600, 10300],
    "z_range": [1.5, 2.2],
    "rest_norm_window": [3020, 3100],
    "upsample_factor": 2,
    "latent_dim": 10,
    "sigma_s": 0.1,
    "distance_threshold": 0.5,
    "extrap_start_epoch": 5,
    "clipnorm": 5.0,
    "initial_lr": 1e-3,
    "lr_decay_steps": 10,
    "lr_decay_rate": 0.5,
    "batch_size": 128,
    "num_epochs":50,
    "patience": 5,
    "delta_z_max": 0.5,
    "sim_loss_weight_schedule": lambda epoch: min(1.0, epoch / 10.0),
    "extrap_loss_weight_schedule": lambda epoch: min(1.0, epoch / 10.0),
    "data_dir": "/burg/home/tjk2147/src/GitHub/qsoml/data/csv-batch",
    "save_dir": "/burg/home/tjk2147/src/GitHub/qsoml/results/data-for-plots",
    "latent_save_dir":"/burg/home/tjk2147/src/GitHub/qsoml/data"
}

### --- Data Loading --- ###

def sample_delta_z(z_true, z_min=1.5, z_max=2.2, delta_z_max=0.5):
    z_true = z_true.flatten()
    n_samples = len(z_true)
    z_aug = np.zeros_like(z_true)
    done = np.zeros(n_samples, dtype=bool)

    while not np.all(done):
        # Only resample for the ones that failed
        delta_z = np.random.uniform(0.0, delta_z_max, size=n_samples)
        z_new = z_true + delta_z
        valid = (z_new >= z_min) & (z_new <= z_max) & (~done)
        z_aug[valid] = z_new[valid]
        done[valid] = True

    return z_aug.reshape(-1, 1).astype(np.float32)

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
    for file in os.listdir(data_dir)[0:500]:
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

def build_encoder(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv1D(128, 5, padding='valid')(input_layer)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(256, 11, padding='valid')(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(512, 21, padding='valid')(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dense(64)(x)
    x = LeakyReLU(alpha=0.01)(x)
    latent_space = Dense(10, name='latent_space')(x)
    return Model(input_layer, latent_space, name='encoder')

def transform_spectrum(inputs):
    """
    Differentiable redshift and interpolation of rest-frame spectrum to observed frame.
    """
    rest_spectrum, z = inputs
    z = tf.reshape(z, (-1, 1))  # shape (batch_size, 1)

    # Compute shifted observed wavelength grid (rest-frame wavelengths)
    wave_obs_shifted = tf.expand_dims(wave_obs, axis=0) / (1 + z)  # (batch_size, obs_length)

    # Remove last dimension from rest_spectrum
    rest_spectrum = tf.squeeze(rest_spectrum, axis=-1)  # (batch_size, rest_length)

    # Differentiable linear interpolation
    x_min = wave_rest[0]
    x_max = wave_rest[-1]
    dx = (x_max - x_min) / (rest_length - 1)

    indices = (wave_obs_shifted - x_min) / dx
    idx_low = tf.clip_by_value(tf.floor(indices), 0, rest_length - 2)
    idx_high = idx_low + 1
    weight_high = indices - idx_low
    weight_low = 1.0 - weight_high

    idx_low = tf.cast(idx_low, tf.int32)
    idx_high = tf.cast(idx_high, tf.int32)

    def gather_interp(spectrum, idx_l, idx_h, w_l, w_h):
        low_vals = tf.gather(spectrum, idx_l, axis=-1, batch_dims=1)
        high_vals = tf.gather(spectrum, idx_h, axis=-1, batch_dims=1)
        return w_l * low_vals + w_h * high_vals

    interpolated = gather_interp(rest_spectrum, idx_low, idx_high, weight_low, weight_high)
    interpolated = tf.expand_dims(interpolated, axis=-1)  # (batch_size, obs_length, 1)

    return interpolated

def build_decoder(latent_dim, rest_length):
    latent_input = Input(shape=(latent_dim,))
    z = Input(shape=(1,), name='z')
    x = Dense(64)(latent_input)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.01)(x)

    # Rest Frame Dense layer
    rest_frame_dense = Dense(rest_length)(x)
    rest_frame_output = LeakyReLU(alpha=0.01)(rest_frame_dense)
    rest_frame_output = Reshape((rest_length, 1))(rest_frame_output)

    # Back to Observed Frame
    observed_output = Lambda(transform_spectrum)([rest_frame_output, z])

    decoder_model = Model([latent_input, z], observed_output, name='decoder')
    return decoder_model, rest_frame_dense

def build_autoencoder(input_shape, latent_dim=10):
    encoder = build_encoder(input_shape)
    decoder, rest_frame_dense = build_decoder(latent_dim, rest_length)
    input_layer = Input(shape=input_shape)
    z = Input(shape=(1,), name='z')
    latent_space = encoder(input_layer)
    reconstructed_output = decoder([latent_space, z])
    autoencoder = Model([input_layer, z], reconstructed_output, name='autoencoder')
    return autoencoder, encoder, decoder, rest_frame_dense

def precompute_z_aug(z_train, num_epochs, z_min=1.5, z_max=2.2, delta_z_max=0.5):
    z_aug_all = []
    for _ in range(num_epochs):
        z_aug_epoch = sample_delta_z(z_train, z_min=z_min, z_max=z_max, delta_z_max=delta_z_max)
        z_aug_all.append(z_aug_epoch)
    return np.stack(z_aug_all, axis=0)

def compute_restframe_mask_tf(z_batch):
    """
    Given a batch of z values, returns a binary mask indicating 
    which rest-frame pixels would be unobserved.
    """
    # z_batch shape: (batch_size, 1)
    z_batch = tf.reshape(z_batch, (-1, 1))
    rest_waves = tf.expand_dims(wave_rest, axis=0)  # (1, rest_length)
    obs_min = CONFIG["observed_range"][0]
    obs_max = CONFIG["observed_range"][1]

    # Compute observed-frame wavelengths for each z
    obs_waves = rest_waves * (1 + z_batch)  # (batch_size, rest_length)
    mask = tf.cast((obs_waves >= obs_min) & (obs_waves <= obs_max), tf.float32)
    return mask  # shape (batch_size, rest_length)

### --- Main --- ###

observed_range = [3600, 10300]
z_range = [1.5, 2.2]

start_time = time.time()
print("Loading and preprocessing data...")
data_dir = '/burg/home/tjk2147/src/GitHub/qsoml/data/csv-batch'
y_train, y_test, z_train, z_test, ivar_train, ivar_test = load_data(data_dir, z_range=z_range)

z_train_tensor = tf.convert_to_tensor(z_train, dtype=tf.float32)
z_test_tensor = tf.convert_to_tensor(z_test, dtype=tf.float32)

print(f"Loaded data: {y_train.shape[0]} train samples, {y_test.shape[0]} val samples")

wave_rest_min = CONFIG["observed_range"][0] / (1 + CONFIG["z_range"][1])
wave_rest_max = CONFIG["observed_range"][1] / (1 + CONFIG["z_range"][0])

obs_length = len(y_train[0])
upsample_factor = 2
rest_length = upsample_factor * obs_length

wave_rest = tf.cast(tf.linspace(wave_rest_min, wave_rest_max, rest_length), dtype=tf.float32)
wave_obs = tf.cast(tf.linspace(observed_range[0], observed_range[1], obs_length), dtype=tf.float32)

train_rest_masks = compute_restframe_mask_tf(z_train_tensor)
test_rest_masks = compute_restframe_mask_tf(z_test_tensor)

autoencoder, encoder, decoder, rest_frame_dense = build_autoencoder(input_shape=(obs_length, 1), latent_dim=10)
decoder_fc = Model([decoder.input[0], decoder.input[1]], rest_frame_dense)

print(f"Data loaded and preprocessed in {time.time() - start_time:.2f} seconds")

num_epochs = CONFIG["num_epochs"]
batch_size = CONFIG['batch_size']

start_time = time.time()
print("Precomputing z_aug...")
z_aug_all = precompute_z_aug(z_train, num_epochs)
print(f"z_aug precomputed in {time.time() - start_time:.2f} seconds")

decay_steps = (len(y_train) // batch_size) * CONFIG["lr_decay_steps"]
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=CONFIG["initial_lr"],
    decay_steps=decay_steps,
    decay_rate=CONFIG["lr_decay_rate"],
    staircase=True,
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=5.0)

history = {"loss": [], "val_loss": []}
history["fid_loss"] = []
history["sim_loss"] = []
history["cons_loss"] = []
history["extrap_loss"] = []

@tf.function
def train_step(y_batch, z_batch, z_aug_batch, ivar_batch, observed_masks_rest, epoch):
    y_batch = tf.expand_dims(y_batch, axis=-1)
    with tf.GradientTape() as tape:
        latent_vectors = encoder(y_batch, training=True)
        y_pred = decoder([latent_vectors, z_batch], training=True)
        rest_spectra = decoder_fc([latent_vectors, z_batch], training=True)
        y_augmented = decoder([latent_vectors, z_aug_batch], training=True)
        latent_augmented = encoder(y_augmented, training=True)

        fid_loss, sim_loss, cons_loss, extrap_loss, total_loss = custom_loss(
            y_batch, y_pred, latent_vectors, latent_augmented, rest_spectra, ivar_batch,
            observed_masks_rest, epoch, extrap_start_epoch=CONFIG["extrap_start_epoch"]
        )


    gradients = tape.gradient(total_loss, autoencoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
    return fid_loss, sim_loss, cons_loss, extrap_loss, total_loss

@tf.function
def val_step(y_batch, z_batch, z_aug_batch, ivar_batch, observed_masks_rest, epoch):
    y_batch = tf.expand_dims(y_batch, axis=-1)
    latent_vectors = encoder(y_batch, training=False)
    y_pred = decoder([latent_vectors, z_batch], training=False)
    rest_spectra = decoder_fc([latent_vectors, z_batch], training=False)
    y_augmented = decoder([latent_vectors, z_aug_batch], training=False)
    latent_augmented = encoder(y_augmented, training=False)

    _, _, _, _, loss = custom_loss(
        y_batch, y_pred, latent_vectors, latent_augmented, rest_spectra, ivar_batch,
        observed_masks_rest, epoch, extrap_start_epoch=CONFIG["extrap_start_epoch"]
    )
    return loss

def sample_z_aug_for_val(z_val, delta_z_max=0.5, z_min=1.5, z_max=2.2):
    return sample_delta_z(z_val, z_min=z_min, z_max=z_max, delta_z_max=delta_z_max)

patience = 5
best_val_loss = np.inf
epochs_without_improvement = 0

start_train = time.time()

print("Starting Training...")

for epoch in range(num_epochs):
    tf.print(f"\nEpoch {epoch+1}/{num_epochs}")
    z_aug_epoch = z_aug_all[epoch]

    epoch_fid_loss = []
    epoch_sim_loss = []
    epoch_cons_loss = []
    epoch_extrap_loss = []
    epoch_total_loss = []

    indices = np.random.permutation(len(y_train))
    indices = tf.convert_to_tensor(indices, dtype=tf.int32)
    y_train_shuffled = tf.gather(y_train, indices)
    z_train_shuffled = tf.gather(z_train, indices)
    z_aug_epoch_shuffled = tf.gather(z_aug_epoch, indices)
    ivar_train_shuffled = tf.gather(ivar_train, indices)
    observed_masks_rest_shuffled = tf.gather(train_rest_masks, indices)

    sim_loss_weight = CONFIG["sim_loss_weight_schedule"](epoch)
    extrap_loss_weight = CONFIG["extrap_loss_weight_schedule"](epoch)

    for i in range(0, len(y_train), batch_size):
        y_batch = tf.convert_to_tensor(y_train_shuffled[i:i+batch_size], dtype=tf.float32)
        z_batch = tf.convert_to_tensor(z_train_shuffled[i:i+batch_size], dtype=tf.float32)
        z_aug_batch = tf.convert_to_tensor(z_aug_epoch_shuffled[i:i+batch_size], dtype=tf.float32)
        ivar_batch = tf.convert_to_tensor(ivar_train_shuffled[i:i+batch_size], dtype=tf.float32)

        fid_loss, sim_loss, cons_loss, extrap_loss, total_loss = train_step(
            y_batch, z_batch, z_aug_batch, ivar_batch,
            observed_masks_rest_shuffled[i:i+batch_size],
            epoch
        )

        #tf.print(f"    Batch {i // batch_size + 1}: Sim Loss = {sim_loss}")

        epoch_fid_loss.append(fid_loss.numpy())
        epoch_sim_loss.append(sim_loss.numpy())
        epoch_cons_loss.append(cons_loss.numpy())
        epoch_extrap_loss.append(extrap_loss.numpy())
        epoch_total_loss.append(total_loss.numpy())

        #if (i // batch_size + 1) % 25 == 0:
        #    print(f"  Batch {i//batch_size + 1}: Fid: {fid_loss.numpy():.4f}, Sim: {sim_loss.numpy():.4f}, Cons: {cons_loss.numpy():.4f}, Total: {total_loss.numpy():.4f}")

    # ---- Modified Validation Loop ----
    val_losses = []
    for i in range(0, len(y_test), batch_size):
        y_batch = tf.convert_to_tensor(y_test[i:i+batch_size], dtype=tf.float32)
        z_batch = tf.convert_to_tensor(z_test[i:i+batch_size], dtype=tf.float32)
        ivar_batch = tf.convert_to_tensor(ivar_test[i:i+batch_size], dtype=tf.float32)

        # Sample new z_aug_batch for validation
        z_aug_batch_np = sample_z_aug_for_val(z_batch.numpy())
        z_aug_batch = tf.convert_to_tensor(z_aug_batch_np, dtype=tf.float32)

        observed_masks_rest_batch = test_rest_masks[i:i+batch_size]
        val_loss = val_step(
            y_batch, z_batch, z_aug_batch, ivar_batch,
            observed_masks_rest_batch, epoch
        )

        val_losses.append(val_loss.numpy())

    avg_val_loss = np.mean(val_losses)

    history["fid_loss"].append(np.mean(epoch_fid_loss))
    history["sim_loss"].append(np.mean(epoch_sim_loss))
    history["cons_loss"].append(np.mean(epoch_cons_loss))
    history["extrap_loss"].append(np.mean(epoch_extrap_loss))
    history["loss"].append(np.mean(epoch_total_loss))
    history["val_loss"].append(avg_val_loss)

    tf.print(f"Epoch {epoch+1} Summary → Train Loss: {np.mean(epoch_total_loss):.4f} | Fid: {np.mean(epoch_fid_loss):.4f} | Sim: {np.mean(epoch_sim_loss):.4f} | Cons: {np.mean(epoch_cons_loss):.4f} | Extrap: {np.mean(epoch_extrap_loss):.4f} | Val Loss: {avg_val_loss:.4f}")

    # ---- Early stopping logic ----
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
    else:
        if epoch > 25: 
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}: no improvement for {patience} epochs.")
                break

end_train = time.time()
print(f"Total training time: {(end_train - start_train)/60:.2f} minutes.")

### SAVE HISTORY OF DECOMPOSED TRAINING LOSS ### 

def save_loss_history(history, save_dir:str='/burg/home/tjk2147/src/GitHub/qsoml/results/data-for-plots'):
    history_df = pd.DataFrame()
    history_df['epoch'] = range(1, len(history["loss"]) + 1)
    history_df['L_fid'] = history["fid_loss"]
    history_df['L_sim'] = history["sim_loss"]
    history_df['L_c'] = history['cons_loss']
    history_df['L_extrap'] = history['extrap_loss']
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

    restframe_df.to_csv(os.path.join(save_dir, f'restframe_prediction_{sample_idx}.csv'), index=False)

### SAVE RECONSTRUCTED SPECTRA ### 

def save_test_prediction(y_test, z_test, sample_idx, save_dir:str='/burg/home/tjk2147/src/GitHub/qsoml/results/data-for-plots'):
    import os
    # Prepare inputs
    y_sample = y_test[sample_idx]
    z_sample = z_test[sample_idx]

    y_input = tf.expand_dims(tf.expand_dims(y_sample, axis=0), axis=-1)
    z_input = tf.constant([[z_sample]], dtype=tf.float32)

    # Reconstruct using autoencoder
    y_pred = autoencoder([y_input, z_input], training=False).numpy().squeeze()

    # Save to CSV
    predictions_df = pd.DataFrame()
    predictions_df['x'] = wave_obs
    predictions_df['y_test'] = y_sample.squeeze()
    predictions_df['y_predicted'] = y_pred

    print(f'idx: {sample_idx}; z={float(z_sample):.2f}')

    predictions_df.to_csv(os.path.join(save_dir, f'reconstruction_{sample_idx}.csv'), index=False)

for i in [1, 10, 3, 21]: 
    save_rest_frame_data(y_train, z_train, sample_idx=i)
    save_test_prediction(y_test, z_test, sample_idx=i)

# Save latent space for all train and val data
print("Encoding latent space for entire dataset...")
latent_train = encoder(tf.expand_dims(y_train, axis=-1), training=False).numpy()
latent_val = encoder(tf.expand_dims(y_test, axis=-1), training=False).numpy()


np.savez(
    os.path.join(CONFIG["latent_save_dir"], "latent_space.npz"),
    latent_train=latent_train,
    z_train=z_train,
    latent_val=latent_val,
    z_val=z_test
)

print(f"Saved latent space to {CONFIG['latent_save_dir']}/latent_space.npz")
