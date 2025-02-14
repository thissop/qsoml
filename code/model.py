import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, PReLU, Input, Reshape, Lambda
from tensorflow.keras.models import Model

def train_test_split(X: tuple, test_prop: float = 0.1):
    import numpy as np

    n_samples = len(X[0])  
    idx = np.arange(n_samples)  
    np.random.shuffle(idx)

    test_size = int(n_samples * test_prop)  
    test_idx, train_idx = idx[:test_size], idx[test_size:]

    split_data = []
    for data in X:
        data = np.array(data)
        if len(data.shape) == 1: # scalars (e.g. z)
            split_data.append(data[train_idx].reshape(-1, 1))  # Ensure correct shape (N,1)
            split_data.append(data[test_idx].reshape(-1, 1))
        else: # vectors (e.g. spectra)
            split_data.append(data[train_idx])
            split_data.append(data[test_idx])

    return tuple(split_data)

def load_data(data_dir: str):
    import pandas as pd
    import os
    import numpy as np

    z_df = pd.read_csv(os.path.join(data_dir, 'zkey.csv'))
    z_map = dict(zip(z_df['name'], z_df['z'].astype(np.float32)))  
    Y, zs_sorted = [], []

    for file in os.listdir(data_dir):
        if file.endswith('.csv') and 'spec' in file:
            name = file.split('.')[0]
            if name in z_map:  
                zs_sorted.append(z_map[name])  

                spectrum_df = pd.read_csv(os.path.join(data_dir, file))
                y = spectrum_df['y'].to_numpy(dtype=np.float32)  # Ensure float32
                Y.append(y / np.median(y))  # Normalize

    Y = np.array(Y, dtype=np.float32)
    zs_sorted = np.array(zs_sorted, dtype=np.float32)  # Ensure 1D 

    y_train, y_test, z_train, z_test = train_test_split((Y, zs_sorted))

    return y_train, y_test, z_train, z_test

data_dir = 'data/csv-batch'
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
def build_encoder(input_shape):
    input_layer = Input(shape=input_shape)

    x = Conv1D(filters=128, kernel_size=5, padding='valid')(input_layer)
    x = PReLU()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=256, kernel_size=11, padding='valid')(x)
    x = PReLU()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=512, kernel_size=21, padding='valid')(x)
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

    # Debug input
    tf.print("Rest Spectrum Shape Before Processing:", tf.shape(rest_spectrum))
    tf.print("Redshift Shape:", tf.shape(z))

    # Ensure correct dtypes
    rest_spectrum = tf.cast(rest_spectrum, dtype=tf.float32)
    z = tf.cast(z, dtype=tf.float32)
    z = tf.reshape(z, (-1, 1))  

    # Redshift transformation
    wave_redshifted = tf.expand_dims(wave_rest, axis=0) * (1 + z)
    wave_redshifted = tf.clip_by_value(wave_redshifted, wave_obs[0], wave_obs[-1])

    # Expand observed wave grid
    batch_size = tf.shape(rest_spectrum)[0]  
    wave_obs_expanded = tf.tile(tf.expand_dims(wave_obs, axis=0), [batch_size, 1])

    # Squeeze rest_spectrum to expected shape
    rest_spectrum = tf.squeeze(rest_spectrum, axis=-1)  

    # Debug before interpolation
    tf.print("Wave Redshifted Shape:", tf.shape(wave_redshifted))
    tf.print("Wave Obs Expanded Shape:", tf.shape(wave_obs_expanded))
    tf.print("Rest Spectrum Shape Before Interpolation:", tf.shape(rest_spectrum))

    # Ensure proper shape before interpolation
    rest_spectrum = tf.reshape(rest_spectrum, [batch_size, rest_length])
    wave_redshifted = tf.reshape(wave_redshifted, [batch_size, rest_length])
    wave_obs_expanded = tf.reshape(wave_obs_expanded, [batch_size, obs_length])

    # Interpolation
    obs_spectrum = tfp.math.batch_interp_rectilinear_nd_grid(
        x=wave_obs_expanded,  
        x_grid_points=(wave_redshifted,),  
        y_ref=rest_spectrum,  
        axis=-1
    )

    # Ensure the final output is (batch_size, obs_length, 1)
    obs_spectrum = tf.reshape(obs_spectrum, [batch_size, obs_length, 1])

    # Debug final output
    tf.print("Final Transform Spectrum Output Shape:", tf.shape(obs_spectrum))

    return obs_spectrum  

def build_decoder(latent_dim, output_dim, rest_length):
    latent_input = Input(shape=(latent_dim,))
    z = Input(shape=(1,), name='z')

    x = Dense(64)(latent_input)
    x = PReLU()(x)
    x = Dense(256)(x)
    x = PReLU()(x)
    x = Dense(1024)(x)
    x = PReLU()(x)

    # Generate rest-frame
    x = Dense(rest_length)(x)
    x = PReLU()(x)

    # Reshape before interpolation (batch_size, rest_length)
    x = Reshape((rest_length, 1))(x)  

    # Interpolate & downsample to observed-frame (from rest_length to output_dim)
    x = Lambda(transform_spectrum)([x, z])  

    return Model([latent_input, z], x, name='decoder')

def build_autoencoder(input_shape, latent_dim:int=10):
    encoder = build_encoder(input_shape)
    decoder = build_decoder(latent_dim=latent_dim, output_dim=input_shape[0], rest_length=rest_length)

    input_layer = Input(shape=input_shape)
    latent_space = encoder(input_layer)
    
    z = Input(shape=(1,), name='z')
    reconstructed_output = decoder([latent_space, z])
    tf.print("Reconstructed Output Shape:", tf.shape(reconstructed_output))

    return Model([input_layer, z], reconstructed_output, name='autoencoder')

autoencoder = build_autoencoder(input_shape=(obs_length, 1), latent_dim=10)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

print(f"y_train shape: {y_train.shape}, z_train shape: {z_train.shape}")
print(f"y_test shape: {y_test.shape}, z_test shape: {z_test.shape}")

history = autoencoder.fit(
    [y_train, z_train], 
    y_train,  
    epochs=5,
    shuffle=True,
    validation_data=([y_test, z_test], y_test) 
)