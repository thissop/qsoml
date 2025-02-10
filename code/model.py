import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, PReLU, Input, Reshape, Lambda
from tensorflow.keras.models import Model

def train_test_split(X: tuple, test_prop: float = 0.1):
    import numpy as np

    idx = np.arange(len(X[0]))  
    np.random.shuffle(idx)

    test_size = int(len(X[0]) * test_prop)  
    test_idx, train_idx = idx[:test_size], idx[test_size:]

    split_data = [np.array(data)[train_idx] for data in X] + [np.array(data)[test_idx] for data in X]
    
    return tuple(split_data)

def load_data(data_dir: str):
    import pandas as pd
    import os
    import numpy as np

    z_df = pd.read_csv(os.path.join(data_dir, 'zkey.csv'))
    names = z_df['name'].to_numpy()
    zs = z_df['z'].to_numpy(dtype=np.float32)  # Cast z to float32

    zs_sorted, Y = [], []

    for file in os.listdir(data_dir):
        if 'csv' in file and 'spec' in file:
            spectrum_name = file.split('.')[0]
            z_val = zs[np.argwhere(names == spectrum_name)].flatten().astype(np.float32)  # Ensure float32
            zs_sorted.append(z_val)

            spectrum_df = pd.read_csv(os.path.join(data_dir, file))
            y = spectrum_df['y'].to_numpy(dtype=np.float32)  # Ensure float32
            Y.append(y / np.median(y))  # Normalize

    y_train, y_test, z_train, z_test = train_test_split((Y, zs_sorted))

    # Convert entire dataset in a single step
    return tuple(map(lambda x: np.array(x, dtype=np.float32), (y_train, y_test, z_train, z_test)))

data_dir = '/burg/home/tjk2147/src/GitHub/qsoml/data/csv-batch'
y_train, y_test, z_train, z_test = load_data(data_dir)

observed_range = [3600, 10300]
z_range = [1.5, 2.2]

wave_rest_min = observed_range[0] / (1 + z_range[1])  # Compute min rest-frame wavelength
wave_rest_max = observed_range[1] / (1 + z_range[0]) 

obs_length = len(y_train[0])
upsample_factor = 2
rest_length = upsample_factor*obs_length 

wave_rest = tf.linspace(wave_rest_min, wave_rest_max, num=rest_length)  # Fixed grid
wave_obs = tf.linspace(observed_range[0], observed_range[1], num=obs_length)

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
    """
    Transforms rest-frame spectra to observed-frame spectra via redshifting & interpolation.
    Uses global `wave_rest` and `wave_obs` to avoid recomputation.
    """
    rest_spectrum, z = inputs
    wave_redshifted = wave_rest[None, :] * (1 + z)  # Broadcast for batch

    obs_spectrum = tfp.math.batch_interp_rectilinear_nd_grid(
        x=tf.expand_dims(wave_obs, axis=-1),  # Target wavelengths, shape (num_points, 1)
        x_grid_points=(wave_redshifted,),  # Redshifted rest-frame wavelengths, tuple required
        y_ref=rest_spectrum,  # Rest-frame spectra
        axis=-1
    )

    return obs_spectrum

def build_decoder(latent_dim, output_dim, rest_length):
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

    # interpolate and downsample
    x = Lambda(transform_spectrum)([x, z])

    # Final Reshaping
    x = Reshape((output_dim, 1))(x)  # If necessary, ensure shape is (batch_size, output_dim, 1)

    return Model([latent_input, z], x, name='decoder')

def build_autoencoder(input_shape, latent_dim:int=10):
    encoder = build_encoder(input_shape)
    decoder = build_decoder(latent_dim=latent_dim, output_dim=input_shape[0], rest_length=rest_length)

    input_layer = Input(shape=input_shape)
    latent_space = encoder(input_layer)
    
    z = Input(shape=(1,), name='z')
    reconstructed_output = decoder([latent_space, z])

    return Model([input_layer, z], reconstructed_output, name='autoencoder')

autoencoder = build_autoencoder(input_shape=(obs_length, 1), latent_dim=10)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

history = autoencoder.fit(y_train, y_train, epochs=5, shuffle=True, validation_data=(y_test, y_test))
