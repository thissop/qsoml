from matplotlib.pylab import zipf
import tensorflow as tf
from zmq import OUT_BATCH_SIZE
import tensorflow_probability as tfp
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, PReLU, Input, Reshape, Lambda
from tensorflow.keras.models import Model

observed_range = [3550, 10400]
z_range = [1.5, 2.2]

wave_rest_min = observed_range[0] / (1 + z_range[1])  # Compute min rest-frame wavelength
wave_rest_max = observed_range[1] / (1 + z_range[0]) 

obs_length = 1500
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
        x=wave_obs[None, :],  # Target wavelengths
        x_ref=wave_redshifted,  # Redshifted rest-frame wavelengths
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

    # Generate Rest-Frame Spectrum
    x = Dense(rest_length)(x)
    x = PReLU()(x)

    # Apply Redshift and Interpolation with Lambda
    x = Lambda(transform_spectrum)([x, z])

    # Final Reshaping
    x = Reshape((output_dim, 1))(x)  # If necessary, ensure shape is (batch_size, output_dim, 1)

    return Model([latent_input, z], x, name='decoder')

def build_autoencoder(input_shape, latent_dim:int=10):
    encoder = build_encoder(input_shape)
    decoder = build_decoder(latent_dim=latent_dim, output_dim=input_shape[0])

    input_layer = Input(shape=input_shape)
    latent_space = encoder(input_layer)
    
    z = Input(shape=(1,), name='z')
    reconstructed_output = decoder([latent_space, z])

    return Model([input_layer, z], reconstructed_output, name='autoencoder')

autoencoder = build_autoencoder(input_shape=(obs_length, 1), latent_dim=10)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()