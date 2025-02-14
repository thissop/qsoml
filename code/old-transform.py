
'''def transform_spectrum(inputs):
    """
    Transforms rest-frame spectra to observed-frame spectra via redshifting & interpolation.
    Uses global `wave_rest` and `wave_obs` to avoid recomputation.
    """
    rest_spectrum, z = inputs

    # Ensure all inputs are float32
    rest_spectrum = tf.cast(rest_spectrum, dtype=tf.float32)
    z = tf.cast(z, dtype=tf.float32)
    wave_redshifted = tf.cast(wave_rest, dtype=tf.float32)[None, :] * (1 + z)

    obs_spectrum = tfp.math.batch_interp_rectilinear_nd_grid(
        x=tf.expand_dims(wave_obs, axis=-1),  # Target wavelengths, shape (num_points, 1)
        x_grid_points=(wave_redshifted,),  # Redshifted rest-frame wavelengths, tuple required
        y_ref=rest_spectrum,  # Rest-frame spectra
        axis=-1
    )

    return obs_spectrum'''

'''
def transform_spectrum(inputs):
    """
    Transforms rest-frame spectra to observed-frame spectra via redshifting & interpolation.
    Uses global `wave_rest` and `wave_obs` to avoid recomputation.
    """
    rest_spectrum, z = inputs

    # Ensure correct data types
    rest_spectrum = tf.cast(rest_spectrum, dtype=tf.float32)
    z = tf.cast(z, dtype=tf.float32)

    # Broadcast redshift transformation properly
    wave_redshifted = tf.expand_dims(wave_rest, axis=0) * (1 + tf.expand_dims(z, axis=1))

    # Ensure wave_redshifted stays within bounds of wave_obs
    wave_redshifted = tf.clip_by_value(wave_redshifted, wave_obs[0], wave_obs[-1])

    # Perform interpolation
    obs_spectrum = tfp.math.batch_interp_rectilinear_nd_grid(
        x=tf.expand_dims(wave_obs, axis=0),  # Shape (1, num_points)
        x_grid_points=(wave_redshifted,),  # Shape (batch_size, rest_length)
        y_ref=rest_spectrum,  # Shape (batch_size, rest_length)
        axis=-1
    )

    return obs_spectrum

'''

'''
def transform_spectrum(inputs):
    rest_spectrum, z = inputs

    # Ensure correct data types
    rest_spectrum = tf.cast(rest_spectrum, dtype=tf.float32)  # Expected shape (batch_size, rest_length, 1)
    z = tf.cast(z, dtype=tf.float32)
    z = tf.reshape(z, (-1, 1))  # Ensure shape (batch_size, 1)

    # Expand wave_rest for broadcasting (batch_size, rest_length)
    wave_redshifted = tf.expand_dims(wave_rest, axis=0) * (1 + z)

    # Clip values within bounds
    wave_redshifted = tf.clip_by_value(wave_redshifted, wave_obs[0], wave_obs[-1])

    # Ensure wave_obs has batch dimension
    wave_obs_expanded = tf.tile(tf.expand_dims(wave_obs, axis=0), [tf.shape(rest_spectrum)[0], 1])

    # **Fix shape of rest_spectrum to match expected input** (batch_size, rest_length)
    rest_spectrum = tf.squeeze(rest_spectrum, axis=-1)  # Convert (batch_size, rest_length, 1) → (batch_size, rest_length)

    # Perform interpolation
    obs_spectrum = tfp.math.batch_interp_rectilinear_nd_grid(
        x=wave_obs_expanded,  # Shape (batch_size, obs_length)
        x_grid_points=(wave_redshifted,),  # Shape (batch_size, rest_length)
        y_ref=rest_spectrum,  # ✅ Fixed shape: (batch_size, rest_length)
        axis=-1
    )

    return obs_spectrum  
'''