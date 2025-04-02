import tensorflow as tf
import warnings
import os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF INFO and WARNING logs

def rest_to_observed_interp(rest_spectra, z_batch, wave_rest, wave_obs):
    """
    Interpolate rest-frame spectrum to observed-frame grid for loss computation.
    """
    z = tf.reshape(z_batch, (-1, 1))
    obs_grid = tf.expand_dims(wave_obs, axis=0) / (1 + z)  # shape (batch_size, obs_length)
    x_min = wave_rest[0]
    x_max = wave_rest[-1]
    dx = (x_max - x_min) / (tf.cast(tf.shape(wave_rest)[0], tf.float32) - 1)

    indices = (obs_grid - x_min) / dx
    idx_low = tf.clip_by_value(tf.floor(indices), 0, tf.cast(tf.shape(wave_rest)[0], tf.float32) - 2)
    idx_high = idx_low + 1
    weight_high = indices - idx_low
    weight_low = 1.0 - weight_high

    idx_low = tf.cast(idx_low, tf.int32)
    idx_high = tf.cast(idx_high, tf.int32)

    def gather_interp(spectrum, idx_l, idx_h, w_l, w_h):
        low_vals = tf.gather(spectrum, idx_l, axis=1, batch_dims=1)
        high_vals = tf.gather(spectrum, idx_h, axis=1, batch_dims=1)
        low_vals = tf.squeeze(low_vals, axis=-1)
        high_vals = tf.squeeze(high_vals, axis=-1)
        result = w_l * low_vals + w_h * high_vals
        return tf.expand_dims(result, axis=-1)


    rest_downsampled = gather_interp(rest_spectra, idx_low, idx_high, weight_low, weight_high)
    return rest_downsampled

def pairwise_squared_distances(X):
    X_sq = tf.reduce_sum(tf.square(X), axis=-1, keepdims=True)
    pairwise_sq_dists = X_sq + tf.transpose(X_sq) - 2 * tf.matmul(X, X, transpose_b=True)
    return pairwise_sq_dists

def consistency_loss(latent_vectors, latent_augmented, sigma_s=0.1):
    S = tf.cast(tf.shape(latent_vectors)[-1], tf.float32)
    latent_dists = tf.reduce_sum(tf.square(latent_vectors - latent_augmented), axis=-1)
    scaled_dists = latent_dists / (S * sigma_s ** 2 + 1e-8)
    L_c = tf.reduce_mean(tf.sigmoid(scaled_dists)) - 0.5
    return L_c

def similarity_loss(latent_vectors, rest_spectra, ivar_batch, observed_masks, eps=1e-8):
    batch_size = tf.shape(latent_vectors)[0]
    S = tf.cast(tf.shape(latent_vectors)[-1], tf.float32)

    # Compute pairwise latent distances
    latent_dists = pairwise_squared_distances(latent_vectors) / S

    # Prepare rest-frame spectra
    rest_spectra_flat = tf.squeeze(rest_spectra, axis=-1)  # (batch_size, rest_length)

    # Compute pairwise observed-frame masks intersection
    obs_i = tf.expand_dims(observed_masks, 1)  # (batch_size, 1, rest_length)
    obs_j = tf.expand_dims(observed_masks, 0)  # (1, batch_size, rest_length)
    intersection = obs_i * obs_j  # (batch_size, batch_size, rest_length)

    # Compute outer product of mean ivar weights
    ivar_mean = tf.reduce_mean(ivar_batch, axis=-1)  # (batch_size,)
    w_outer = tf.expand_dims(ivar_mean, 1) * tf.expand_dims(ivar_mean, 0)  # (batch_size, batch_size)

    # Compute pairwise spectral distances
    spec_i = tf.expand_dims(rest_spectra_flat, 1)  # (batch_size, 1, rest_length)
    spec_j = tf.expand_dims(rest_spectra_flat, 0)  # (1, batch_size, rest_length)
    diff = tf.square(spec_i - spec_j) * intersection  # Masked difference

    spectral_term = tf.reduce_sum(diff, axis=-1) / (tf.reduce_sum(intersection, axis=-1) + eps)
    spectral_term *= w_outer

    # Normalize both distance matrices
    latent_mean = tf.reduce_mean(latent_dists)
    latent_std = tf.math.reduce_std(latent_dists) + eps
    latent_norm = (latent_dists - latent_mean) / latent_std

    spectral_mean = tf.reduce_mean(spectral_term)
    spectral_std = tf.math.reduce_std(spectral_term) + eps
    spectral_norm = (spectral_term - spectral_mean) / spectral_std

    # Ranking loss
    delta = latent_norm - spectral_norm
    sigmoid_pos = tf.sigmoid(5.0 * delta - 2.5)
    sigmoid_neg = tf.sigmoid(-5.0 * delta - 2.5)
    sim_loss = tf.reduce_mean(sigmoid_pos + sigmoid_neg)

    return sim_loss

def extrapolation_loss(latent_vectors, rest_spectra, observed_masks, distance_scale=1.0, eps=1e-8):
    N = tf.shape(latent_vectors)[0]
    L = tf.shape(rest_spectra)[1]

    latent_dists = pairwise_squared_distances(latent_vectors)
    S = tf.cast(tf.shape(latent_vectors)[-1], tf.float32)
    mean_dist = tf.reduce_mean(latent_dists)
    normalized_latent_dists = latent_dists / (S * mean_dist + eps)

    obs_i = tf.expand_dims(observed_masks, 1)  # (batch_size, 1, rest_length)
    obs_j = tf.expand_dims(observed_masks, 0)  # (1, batch_size, rest_length)

    mask_AB = (1.0 - obs_i) * obs_j
    mask_BA = (1.0 - obs_j) * obs_i

    spec_i = tf.expand_dims(tf.squeeze(rest_spectra, axis=-1), 1)  # shape (batch_size, 1, rest_length)
    spec_j = tf.expand_dims(tf.squeeze(rest_spectra, axis=-1), 0)  # shape (1, batch_size, rest_length)

    diff = tf.square(spec_i - spec_j)
    loss_AB = tf.reduce_sum(diff * mask_AB, axis=-1)
    loss_BA = tf.reduce_sum(diff * mask_BA, axis=-1)
    total_loss_matrix = loss_AB + loss_BA

    distance_weight = tf.exp(-distance_scale * normalized_latent_dists)

    total_loss_matrix *= distance_weight

    count = tf.reduce_sum(distance_weight * (tf.reduce_sum(mask_AB + mask_BA, axis=-1))) + eps
    total_loss = tf.reduce_sum(total_loss_matrix) / count

    return total_loss

def custom_loss(y_true, rest_spectra, latent_vectors, latent_augmented, ivar_batch, observed_masks, z_batch, wave_rest, wave_obs, epoch, extrap_start_epoch=10):
    obs_min = 3600
    obs_max = 10300
    rest_min = tf.reduce_min(wave_rest)
    rest_max = tf.reduce_max(wave_rest)
    rest_len = tf.shape(rest_spectra)[1]
    obs_len = tf.shape(y_true)[1]

    dx = (rest_max - rest_min) / tf.cast(rest_len - 1, tf.float32)
    indices = (wave_obs - rest_min) / dx
    idx_low = tf.clip_by_value(tf.floor(indices), 0, tf.cast(rest_len - 2, tf.float32))
    idx_high = idx_low + 1
    weight_high = indices - idx_low
    weight_low = 1.0 - weight_high

    idx_low = tf.cast(idx_low, tf.int32)
    idx_high = tf.cast(idx_high, tf.int32)

    # Fidelity loss
    rest_spectra_downsampled = rest_to_observed_interp(rest_spectra, z_batch, wave_rest, wave_obs)
    mse = tf.square(rest_spectra_downsampled - y_true)
    mse = tf.squeeze(mse, axis=-1)  # shape now (batch_size, obs_length)
    weighted_mse = tf.reduce_sum(ivar_batch * mse)
    ivar_sum = tf.reduce_sum(ivar_batch) + 1e-8  # to avoid division by zero
    fid_loss = 0.5 * weighted_mse / ivar_sum

    # Consistency loss
    cons_loss = consistency_loss(latent_vectors, latent_augmented)

    # Similarity loss weight schedule
    sim_weight = min(1.0, epoch / 10.0)
    sim_loss = similarity_loss(latent_vectors, rest_spectra, ivar_batch, observed_masks)

    # Extrapolation loss weight schedule
    if epoch < extrap_start_epoch:
        extrap_weight = 0.0
    else:
        extrap_weight = min(1.0, (epoch - extrap_start_epoch) / 10.0)
    
    extrap_loss = extrapolation_loss(latent_vectors, rest_spectra, observed_masks)

    total_loss = fid_loss + sim_weight * sim_loss + cons_loss + extrap_weight * extrap_loss

    return fid_loss, sim_loss, cons_loss, extrap_loss, total_loss