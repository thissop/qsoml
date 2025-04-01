import tensorflow as tf

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

def similarity_loss(latent_vectors, rest_spectra, ivar_batch, k1=5.0, k0=2.5, eps=1e-8):
    N = tf.cast(tf.shape(latent_vectors)[0], tf.float32)
    S = tf.cast(tf.shape(latent_vectors)[-1], tf.float32)
    M = tf.cast(tf.shape(rest_spectra)[-1], tf.float32)

    latent_dists = pairwise_squared_distances(latent_vectors) / S

    spectral_dists = pairwise_squared_distances(rest_spectra)
    ivar_weights = tf.reduce_mean(ivar_batch, axis=-1)
    w_outer = tf.expand_dims(ivar_weights, 1) * tf.expand_dims(ivar_weights, 0)
    spectral_term = (w_outer * spectral_dists) / M

    latent_mean = tf.reduce_mean(latent_dists)
    latent_std = tf.math.reduce_std(latent_dists) + eps
    latent_norm = (latent_dists - latent_mean) / latent_std

    spectral_mean = tf.reduce_mean(spectral_term)
    spectral_std = tf.math.reduce_std(spectral_term) + eps
    spectral_norm = (spectral_term - spectral_mean) / spectral_std

    delta = latent_norm - spectral_norm
    sigmoid_pos = tf.sigmoid(k1 * delta - k0)
    sigmoid_neg = tf.sigmoid(-k1 * delta - k0)
    sim_loss = tf.reduce_mean(sigmoid_pos + sigmoid_neg)

    return sim_loss

def extrapolation_loss(latent_vectors, rest_spectra, observed_masks, distance_scale=1.0, eps=1e-8):
    N = tf.shape(latent_vectors)[0]
    L = tf.shape(rest_spectra)[1]

    latent_dists = pairwise_squared_distances(latent_vectors)
    S = tf.cast(tf.shape(latent_vectors)[-1], tf.float32)
    mean_dist = tf.reduce_mean(latent_dists)
    normalized_latent_dists = latent_dists / (S * mean_dist + eps)

    obs_i = tf.expand_dims(observed_masks, 1)
    obs_j = tf.expand_dims(observed_masks, 0)

    mask_AB = (1.0 - obs_i) * obs_j
    mask_BA = (1.0 - obs_j) * obs_i

    spec_i = tf.expand_dims(rest_spectra, 1)
    spec_j = tf.expand_dims(rest_spectra, 0)

    diff = tf.square(spec_i - spec_j)
    loss_AB = tf.reduce_sum(diff * mask_AB, axis=-1)
    loss_BA = tf.reduce_sum(diff * mask_BA, axis=-1)
    total_loss_matrix = loss_AB + loss_BA

    distance_weight = tf.exp(-distance_scale * normalized_latent_dists)

    total_loss_matrix *= distance_weight

    count = tf.reduce_sum(distance_weight * (tf.reduce_sum(mask_AB + mask_BA, axis=-1))) + eps
    total_loss = tf.reduce_sum(total_loss_matrix) / count

    return total_loss

def custom_loss(y_true, y_pred, latent_vectors, latent_augmented, rest_spectra, ivar_batch, observed_masks, epoch, extrap_start_epoch=10):
    # Fidelity loss
    fid_loss = tf.reduce_mean(tf.square(y_true - y_pred)) # weight by ivar? spender does not. 

    # Consistency loss
    cons_loss = consistency_loss(latent_vectors, latent_augmented)

    # Similarity loss weight schedule
    sim_weight = min(1.0, epoch / 10.0)
    sim_loss = similarity_loss(latent_vectors, rest_spectra, ivar_batch)

    # Extrapolation loss weight schedule
    if epoch < extrap_start_epoch:
        extrap_weight = 0.0
    else:
        extrap_weight = min(1.0, (epoch - extrap_start_epoch) / 10.0)
    
    extrap_loss = extrapolation_loss(latent_vectors, rest_spectra, observed_masks)

    total_loss = fid_loss + sim_weight * sim_loss + cons_loss + extrap_weight * extrap_loss

    return fid_loss, sim_loss, cons_loss, extrap_loss, total_loss