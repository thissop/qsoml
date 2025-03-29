import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import umap
import matplotlib.pyplot as plt
import smplotlib 

tfd = tfp.distributions
tfb = tfp.bijectors

def build_maf_flow(latent_dim, num_layers=5, hidden_units=50):
    bijectors = []
    for _ in range(num_layers):
        maf = tfb.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn=tfb.AutoregressiveNetwork(
                params=2,
                hidden_units=[hidden_units, hidden_units],
                activation='relu'
            )
        )
        bijectors.append(maf)
        bijectors.append(tfb.Permute(permutation=list(reversed(range(latent_dim)))))  # reverse permutation

    flow_bijector = tfb.Chain(list(reversed(bijectors)))
    base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros(latent_dim))
    flow = tfd.TransformedDistribution(distribution=base_dist, bijector=flow_bijector)
    return flow

def train_maf(flow, latent_train, n_epochs=2000, batch_size=10000):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    flow_vars = flow.trainable_variables

    dataset = tf.data.Dataset.from_tensor_slices(latent_train.astype(np.float32))
    dataset = dataset.shuffle(buffer_size=latent_train.shape[0]).batch(batch_size)

    for epoch in range(n_epochs):
        epoch_loss = []
        for batch in dataset:
            with tf.GradientTape() as tape:
                nll = -tf.reduce_mean(flow.log_prob(batch))
            grads = tape.gradient(nll, flow_vars)
            optimizer.apply_gradients(zip(grads, flow_vars))
            epoch_loss.append(nll.numpy())
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}: NLL = {np.mean(epoch_loss):.4f}")

def plot_umap(flow, latent_train, z_train):
    z_train_norm = flow.bijector.inverse(latent_train.astype(np.float32)).numpy()

    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1)
    z_umap = umap_model.fit_transform(z_train_norm)

    plt.figure(figsize=(8, 6))
    plt.scatter(z_umap[:, 0], z_umap[:, 1], c=z_train.flatten(), cmap='viridis', s=10)
    plt.colorbar(label='Redshift')
    plt.title('UMAP of normalized latent space')
    plt.show()

# Assuming latent_train is already computed from encoder(y_train)
latent_dim = latent_train.shape[1]
flow = build_maf_flow(latent_dim)

train_maf(flow, latent_train, n_epochs=2000, batch_size=10000)

plot_umap(flow, latent_train, z_train)
