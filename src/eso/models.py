"""PCA, autoencoder and sparse variational GPLVM with a fixed holdout task."""
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def pca_model(train, test, observed, latent_dim, seed):
    model = PCA(n_components=latent_dim, svd_solver='full').fit(train)
    z = model.transform(train)
    # Infer held-out GSP coordinates from observed periods only.
    ztest = np.linalg.lstsq(model.components_[:, observed].T,
                           (test[:, observed] - model.mean_[observed]).T, rcond=None)[0].T
    return z, model.inverse_transform(z), model.inverse_transform(ztest), model


def infer_decoder(decoder, test, observed, latent_dim, steps, rate=0.03):
    target = torch.as_tensor(test, dtype=torch.float64)
    z = torch.nn.Parameter(torch.zeros(len(test), latent_dim, dtype=torch.float64))
    opt = torch.optim.Adam([z], lr=rate)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((decoder(z)[:, observed] - target[:, observed]) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return decoder(z).cpu().numpy()


def autoencoder_model(train, test, observed, latent_dim, seed, steps, infer_steps):
    torch.manual_seed(seed)
    d = train.shape[1]
    hidden = min(64, max(8, d // 2))
    encoder = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.Tanh(),
                                  torch.nn.Linear(hidden, latent_dim)).double()
    decoder = torch.nn.Sequential(torch.nn.Linear(latent_dim, hidden), torch.nn.Tanh(),
                                  torch.nn.Linear(hidden, d)).double()
    model = torch.nn.Sequential(encoder, decoder)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    y = torch.as_tensor(train, dtype=torch.float64)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = ((model(y) - y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        z = encoder(y).numpy()
        reconstruction = model(y).numpy()
    prediction = infer_decoder(decoder, test, observed, latent_dim, infer_steps)
    artifact = {'state_dict': model.state_dict(), 'input_dim': d, 'hidden': hidden,
                'latent_dim': latent_dim, 'activation': 'tanh', 'dtype': 'float64'}
    return z, reconstruction, prediction, artifact, losses


def gplvm_model(train, test, observed, labels, latent_dim, seed, prior_mode,
                prior_scale, inducing_count, inducing_strategy, steps, infer_steps):
    import pyro
    import pyro.contrib.gp as gp
    import pyro.distributions as dist

    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    initial = PCA(n_components=latent_dim, svd_solver='full').fit_transform(train)
    initial /= np.maximum(initial.std(axis=0), 1e-8)
    initial += np.random.default_rng(seed).normal(0, 0.01, initial.shape)
    x = torch.tensor(initial, dtype=torch.float64)
    prior = torch.zeros_like(x)
    if prior_mode == 'label_informed':
        prior[:, 0] = torch.as_tensor(labels, dtype=x.dtype)
    elif prior_mode != 'independent':
        raise ValueError(f'Unknown prior mode {prior_mode}')
    m = min(inducing_count, len(train))
    if inducing_strategy == 'kmeans':
        xu = KMeans(m, random_state=seed, n_init=10).fit(initial).cluster_centers_
    elif inducing_strategy == 'random':
        xu = initial[np.random.default_rng(seed).choice(len(train), m, replace=False)]
    else:
        raise ValueError(f'Unknown inducing strategy {inducing_strategy}')
    y = torch.tensor(train.T, dtype=torch.float64)
    kernel = gp.kernels.RBF(latent_dim, lengthscale=torch.ones(latent_dim, dtype=x.dtype))
    model = gp.models.SparseGPRegression(x, y, kernel, torch.tensor(xu, dtype=x.dtype),
                                         noise=torch.tensor(0.01, dtype=x.dtype), jitter=1e-5).double()
    model.X = pyro.nn.PyroSample(dist.Normal(prior, prior_scale).to_event())
    model.autoguide('X', dist.Normal)
    with torch.no_grad():
        model.X_loc.copy_(x)
    losses = gp.util.train(model, optimizer=torch.optim.Adam(model.parameters(), lr=0.01), num_steps=steps)
    if not np.isfinite(losses).all():
        raise ValueError('Non-finite GPLVM training objective')
    z = model.X_loc.detach().clone()
    sd = model.X_scale.detach().cpu().numpy().copy()
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # Freeze fitted hyperparameters and latent means for deterministic decoding.
    fixed = gp.models.SparseGPRegression(z, y, model.kernel, model.Xu.detach().clone(),
                                         noise=model.noise.detach().clone(), jitter=1e-5).double()
    for p in fixed.parameters():
        p.requires_grad_(False)

    def decoder(coords):
        return fixed(coords, full_cov=False, noiseless=True)[0].T

    with torch.no_grad():
        reconstruction = decoder(z).numpy()
    prediction = infer_decoder(decoder, test, observed, latent_dim, infer_steps)
    artifact = {'state_dict': state, 'latent_mean': z, 'latent_sd': sd,
                'train_y': y, 'inducing': fixed.Xu.detach(),
                'prior_mean': prior, 'prior_scale': prior_scale,
                'latent_dim': latent_dim, 'jitter': 1e-5}
    return z.numpy(), reconstruction, prediction, artifact, losses, sd
