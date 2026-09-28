"""Descriptive label agreement is distinct from independent physical validation."""
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors


def latent_metrics(latent, labels):
    n = len(latent)
    out = {}
    if 1 < len(np.unique(labels)) < n:
        out.update(silhouette=float(silhouette_score(latent, labels)),
                   davies_bouldin=float(davies_bouldin_score(latent, labels)),
                   calinski_harabasz=float(calinski_harabasz_score(latent, labels)))
    else:
        out.update(silhouette=None, davies_bouldin=None, calinski_harabasz=None)
    if n > 1:
        # Explicitly remove self, including when coordinates are duplicated.
        indices = NearestNeighbors(n_neighbors=min(6, n)).fit(latent).kneighbors(latent, return_distance=False)
        matches = [np.mean(labels[ix[ix != i][:5]] == labels[i]) for i, ix in enumerate(indices)]
        out['knn_label_agreement'] = float(np.mean(matches))
    return out


def distance_stability(first, second):
    """Rank agreement between pair distances; invariant to rotation/translation."""
    a, b = pdist(first), pdist(second)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)
