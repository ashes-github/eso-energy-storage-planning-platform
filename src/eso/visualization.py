"""Plots with stable semantic labels and explicit unmatched map reporting."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def latent_plot(z, labels, path, title):
    fig, ax = plt.subplots(figsize=(7, 5), layout='constrained')
    for flag, color, name in [(False, '#7a8899', 'Not flagged'), (True, '#007f73', 'Candidate')]:
        mask = np.asarray(labels) == flag
        if mask.any():
            ax.scatter(z[mask, 0], z[mask, 1], color=color, label=name, s=18, alpha=0.8)
    ax.set(xlabel='Latent coordinate 1', ylabel='Latent coordinate 2', title=title)
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def loss_plot(losses, path):
    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    ax.plot(losses, color='#007f73')
    ax.set(xlabel='Training step', ylabel='Training objective', title='Optimization history')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def geospatial_plot(features, shapefile, output):
    import geopandas as gpd
    geo = gpd.read_file(shapefile)
    if 'GSPs' not in geo:
        raise ValueError('Expected GSPs identifier column in boundary file')
    joined = geo.merge(features[['screening_status']], left_on='GSPs', right_index=True, how='left')
    unmatched = features.loc[~features.index.isin(geo['GSPs'])]
    unmatched.to_csv(output / 'unmatched_gsps.csv')
    fig, ax = plt.subplots(figsize=(7, 9), layout='constrained')
    colors = {'candidate': '#007f73', 'not_flagged': '#b0bac5',
              'requires_review_negative_volume': '#dc933b',
              'candidate_imputed': '#43b6a7', 'not_flagged_imputed': '#8599bb',
              'requires_review_negative_volume_imputed': '#dc933b',
              'insufficient_evidence': '#dc933b'}
    joined['map_color'] = joined['screening_status'].map(colors).fillna('#eeeeee')
    joined.plot(ax=ax, color=joined['map_color'], edgecolor='white', linewidth=0.15)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=color, label=name.replace('_', ' ')) for name, color in colors.items()
                       if joined['screening_status'].eq(name).any()]
              + [Patch(color='#eeeeee', label='Not analysed')], loc='upper left', fontsize=8)
    ax.set_title('GSP demand-pattern screening (uncalibrated)')
    ax.set_axis_off()
    fig.savefig(output / 'screening_map.png', dpi=150)
    plt.close(fig)
    return len(unmatched)
