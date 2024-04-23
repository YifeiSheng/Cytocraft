import pickle, copy, importlib, math, os
import seaborn as sns
import numpy as np
import pandas as pd
import umap
import scanpy as sc
from tqdm import tqdm

# from pathlib import Path
# from scipy import linalg as LA
# from numpy.linalg import svd, solve, lstsq
# from math import cos, sin
# from scipy.sparse import spmatrix, issparse, csr_matrix
# from anndata import AnnData
# from typing import Optional, Union
# from shapely.geometry import Point, MultiPoint
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.stats import ttest_ind
# import itertools
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from cytocraft.craft import *
from cytocraft import model
from cytocraft import util
from cytocraft.model import BasisShapeModel
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import fcluster

# seed = np.random.RandomState(seed=3)


def get_cmap(cmap, ncolor):
    # Define the color map
    cmap = plt.get_cmap(cmap)

    # Get the RGB values of each color in the color map
    hex_values = []
    for i in range(ncolor):
        hex_values.append(mcolors.rgb2hex(cmap(int(256 / ncolor) * i)))

    return hex_values


def plot_RMSD_heatmap(
    D,
    labels,
    cmap="viridis",
    method="average",
    cthresh=4.8,
    font_scale=1,
    cbbox=(0.07, 0.84, 0.04, 0.12),
    lgbox=(0.86, 0.96, 0.01, 0.01),
    ticks=[0, 0.5, 1, 1.5],
    figsize=(8, 8),
    dpi=300,
):
    clustergrid = sns.clustermap(
        D, xticklabels=labels, yticklabels=labels, method=method
    )
    plt.close()

    # Get clusters with threshold
    linkage = clustergrid.dendrogram_row.linkage
    threshold = cthresh
    clusters = fcluster(linkage, threshold, criterion="distance")

    # Create a color map for clusters
    n_clusters = len(np.unique(clusters))
    color_map = sns.color_palette("rainbow", n_clusters)

    # Assign colors to clusters
    cluster_colors = [color_map[c - 1] for c in clusters]

    # Draw plot
    sns.set(rc={"figure.dpi": dpi, "figure.figsize": figsize}, font_scale=font_scale)

    ax = sns.clustermap(
        D,
        xticklabels=labels,
        yticklabels=labels,
        cmap=cmap,
        method=method,
        row_colors=[cluster_colors],
        col_colors=[cluster_colors],
        cbar_kws={"ticks": ticks},
    )
    ax.ax_cbar.set_title("RMSD")
    ax.ax_cbar.set_position(cbbox)

    # Add legend for row color bar
    from matplotlib.patches import Patch

    lut = dict(zip(clusters, cluster_colors))
    handles = [Patch(facecolor=lut[name]) for name in sorted(lut)]
    legend = plt.legend(
        handles,
        sorted(lut),
        title="Clusters",
        bbox_to_anchor=lgbox,
        bbox_transform=plt.gcf().transFigure,
        loc="upper right",
        ncol=2,
    )
    legend.get_frame().set_facecolor("white")


def plot_network(
    D,
    labels,
    cmap="viridis",
    csep="\n",
    solver="barnesHut",
    corder=0,
    N_neighbor=3,
    html="test.html",
    fontsize=15,
    width=1400,
    height=1000,
    edge_scale=150,
    edge_adjust=-60,
):
    from pyvis.network import Network
    from IPython.display import HTML
    import plotly.graph_objs as go
    import itertools
    import networkx as nx
    import plotly.io as pio
    import kaleido
    import json

    # create an empty graph
    G = nx.Graph()

    # add nodes
    N = D.shape[0]
    for i in range(N):
        G.add_node(i, label=labels[i])

    # add edges
    edges = []

    for i in range(N):
        arr = D[i, :]
        min_indices = np.argpartition(arr, int(N_neighbor + 1))[: int(N_neighbor + 1)]
        min_values = arr[min_indices]
        for j in min_indices:
            if not i == j:
                edges.append(
                    (
                        i,
                        int(j),
                        {"weight": float(1 / arr[j] * edge_scale + edge_adjust)},
                    )
                )

    G.add_edges_from(edges)

    # Draw pyvis
    net = Network(width=width, height=height, notebook=True, cdn_resources="remote")
    net.from_nx(G)

    # set node color based on label
    for node in net.nodes:
        node["group"] = node["label"].split(csep)[corder]
        node["color"] = cmap[node["group"]]
        # print(node['color'])

    # show
    net.show_buttons()

    json_obj = {
        "configure": {"enabled": True, "filter": ["nodes", "edges", "physics"]},
        "nodes": {
            "borderWidth": 3,
            "opacity": 1,
            "font": {"size": fontsize, "strokeWidth": 5},
            "size": 0,
        },
        "edges": {
            "color": {"opacity": 0.7},
            "selfReferenceSize": 0,
            "selfReference": {"size": 0, "angle": 0.7853981633974483},
            "smooth": {"forceDirection": "vertical"},
        },
        "physics": {
            "minVelocity": 0.75,
            "solver": solver,
        },
    }

    net.set_options("const options = " + json.dumps(json_obj))

    net.show(html)


# UMAP
def umap_leiden(
    distance_matrix,
    n_neighbors=15,
    min_dist=0.1,
    resolution=1,
    plot_umap=True,
    seed=99,
    title="",
):
    umap_coord = umap.UMAP(
        metric="precomputed",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    ).fit_transform(distance_matrix)
    adata = sc.AnnData(pd.DataFrame(np.zeros((len(umap_coord), len(umap_coord)))))
    adata.obsm["umap"] = umap_coord
    sc.pp.neighbors(adata, use_rep="umap")
    sc.tl.leiden(adata, resolution=resolution)
    if plot_umap:
        sc.pl.umap(adata, color=["leiden"], title=title)
    return adata.obs["leiden"]
