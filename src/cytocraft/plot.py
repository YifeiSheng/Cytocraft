import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import pickle, copy, importlib, math, os
import seaborn as sns
import numpy as np
import pandas as pd
import umap
import scanpy as sc
from tqdm import tqdm
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
    legend=True,
    cbbox=(0.07, 0.84, 0.04, 0.12),
    lgbox=(0.86, 0.96, 0.01, 0.01),
    ticks=[0, 0.5, 1, 1.5],
    figsize=(8, 8),
    dpi=300,
    legend_ncol=2,
    mask=None,
):
    import matplotlib.pyplot as plt

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
        mask=mask,
    )
    ax.ax_cbar.set_title("RMSD")
    ax.ax_cbar.set_position(cbbox)

    if legend:
        # Add legend for row color bar
        from matplotlib.patches import Patch

        lut = dict(zip(clusters, cluster_colors))
        handles = [Patch(facecolor=lut[name]) for name in sorted(lut)]
        legend = plt.legend(
            handles,
            sorted(lut),
            title="Groups",
            bbox_to_anchor=lgbox,
            bbox_transform=plt.gcf().transFigure,
            loc="upper right",
            ncol=legend_ncol,
        )
        legend.get_frame().set_facecolor("white")

    return ax


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


def connect_edge_knn(G, k):
    """
    Connect nodes in the graph G with k-nearest neighbors.
    Parameters:
        G (nx.Graph): The input graph.
        k (int): The number of nearest neighbors to connect.
    Returns:
        nx.Graph: The graph with edges added
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    # Get the node coordinates
    node_coords = np.array([G.nodes[u]["pos"] for u in G.nodes()])

    # Fit the k-nearest neighbors model
    knn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    knn.fit(node_coords)
    # Find the k-nearest neighbors
    distances, indices = knn.kneighbors(node_coords)

    nodes = list(G.nodes())
    # Add edges to the graph
    for i in range(len(node_coords)):
        for j in range(1, k + 1):
            start = indices[i][0]
            end = indices[i][j]
            d = distances[i][j]
            G.add_edge(nodes[start], nodes[end], weight=1 / d, capacity=1 / d)

    # components,components_draw = get_components(G,save_path)
    # run_Go(components_draw,save_path,species,label,df,draw)

    return G


def draw_5NN(G, title, save_path):
    """
    Draw the 5-nearest neighbors graph.
    Parameters:
        G (nx.Graph): The input graph.
        title (str): The title of the plot.
        save_path (str): The path to save the plot.
    Returns:
        None
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import networkx as nx
    import itertools
    from itertools import chain
    import os

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    components, components_draw = get_components(G, save_path)

    colors = plt.get_cmap("rainbow")(np.linspace(0, 1, len(components_draw)))[:, :3]
    # color_map = {node: colors[i] for i, comp in enumerate(components_draw) for node in comp}
    color_map = {
        node: colors[i] if comp in components_draw else "gray"
        for i, comp in enumerate(components)
        for node in comp
    }

    # alpha_map = {node: 1.0 if comp in components_draw else 0.3 for i, comp in enumerate(components) for node in comp}

    color_legend = {
        "component " + str(i): colors[i]
        for i, comp in enumerate(components_draw)
        if comp in components_draw
    }
    # with open(out_f + str(k) + "_knn.pkl", "wb") as f:
    #     pickle.dump(G, f)

    # from IPython.display import display

    node_size = nx.pagerank(G, alpha=0.85, weight=None)
    sorted_dict = sorted(node_size.items(), key=lambda x: x[1], reverse=True)
    labels = {}
    for i in sorted_dict[:10]:
        labels[i[0]] = i[0]

    node_positions = nx.get_node_attributes(G, "pos")

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")

    nodes_x = []
    nodes_y = []
    nodes_z = []
    node_s = []
    alphas = []
    node_color = []
    nodes = []

    for node, pos in node_positions.items():
        x, y, z = pos
        nodes_x.append(x)
        nodes_y.append(y)
        nodes_z.append(z)
        # x, y, z = pos
        node_s.append(np.sqrt(node_size[node]).tolist() * 900)
        nodes.append(node)

        # ax.scatter(x, y, z, c='b', s=node_s)
        # ax.scatter(x, y, z, c=[color_map[node]],zorder=1)
        node_color.append(color_map[node])
        # alphas.append(alpha_map[node])

    ax.set_xlim(min(nodes_x), max(nodes_x))
    ax.set_ylim(min(nodes_y), max(nodes_y))
    ax.set_zlim(min(nodes_z), max(nodes_z))

    for edge in G.edges():
        start = node_positions[edge[0]]
        end = node_positions[edge[1]]
        xs = [start[0], end[0]]
        ys = [start[1], end[1]]
        zs = [start[2], end[2]]
        # ax.quiver(xs[0], ys[0], zs[0], xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0],color=color_map[start.name],linewidth=2, pivot='tail', length=1,zorder=0)

        if color_map[start.name] == "gray" or color_map[end.name] == "gray":
            alpha = 0.3
        else:
            alpha = 1

        ax.plot(xs, ys, zs, c=color_map[start.name], linewidth=1.0, alpha=alpha)

    length = max(nodes_z) - min(nodes_z)
    sc = ax.scatter(
        nodes_x, nodes_y, nodes_z, alpha=0.3, c=node_color, s=[x * 2 for x in node_s]
    )

    for i, label in enumerate(labels):
        ax.text(
            node_positions[label][0],
            node_positions[label][1],
            node_positions[label][2] + 0.01 * length,
            label,
            fontsize=8,
            fontweight="bold",
            horizontalalignment="center",
            path_effects=[pe.withStroke(linewidth=1, foreground="white")],
        )

    legend_elements = [
        plt.Line2D([0], [0], color=color, lw=4, label=component)
        for component, color in color_legend.items()
    ]
    if len(legend_elements) > 50:
        plt.legend(
            handles=legend_elements,
            loc="center right",
            ncol=2,
            bbox_to_anchor=(1.4, 0.5),
        )
        plt.subplots_adjust(right=0.75)
        plt.tight_layout()

    else:
        plt.legend(
            handles=legend_elements, loc="center right", bbox_to_anchor=(1.15, 0.5)
        )
        plt.subplots_adjust(right=0.85)
        # plt.tight_layout()

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.title(title, fontsize=55)
    plt.show()
    plt.savefig(save_path + "3D.png", dpi=300, transparent=True)
    plt.clf()

    G_new = nx.DiGraph()

    for node in G.nodes:
        G_new.add_node(node)

    for u, v in G.edges:
        G_new.add_edge(u, v)

    inter = True

    keys1 = set({n: n for n in G}.keys())
    keys2 = set(labels.keys())
    mark_elements = set(chain(*components_draw))

    unimportant = {
        key: ({n: n for n in G}[key]) for key in (keys1 - keys2 - mark_elements)
    }

    marked = {key: ({n: n for n in G}[key]) for key in (mark_elements - keys2)}

    G_new = nx.DiGraph()

    node_s_marked = []
    count = 0
    for node in G.nodes:
        if node in mark_elements:

            if len(components_draw) > 20:
                node_s_marked.append(node_s[count] * 2)
            else:
                node_s_marked.append(node_s[count] * 4)

            G_new.add_node(node)
        count = count + 1

    components_df = sets_to_dataframe(components_draw)
    for u, v in G.edges:
        if (
            u in mark_elements
            and v in mark_elements
            and components_df.loc[u, "Index"] == components_df.loc[v, "Index"]
        ):
            G_new.add_edge(u, v)

    count = 0

    while inter and count < 50:
        count = count + 1
        # print(inter)
        pos = generate_pos(G_new)

        edges = list(G_new.edges)

        for e1, e2 in itertools.combinations(edges, 2):
            x1, y1 = pos[e1[0]]
            x2, y2 = pos[e1[1]]
            x3, y3 = pos[e2[0]]
            x4, y4 = pos[e2[1]]

            line1 = (x1, y1, x2, y2)
            line2 = (x3, y3, x4, y4)

            inter = line_intersection(line1, line2)

            if inter:
                inter = True

                break
            else:
                inter = False

    edge_colors = []
    for u, v in G_new.edges():
        edge_colors.append(color_map[u])

    plt.figure(figsize=(10, 10))
    # plt.tight_layout()

    if len(legend_elements) > 50:
        plt.legend(
            handles=legend_elements,
            loc="center right",
            fontsize=9,
            ncol=2,
            bbox_to_anchor=(1.3, 0.5),
        )
        plt.subplots_adjust(right=0.8)
        # plt.tight_layout()
    else:
        plt.legend(
            handles=legend_elements,
            loc="center right",
            fontsize=9,
            bbox_to_anchor=(1.15, 0.5),
        )
        plt.subplots_adjust(right=0.85)
        # plt.tight_layout()

    nx.draw(
        G_new,
        pos,
        with_labels=False,
        node_color=[color_map[node] for node in G_new.nodes()],
        edge_color=edge_colors,
        node_size=node_s_marked,
        arrowsize=5,
    )
    # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

    pos_higher = {}
    y_off = 0.2  # offset on the y axis
    for k, v in pos.items():
        pos_higher[k] = (v[0], v[1] + y_off)

    pos_bigger_higher = {}
    y_off = 0.5  # offset on the y axis
    for k, v in pos.items():
        pos_bigger_higher[k] = (v[0], v[1] + y_off)

    pos_mark_higher = {}
    y_off = 0.3  # offset on the y axis
    for k, v in pos.items():
        pos_mark_higher[k] = (v[0], v[1] + y_off)

    labels = {}
    count = 0
    for i in sorted_dict:
        if i[0] in G_new.nodes():
            labels[i[0]] = i[0]
            count = count + 1
        if count == 10:
            break

    marked = {
        key: ({n: n for n in G}[key]) for key in (mark_elements - set(labels.keys()))
    }

    # labels = {key: value for key, value in labels.items() if key in mark_elements}
    # nx.draw_networkx_labels(G_new, pos_higher, labels=unimportant, font_color='black', font_size=5)
    if len(pos_bigger_higher) < 400:
        nx.draw_networkx_labels(
            G_new,
            pos_bigger_higher,
            labels=labels,
            font_color="black",
            font_size=5,
            font_weight="bold",
        )
        nx.draw_networkx_labels(
            G_new, pos_mark_higher, labels=marked, font_color="black", font_size=3
        )
    else:
        nx.draw_networkx_labels(
            G_new,
            pos_bigger_higher,
            labels=labels,
            font_color="black",
            font_size=5,
            font_weight="bold",
        )
        nx.draw_networkx_labels(
            G_new, pos_mark_higher, labels=marked, font_color="black", font_size=3
        )

    # plt.show()
    plt.title(title, fontsize=35)

    # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    # plt.tight_layout()
    plt.show()
    plt.savefig(save_path + "2D.png", dpi=200, transparent=True)
    plt.clf()
    # cbar = plt.colorbar(sc,shrink=0.7)


def get_components(G, save_path, mini=5):
    import networkx as nx

    components = list(nx.weakly_connected_components(G))

    if len(components) < 5:
        # communities_generator = nx.community.girvan_newman(G)
        next_level_communities = nx.community.louvain_communities(G)
        # top_level_communities = next(communities_generator)
        # next_level_communities = next(communities_generator)
        # components = sorted(map(sorted, next_level_communities))

        components = [
            set(sub_list) for sub_list in sorted(map(sorted, next_level_communities))
        ]

    components = sorted(components, key=lambda x: len(x), reverse=True)

    components_draw = [comp for comp in components if len(comp) >= mini]

    components_length = sets_to_dataframe(components)

    components_length.to_csv(save_path + "/components_length.csv")

    if len(components_draw) == 0:
        components_draw = [comp for comp in components if len(comp) > 2]

    if len(components_draw) > 30:
        components_draw = components[:30]

    return components, components_draw


def sets_to_dataframe(sets_list):
    import pandas as pd

    """
    Convert a list of sets to a DataFrame with values as index and set lengths as column.
    
    Parameters:
    sets_list (list of sets): The input list of sets.
    
    Returns:
    pd.DataFrame: The resulting DataFrame with values as index and set lengths as the only column.
    """
    # Create a dictionary to store the data
    data = {"Length": []}

    # Iterate over each set in the list
    index = 1
    for s in sets_list:
        length = len(s)
        # print(index)
        for value in s:
            data["Length"].append((value, length, index))
            # ,index
        index = index + 1

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data["Length"], columns=["Value", "Length", "Index"])

    # Set the index to 'Value'
    df.set_index("Value", inplace=True)

    return df


def generate_pos(G_new):
    import igraph as ig

    G_2 = ig.Graph.from_networkx(G_new)
    labels = list(G_2.vs["_nx_name"])
    N = len(labels)
    E = [e.tuple for e in G_2.es]  # list of edges
    layt = G_2.layout("fr")

    Xn = [layt[k][0] for k in range(N)]
    Yn = [layt[k][1] for k in range(N)]

    pos = {}
    for n in range(N):
        node = labels[n]
        pos[node] = [Xn[n], Yn[n]]

    return pos


def line_intersection(line1, line2):

    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    if x1 == x3 or x1 == x4 or x2 == x3 or x2 == x4:
        return False

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:
        return False

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

    if 0 < ua < 1 and 0 < ub < 1:
        return True

    return False
