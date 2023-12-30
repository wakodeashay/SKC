#!/usr/bin/python3.8
from hilbertcurve.hilbertcurve import HilbertCurve
import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
from numba import njit
import random

# plt.rcParams["figure.dpi"] = 1200

fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
# Variables
# Iteration of Hilbert's curve
iter = 4
# Dimension of Hilber's curve
dim = 2
# Number of points in Hilbert's curve
size = 2 ** (iter * dim)
side_size = 2 ** (iter * dim / 2)
# Area covered by the Hibert's curve
xmin = 0
ymin = 0
xmax = 10 * side_size
ymax = 10 * side_size
# Grid size
xgrid = 1
ygrid = 1
# Bounding grid
xmin_grid = xmin - xgrid / 2
ymin_grid = ymin - ygrid / 2
xmax_grid = xmax + xgrid / 2
ymax_grid = ymax + ygrid / 2

# Creating Hilbert's curve
hilbert_curve = HilbertCurve(iter, dim)
distances = list(range(size))
points = hilbert_curve.points_from_distances(distances)

x = np.array([points[i][0] for i in range(size)])
y = np.array([points[i][1] for i in range(size)])


@njit
def get_point_index(xl, yl):
    """Calculates the index of the point on the hilbert's curve"""
    global x, y
    x_index = np.where(x == xl)[0]
    y_index = np.where(y == yl)[0]
    return list(np.intersect1d(x_index, y_index))


def get_adjacent_nodes(i):
    """Outputs the adjacent nodes of a given node"""
    x_i = x[i]
    y_i = y[i]

    adjacent_points = {
        "p1": [x_i + xgrid, y_i],
        "p2": [x_i, y_i + ygrid],
        "p3": [x_i - xgrid, y_i],
        "p4": [x_i, y_i - ygrid],
    }

    adjacent_nodes = [
        get_point_index(adjacent_points[point][0], adjacent_points[point][1])
        if adjacent_points[point][0] in x and adjacent_points[point][1] in y
        else [-1]
        for point in adjacent_points
    ]

    return adjacent_nodes


def get_adjacency_list(v_list, obs_list):
    """Gives the Adjacency list"""
    v_neigh = g.neighborhood(vertices=v_list)
    result_set = {v for neigh_list in v_neigh for v in neigh_list} - \
        set(v_list) - set(obs_list)
    return list(result_set)


def get_req_path(subgraph_list, l):
    """Gives the smallest path present within the subgraph"""
    k = np.array(l)
    if np.shape(k)[0] == 1:
        return k
    k = k[:, 0:-1]
    for i in range(size):
        if set(k[i]).intersection(subgraph_list) == set(k[i]):
            return k[i]


def get_subgraph(v):
    """Generates subgraph for vertex list v"""
    o = ig.Graph(n=size)
    for i in v:
        neigh = get_adjacent_nodes(i)
        for j in range(4):
            if (
                neigh[j][0] != -1
                and o.are_connected(i, neigh[j][0]) == False
                and neigh[j][0] in v
            ):
                o.add_edge(i, neigh[j][0])
    o.vs["name"] = [str(i) for i in range(size)]
    o.vs["label"] = o.vs["name"]
    return o


g = ig.Graph(n=size)

# graph Generation
for i in range(size):
    neigh = get_adjacent_nodes(i)
    for j in range(4):
        if neigh[j][0] != -1 and g.are_connected(i, neigh[j][0]) == False:
            g.add_edge(i, neigh[j][0])

xm = [0]
ym = [0]

# investigation example iteration = 3
# obstacle = np.array([32,33,46,47,48,51,52,53])
# investigation example iteration = 5
# obstacle = np.array([130,134,135,191,190,192, 193, 194, 195, 184,185, 188, 189 ,204, 205, 209, 210, 212, 215, 216])
# Example 1 Normal; iteration = 3
# obstacle = np.array([22, 23,24,25, 49,50,55,56,59,60,61,62,63])
# Example 2 Sparse; iteration = 5


def get_sparse_obstacle(start, end, size):
    r = []
    for _ in range(size):
        r.append(random.randint(start, end))
    return r

# Obstacle list for sparse obstacles
# obstacle = np.array(get_sparse_obstacle(1, size, 350))


# Non Uniform coverage example
# Iteration 4
obstacle = np.array([40, 41, 42, 43, 52, 53, 54, 55, 92, 93, 94, 95, 164, 165, 166, 167, 168, 169,
                    170, 171, 172, 173, 174, 175, 176, 177, 209, 210, 221, 222, 252, 253, 254, 255, 242, 241, 240])
# Iteration 2
# obstacle = np.array([12, 15])
# Iteration 1
# obstacle = np.array([1])

# Initiaize visited_node list with the starting node
visited_nodes = np.array([0])
obstacle_detected = np.array([])

i = get_subgraph(visited_nodes)

while get_adjacency_list(visited_nodes, obstacle_detected) != []:
    min_adj = min(get_adjacency_list(visited_nodes, obstacle_detected))
    o = get_subgraph(np.append(visited_nodes, min_adj))
    if min_adj == visited_nodes[-1] + 1:
        if min_adj in obstacle:
            obstacle_detected = np.append(obstacle_detected, min_adj)
        else:
            visited_nodes = np.append(visited_nodes, min_adj)
    else:
        l = get_req_path(
            visited_nodes, o.get_all_shortest_paths(
                visited_nodes[-1], to=min_adj)
        )
        if min_adj in obstacle:
            try:
                obstacle_detected = np.append(obstacle_detected, l[0][-1])
                visited_nodes = np.append(visited_nodes, l[0][1:-1])
            except:
                obstacle_detected = np.append(obstacle_detected, l[-1])
                visited_nodes = np.append(visited_nodes, l[1:-1])
        else:
            try:
                visited_nodes = np.append(visited_nodes, l[0][1:])
            except:
                visited_nodes = np.append(visited_nodes, l[1:])


x_visited = [x[i] for i in visited_nodes]
y_visited = [y[i] for i in visited_nodes]

x_bound = [min(x) - xgrid/2, min(x) - xgrid/2, max(x) +
           xgrid/2, max(x) + xgrid/2, min(x) - xgrid/2]
y_bound = [min(y) - ygrid/2, max(y) + ygrid/2, max(y) +
           ygrid/2, min(y) - ygrid/2, min(y) - ygrid/2]


for i in range(size):
    if i in obstacle_detected:
        rectangle = plt.Rectangle(
            (x[i] - xgrid / 2, y[i] - ygrid / 2),
            1,
            1,
            fc="red",
            alpha=0.2,
            ec="black",
        )
        plt.gca().add_patch(rectangle)
    else:
        rectangle = plt.Rectangle(
            (x[i] - xgrid / 2, y[i] - ygrid / 2),
            1,
            1,
            fc="grey",
            alpha=0.04,
            ec="black",
        )
        plt.gca().add_patch(rectangle)

node_coord = [[x[i], y[i]] for i in range(size)]
visual_style = {}
g.vs["name"] = [str(i) for i in range(size)]

visual_style["edge_width"] = [0.3]
visual_style["vertex_size"] = 10
visual_style["edge_color"] = "orange"
layout_subgraph = ig.Layout(coords=node_coord)

ig.plot(g, target=ax, layout=layout_subgraph, **visual_style)
plt.plot(x_visited[0], y_visited[0],  marker="o", markersize=15, markeredgecolor="blue", markerfacecolor="blue")
plt.plot(x_visited[-1], y_visited[-1],  marker="o", markersize=15, markeredgecolor="gold", markerfacecolor="gold")
plt.plot(x_bound, y_bound, linestyle="solid", color="black", linewidth=0.5)
plt.plot(x_visited, y_visited, linestyle="solid", color="green", linewidth=1.0)
plt.show()
