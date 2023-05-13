#!/usr/bin/python3.8
from hilbertcurve.hilbertcurve import HilbertCurve
import matplotlib.pyplot as plt
import numpy as np
import igraph as ig

# plt.style.use(["ieee"])
plt.rcParams["figure.dpi"] = 1200

fig, ax = plt.subplots()

# Variables
# Iteration of Hilbert's curve
iter = 6
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


def get_point_index(xl, yl):
    """Calculates the index of the point on the hilbert's curve"""
    global x,y
    x_index = np.where(x == xl)[0]
    y_index = np.where(y == yl)[0]
    return list(np.intersect1d(x_index, y_index))


def get_adjacent_nodes(i):
    """Outputs the adjacent nodes of a given node"""
    x_i = x[i]
    y_i = y[i]

    p1_x = x_i + xgrid
    p1_y = y_i
    p1_bool = p1_x in x

    p2_x = x_i
    p2_y = y_i + ygrid
    p2_bool = p2_y in y

    p3_x = x_i - xgrid
    p3_y = y_i
    p3_bool = p3_x in x

    p4_x = x_i
    p4_y = y_i - ygrid
    p4_bool = p4_y in y

    if p1_bool:
        p1_index = get_point_index(p1_x, p1_y)
    else:
        p1_index = [-1]

    if p2_bool:
        p2_index = get_point_index(p2_x, p2_y)
    else:
        p2_index = [-1]

    if p3_bool:
        p3_index = get_point_index(p3_x, p3_y)
    else:
        p3_index = [-1]

    if p4_bool:
        p4_index = get_point_index(p4_x, p4_y)
    else:
        p4_index = [-1]

    return [p1_index, p2_index, p3_index, p4_index]


def get_adjacency_list(v_list, obs_list):
    """Gives the Adjacency list"""
    v_neigh = g.neighborhood(vertices=v_list)
    resultList = [element for nestedlist in v_neigh for element in nestedlist]
    return list(set(resultList) - set(v_list) - set(obs_list))


def get_req_path(subgraph_list, l):
    """Gives the smallest path present within the subgraph"""
    k = np.array(l)
    if np.shape(k)[0] == 1:
        return k
    k = k[:, 0:-1]

    for i in range(size):
        if all(item in subgraph_list for item in k[i]):
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

obstacle = np.array([3, 45, 23, 24, 14])
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
            visited_nodes, o.get_all_shortest_paths(visited_nodes[-1], to=min_adj)
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

# TODO Find better way to show the modified path
# TODO make the code modular

x_visited = [x[i] for i in visited_nodes]
y_visited = [y[i] for i in visited_nodes]


for i in range(size):
    if i in obstacle_detected:
        rectangle = plt.Rectangle(
            (x[i] - xgrid / 2, y[i] - ygrid / 2), 1, 1, fc="brown", alpha=0.5, ec="black")
        plt.gca().add_patch(rectangle)
    else:
        rectangle = plt.Rectangle(
            (x[i] - xgrid / 2, y[i] - ygrid / 2),
            1,
            1,
            fc="brown",
            alpha=0.1,
            ec="black")
        plt.gca().add_patch(rectangle)

node_coord = [[x[i], y[i]] for i in range(size)]
visual_style = {}
g.vs["name"] = [str(i) for i in range(size)]

visual_style["vertex_color"] = "black"
visual_style["vertex_size"] = [0.1]
visual_style["edge_width"] = [0.3]
visual_style["edge_color"] = "grey"

layout_subgraph = ig.Layout(coords=node_coord)

ig.plot(
    g,
    target=ax,
    layout=layout_subgraph,
    **visual_style
)

plt.plot(x_visited, y_visited, linestyle="solid", color="green", linewidth=0.5)
plt.plot(x, y, linestyle="dotted", linewidth=0.4)
plt.show()
