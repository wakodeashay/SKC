from hilbertcurve.hilbertcurve import HilbertCurve
import matplotlib.pyplot as plt
import scienceplots
import matplotlib.pyplot as plt
import numpy as np
import igraph as ig

plt.style.use(["ieee"])
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# Variables
# Iteration of Hilbert's curve
iter = 2
# Dimension of Hilber's curve
dim = 2
# Number of points in Hilbert's curve
size = 2 ** (iter * dim)
side_size = 2 ** (iter * dim / 2)
# Area covered by the Hibert's curve
xmin = 0
ymin = 0
xmax = 10
ymax = 10
# Grid size
xgrid = 1
ygrid = 1

# Creating Hilbert's curve
hilbert_curve = HilbertCurve(iter, dim)
distances = list(range(size))
points = hilbert_curve.points_from_distances(distances)
x = []
y = []
# print(points[size - 1][1])
for i in range(size):
        x.append(points[i][0])
        y.append(points[i][1])

def get_point_index(xl,yl):
     x_index = [index for (index, item) in enumerate(x) if item == xl]
     y_index = [index for (index, item) in enumerate(y) if item == yl]
     return list(np.intersect1d(x_index, y_index))

def get_adjacent_nodes(i):
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

    if p1_bool == True :
         p1_index = get_point_index(p1_x,p1_y)
    else :
         p1_index = [-1]

    if p2_bool == True :
         p2_index = get_point_index(p2_x,p2_y)
    else :
         p2_index = [-1]

    if p3_bool == True :
         p3_index = get_point_index(p3_x,p3_y)
    else :
         p3_index = [-1]

    if p4_bool == True :
         p4_index = get_point_index(p4_x,p4_y)
    else :
         p4_index = [-1]

    return [p1_index, p2_index, p3_index, p4_index]

def get_adjacency_list(v_list, obs_list):
     # print(v_list)
     v_neigh = g.neighborhood(vertices = v_list)
     # print(v_neigh[0])
     resultList = [element for nestedlist in v_neigh for element in nestedlist]
     # print(resultList)
     return list(set(resultList) - set(v_list) - set(obs_list))


    
g = ig.Graph(n=size)

# graph Generation
for i in range(size):
     neigh = get_adjacent_nodes(i)
     for j in range(4):
          if neigh[j][0] != -1 and g.are_connected(i,neigh[j][0]) == False:
               g.add_edge(i,neigh[j][0])
# print(get_adjacency_list([0,1,2],[3,4]))
# print(g.get_all_shortest_paths(2, to = 7))
#visited node
xm = [0]
ym = [0]
# print(g.neighborhood(vertices = [7]))
## Subgraph function
obstacle = [3]
visited_nodes = [0]
obstacle_detected = []
node_pointer = visited_nodes[-1]
visited = g.subgraph(visited_nodes)


while get_adjacency_list(visited_nodes, obstacle_detected) != []:
     min_adj = min(get_adjacency_list(visited_nodes, obstacle_detected))
     print(min_adj)
     # visited.add_vertex(min_adj)
     l  = g.get_all_shortest_paths(visited_nodes[-1], to = min_adj)
     print(l)
     # visited.delete_vertices(min_adj)
     if l[0][-1] in obstacle:
          obstacle_detected.append(l[0][-1])
          visited_nodes.extend(l[0][1:-1])
     else:
          visited_nodes.extend(l[0][1:])


print(visited_nodes)
print(obstacle_detected)
node_coord = [[x[i],y[i]] for i in range(size)]
visual_style = {}
visual_style["edge_width"] = [0.5]
g.vs["vertex_label_dist"] = 10
layout_subgraph = ig.Layout(coords=node_coord)
ig.plot(g, target=ax, layout = layout_subgraph,**visual_style)
plt.plot(x,y,color='blue')
plt.show()

#### Create grid ######

# plt.clf()
# plt.xlim(xmin, xmax)
# plt.ylim(ymin, ymax)
# # params
# m = min(xmin, ymin)
# M = max(xmax, ymax)
# # add : plt.style.context('dark_background') for dark mode plotting
# y = np.array([M, m])
# x = np.array([m, M])
# for i in range(m, M+1):
#     for k in range(m, M+1):
#         plt.plot(np.array([i, i]), y, 'k-')
#         plt.plot(x, np.array([k, k]), 'k-')
# plt.show()
