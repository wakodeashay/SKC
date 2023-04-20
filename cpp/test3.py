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

def get_req_path(subgraph_list, l):
     # print(subgraph_list)
     k = np.array(l)
     # print(k)
     k = k[:,0:-1]
     # print(k)
     n = np.shape(k)[0]
     # print(n)
     if n ==   1:
          return k
     else:
          for i in range(n):
               # print(k[i])
               if all(item in subgraph_list for item in k[i]):
                    # print("I'm here")
                    return k[i]

# print(get_req_path([0,1,2,7,5,6],[[5, 6, 9, 8], [5, 4, 7, 8], [5, 6, 7, 8]]))

# print("--------------------")
# def add_adjacent_vertex(sub_graph, vertex):
#      v = get_adjacent_nodes(vertex)
#      for i in v:
#           print(sub_graph.vs.indices)
#           print(i[0])
#           print("--------")
#           if i!=-1 and i[0] in sub_graph.vs.indices:
#                print("yes")
#                sub_graph.add_edge(vertex, i[0])
    
g = ig.Graph(n=size)

# graph Generation
for i in range(size):
     neigh = get_adjacent_nodes(i)
     for j in range(4):
          if neigh[j][0] != -1 and g.are_connected(i,neigh[j][0]) == False:
               g.add_edge(i,neigh[j][0])
g.vs["name"] = [ str(i) for i in range(size)]
g.vs["label"] = g.vs["name"]
o = g.subgraph([0,1,2,3,4,5,6,7, 8,9, 15, 12, 13])
# o= subgraph.edges(g, [0,1,2,3,4,5,6,7, 8,9, 15, 12, 13], delete.vertices = True)
print(get_adjacent_nodes(15))
# sub_grap = [0,1,2,3,4,5,6,7, 8,9, 15, 12, 13]
# o = ig.Graph()
# for i in sub_grap:
#      neigh = get_adjacent_nodes(i)
#      for j in range(4):
#           if neigh[j][0] != -1 and g.are_connected(i,neigh[j][0]) == True:
#                o.add_edge(i,neigh[j][0])


xm = [0]
ym = [0]

node_coord = [[x[i],y[i]] for i in range(size)]
visual_style = {}
visual_style["edge_width"] = [0.5]
g.vs["vertex_label_dist"] = 10
layout_subgraph = ig.Layout(coords=node_coord)
ig.plot(o, target=ax, layout = layout_subgraph,**visual_style)
# plt.plot(x,y,color='blue')
plt.show()