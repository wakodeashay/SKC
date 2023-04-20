import igraph as ig
import matplotlib.pyplot as plt
import random

# random.seed(0)
# g = ig.Graph.GRG(50, 0.15)
gt = ig.Graph(n=8)
# layout = gt.layout("kk")
gt.add_edges([(0, 1),(1,2),(1,3),(4,2),(3,5),(4,5), (5,6), (1,7), (5,7)])
sub = gt.subgraph([0,1,2,3,4,5,6])
glayout = sub.layout()
# lay_coor = ig.Layout()
# print(gt.distances([0]))
print(gt.vs(7))
# l = gt.get_all_shortest_paths(1, to= 5)
c= []
sub.add_vertex(gt.vs(7))
# for i in gt.vs():
#     c.append(i)
# print(gt.vs.indices)
# print(l[0][-2])
# x=[0,1,2]
# y= [0,1,-2]
# glayout.coords[0][:] = x
# glayout.coords[1][:] = y
components = sub.connected_components(mode='weak')

fig, ax = plt.subplots()
ig.plot(
    components,
    target=ax,
    layout =glayout,
    palette=ig.RainbowPalette(),
    vertex_size=0.07,
    vertex_color=list(map(int, ig.rescale(components.membership, (0, 200), clamp=True))),
    edge_width=0.7
)
plt.show()