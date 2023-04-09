import igraph as ig
import matplotlib.pyplot as plt

# from igraph import *
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
gt = ig.Graph(n=4)
# layout = gt.layout("kk")
gt.add_edges([(1,2),(2,3)])
gtt= gt.subgraph([0,1])
# neis = gt.neighbors(1)
# neis = gt.vs[1,2].neighbors()
neis = gt.neighborhood(vertices = [0,1,2])
resultList = [element for nestedlist in neis for element in nestedlist]
print(list(dict.fromkeys(resultList)))

# neis = gt.vs[0].neighbors()
# neis = gt.neighbors(0)
# print(neis[0])
# v=[(1,2)]
# print('yess')
# if v in gt.es:

# print(gt.degree(1))

glayout = gt.layout()
components = gt.connected_components(mode='weak')


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


gt.plot()