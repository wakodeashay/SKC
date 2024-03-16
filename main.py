#!/usr/bin/python3.11
from hilbertcurve.hilbertcurve import HilbertCurve
import matplotlib.pyplot as plt
import numpy as np
import igraph as ig

fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')

class Route:

    def __init__(self, iteration, obstacle, start):
        self.iter = iteration
        self.obstacle = obstacle
        self.start = start

        # Visited nodes
        # Initialize visited_node list with the starting node
        self.visited_nodes = np.array([self.start])
        # Initialize detected obstacle array
        self.obstacle_detected = np.array([])

        # Dimension 
        self.dim = 2
        
        # Number of points in Hilbert's curve
        self.size = 2 ** (self.iter * self.dim)
        self.side_size = 2 ** (self.iter * self.dim / 2)

        # Generate Graph
        self.g = ig.Graph(n=self.size)

        # Area covered by the Hilbert's curve
        self.xmin = 0
        self.ymin = 0
        self.xmax = 10 * self.side_size
        self.ymax = 10 * self.side_size
        # Grid size
        self.xgrid = 1
        self.ygrid = 1
        # Bounding grid
        self.xmin_grid = self.xmin - self.xgrid / 2
        self.ymin_grid = self.ymin - self.ygrid / 2
        self.xmax_grid = self.xmax + self.xgrid / 2
        self.ymax_grid = self.ymax + self.ygrid / 2

        # Creating Hilbert's curve
        self.hilbert_curve = HilbertCurve(self.iter, self.dim)
        self.distances = list(range(self.size))
        self.points = self.hilbert_curve.points_from_distances(self.distances)

        self.x = np.array([self.points[i][0] for i in range(self.size)])
        self.y = np.array([self.points[i][1] for i in range(self.size)])

        # Initialize the alternate path suggested
        self.x_visited = np.array([])
        self.y_visited = np.array([])

        # Highest numbered waypoint reached
        self.max_point = 0


    def get_point_index(self, xl, yl):
        """Calculates the index of the point on the hilbert's curve"""
        x_index = np.where(self.x == xl)[0]
        y_index = np.where(self.y == yl)[0]
        return list(np.intersect1d(x_index, y_index))


    def get_adjacent_nodes(self, i):
        """Outputs the adjacent nodes of a given node"""
        x_i = self.x[i]
        y_i = self.y[i]

        adjacent_points = {
            "p1": [x_i + self.xgrid, y_i],
            "p2": [x_i, y_i + self.ygrid],
            "p3": [x_i - self.xgrid, y_i],
            "p4": [x_i, y_i - self.ygrid],
        }

        adjacent_nodes = [
            self.get_point_index(adjacent_points[point][0], adjacent_points[point][1])
            if adjacent_points[point][0] in self.x and adjacent_points[point][1] in self.y
            else [-1]
            for point in adjacent_points
        ]

        return adjacent_nodes


    def get_adjacency_list(self, v_list, obs_list):
        """Gives the Adjacency list"""
        v_neigh = self.g.neighborhood(vertices=v_list)
        result_set = {v for neigh_list in v_neigh for v in neigh_list} - \
                    set(v_list) - set(obs_list)
        return list(result_set)


    def get_req_path(self, subgraph_list, l):
        """Gives the smallest path present within the subgraph"""
        k = np.array(l)
        if np.shape(k)[0] == 1:
            return k
        k = k[:, 0:-1]
        for i in range(self.size):
            if set(k[i]).intersection(subgraph_list) == set(k[i]):
                return k[i]

    
    def get_path_length(self, node_list):
        n = len(node_list)
        path = 0

        for i in range(n-1):
            if (node_list[i] != node_list[i+1]):
                path = path + 1
        
        return path
    

    def get_revisits(self, node_list):
        total_visits = len(node_list)
        unique_visits = len(set(node_list))

        return (total_visits - unique_visits)/unique_visits


    def get_subgraph(self, v):
        """Generates subgraph for vertex list v"""
        o = ig.Graph(n=self.size)
        for i in v:
            neigh = self.get_adjacent_nodes(i)
            for j in range(4):
                if (
                        neigh[j][0] != -1
                        and o.are_connected(i, neigh[j][0]) == False
                        and neigh[j][0] in v
                ):
                    o.add_edge(i, neigh[j][0])
        o.vs["name"] = [str(i) for i in range(self.size)]
        o.vs["label"] = o.vs["name"]
        return o


    def generate_graph(self):
        # graph Generation
        for i in range(self.size):
            neigh = self.get_adjacent_nodes(i)
            for j in range(4):
                if neigh[j][0] != -1 and self.g.are_connected(i, neigh[j][0]) == False:
                    self.g.add_edge(i, neigh[j][0])


    def get_alt_path(self):
        self.generate_graph()
        i = self.get_subgraph(self.visited_nodes)

        while not self.get_adjacency_list(self.visited_nodes, self.obstacle_detected) == []:
            min_adj = min(self.get_adjacency_list(self.visited_nodes, self.obstacle_detected))
            o = self.get_subgraph(np.append(self.visited_nodes, min_adj))
            if min_adj == self.visited_nodes[-1] + 1:
                if min_adj in self.obstacle:
                    self.obstacle_detected = np.append(self.obstacle_detected, min_adj)
                else:
                    self.visited_nodes = np.append(self.visited_nodes, min_adj)
            else:
                l = self.get_req_path(
                    self.visited_nodes, o.get_all_shortest_paths(
                        self.visited_nodes[-1], to=min_adj)
                )
                if min_adj in self.obstacle:
                    try:
                        self.obstacle_detected = np.append(self.obstacle_detected, l[0][-1])
                        self.visited_nodes = np.append(self.visited_nodes, l[0][1:-1])
                    except:
                        self.obstacle_detected = np.append(self.obstacle_detected, l[-1])
                        self.visited_nodes = np.append(self.visited_nodes, l[1:-1])
                else:
                    try:
                        self.visited_nodes = np.append(self.visited_nodes, l[0][1:])
                    except:
                        self.visited_nodes = np.append(self.visited_nodes, l[1:])


        self.x_visited = [self.x[i] for i in self.visited_nodes]
        self.y_visited = [self.y[i] for i in self.visited_nodes]
        self.x_visited = np.array(self.x_visited)
        self.y_visited = np.array(self.y_visited)

        path  = self.get_path_length(self.visited_nodes)
        rev = self.get_revisits(self.visited_nodes)
        self.max_point = max(self.visited_nodes)

        return self.x_visited, self.y_visited, rev, path, self.max_point


    def plot(self):
        self.get_alt_path()

        x_bound = [min(self.x) - self.xgrid / 2, min(self.x) - self.xgrid / 2, max(self.x) +
                self.xgrid / 2, max(self.x) + self.xgrid / 2, min(self.x) - self.xgrid / 2]
        y_bound = [min(self.y) - self.ygrid / 2, max(self.y) + self.ygrid / 2, max(self.y) +
                self.ygrid / 2, min(self.y) - self.ygrid / 2, min(self.y) - self.ygrid / 2]

        for i in range(self.size):
            if i in self.obstacle_detected:
                rectangle = plt.Rectangle(
                    (self.x[i] - self.xgrid / 2, self.y[i] - self.ygrid / 2),
                    1,
                    1,
                    fc="red",
                    alpha=0.2,
                    ec="black",
                )
                plt.gca().add_patch(rectangle)
            else:
                rectangle = plt.Rectangle(
                    (self.x[i] - self.xgrid / 2, self.y[i] - self.ygrid / 2),
                    1,
                    1,
                    fc="grey",
                    alpha=0.04,
                    ec="black",
                )
                plt.gca().add_patch(rectangle)

        node_coord = [[self.x[i], self.y[i]] for i in range(self.size)]
        visual_style = {}
        self.g.vs["name"] = [str(i) for i in range(self.size)]

        visual_style["edge_width"] = [0.3]
        visual_style["vertex_size"] = 10
        visual_style["edge_color"] = "orange"
        layout_subgraph = ig.Layout(coords=node_coord)

        ig.plot(self.g, target=ax, layout=layout_subgraph, **visual_style)
        plt.plot(self.x_visited[0], self.y_visited[0], marker="o", markersize=15, markeredgecolor="blue", markerfacecolor="blue")
        plt.plot(self.x_visited[-1], self.y_visited[-1], marker="o", markersize=15, markeredgecolor="gold", markerfacecolor="gold")
        plt.plot(x_bound, y_bound, linestyle="solid", color="black", linewidth=0.5)
        plt.plot(self.x_visited, self.y_visited, linestyle="solid", color="green", linewidth=1.0)
        plt.show()


if __name__ == "__main__":

    # investigation example iteration = 3
    # obstacle = np.array([32,33,46,47,48,51,52,53])
    # investigation example iteration = 5
    # obstacle = np.array([130,134,135,191,190,192, 193, 194, 195, 184,185, 188, 189 ,204, 205, 209, 210, 212, 215, 216])
    # Example 1 Normal; iteration = 3
    # obstacle = np.array([22, 23,24,25, 49,50,55,56,59,60,61,62,63])
    # Example 2 Sparse; iteration = 5

    # Non-Uniform coverage example
    # Iteration 4
    # obstacle = np.array([40, 41, 42, 43, 52, 53, 54, 55, 92, 93, 94, 95, 164, 165, 166, 167, 168, 169,
    # 170, 171, 172, 173, 174, 175, 176, 177, 209, 210, 221, 222, 252, 253, 254, 255, 242, 241, 240])
    # Iteration 2
    # obstacle = np.array([12, 15])
    # Iteration 1
    # obstacle = np.array([1])

    obs = np.array([3, 4, 9, 34])
    skc = Route(3, obs, 0)
    skc.plot()

