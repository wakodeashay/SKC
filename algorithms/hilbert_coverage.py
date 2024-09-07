#!/usr/bin/python3.11
import os
from hilbertcurve.hilbertcurve import HilbertCurve
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import igraph as ig

# Current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Specify directory to save animation
doc_anim_dir = os.path.abspath(os.path.join(current_dir, "../docs/animation"))

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.set_axis_off()


class HilbertRoute:

    def __init__(self, iteration, obstacle, detection_level, plot_flag, animate_flag):
        self.node_coord = None
        self.layout_subgraph = None

        self.iter = iteration
        self.obstacle = obstacle
        self.detection_level = detection_level
        self.plot_flag = plot_flag
        self.animate_flag = animate_flag

        self.points_visited = [0]
        self.obstacle_detected = []

        # Dimension
        self.dim = 2

        # Number of points in Hilbert's curve
        self.size = 2 ** (self.iter * self.dim)
        self.side_size = 2 ** (self.iter * self.dim / 2)

        # Generate Graph
        self.g = ig.Graph(n=self.size)
        self.subgraph = ig.Graph(n=self.size)
        self.subgraph.vs["name"] = [str(i) for i in range(self.size)]
        self.subgraph.vs["label"] = self.subgraph.vs["name"]

        # Area covered by the Hilbert's curve
        self.xmin = 0
        self.ymin = 0

        # Grid size
        self.grid = 1

        # Creating Hilbert's curve
        self.hilbert_curve = HilbertCurve(self.iter, self.dim)
        self.distances = list(range(self.size))
        self.points = self.hilbert_curve.points_from_distances(self.distances)

        self.x_nom = np.array([self.points[i][0] for i in range(self.size)])
        self.y_nom = np.array([self.points[i][1] for i in range(self.size)])

        # Initialize the alternate path suggested
        self.x_visited = np.array([])
        self.y_visited = np.array([])

        # Highest numbered waypoint reached
        self.max_point = 0
        self.adj_dict = {}

        ## If detection level -1, do contact based obstacle detection
        if self.detection_level == -1:
            self.get_alt_path_contact()
        else:
            self.get_alt_path_detection_level()

        self.agent, = ax.plot([], [], 'o', color='blue')
        self.path, = ax.plot([], [], 'g-', linewidth=2)

    def update_detected_obstacle(self):
        self.points_visited

    def get_point_index(self, xl, yl):
        """Calculates the index of the point on the hilbert's curve"""
        x_index = np.where(self.x_nom == xl)[0]
        y_index = np.where(self.y_nom == yl)[0]
        return list(np.intersect1d(x_index, y_index))

    def get_adjacent_nodes(self, i):
        """Outputs the adjacent nodes of a given node"""
        x_i = self.x_nom[i]
        y_i = self.y_nom[i]

        adjacent_points = {
            "p1": [x_i + self.grid, y_i],
            "p2": [x_i, y_i + self.grid],
            "p3": [x_i - self.grid, y_i],
            "p4": [x_i, y_i - self.grid],
        }

        adjacent_nodes = [
            self.get_point_index(adjacent_points[point][0], adjacent_points[point][1])
            if adjacent_points[point][0] in self.x_nom and adjacent_points[point][1] in self.y_nom
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

        for i in range(n - 1):
            if node_list[i] != node_list[i + 1]:
                path = path + 1

        return path

    def get_revisits(self, node_list):
        total_visits = len(node_list)
        unique_visits = len(set(node_list))

        return (total_visits - unique_visits) / unique_visits

    def generate_subgraph(self, v):
        """Generates subgraph for vertex list v"""
        for i in v:
            if i not in self.points_visited:
                neigh = self.get_adjacent_nodes(i)
                for j in range(4):
                    if (
                            neigh[j][0] != -1
                            and self.subgraph.are_connected(i, neigh[j][0]) == False
                            and neigh[j][0] in v
                    ):
                        self.subgraph.add_edge(i, neigh[j][0])

    def generate_graph(self):
        # graph Generation
        for i in range(self.size):
            neigh = self.get_adjacent_nodes(i)
            # neigh = self.adj_dict[i]
            for j in range(4):
                if neigh[j][0] != -1 and self.g.are_connected(i, neigh[j][0]) == False:
                    self.g.add_edge(i, neigh[j][0])

    def get_alt_path_detection_level(self):
        self.generate_graph()
        self.generate_subgraph(self.points_visited)
        while not self.get_adjacency_list(self.points_visited, self.obstacle_detected) == []:
            self.update_detected_obstacle()
            min_adj = min(self.get_adjacency_list(self.points_visited, self.obstacle_detected))
            self.generate_subgraph(np.append(self.points_visited, min_adj))
            if min_adj == self.points_visited[-1] + 1:
                if min_adj in self.obstacle:
                    self.obstacle_detected = np.append(self.obstacle_detected, min_adj)
                else:
                    self.points_visited = np.append(self.points_visited, min_adj)
            else:
                l = self.get_req_path(
                    self.points_visited, self.subgraph.get_all_shortest_paths(
                        self.points_visited[-1], to=min_adj)
                )
                if min_adj in self.obstacle:
                    try:
                        self.obstacle_detected = np.append(self.obstacle_detected, l[0][-1])
                        self.points_visited = np.append(self.points_visited, l[0][1:-1])
                    except:
                        self.obstacle_detected = np.append(self.obstacle_detected, l[-1])
                        self.points_visited = np.append(self.points_visited, l[1:-1])
                else:
                    try:
                        self.points_visited = np.append(self.points_visited, l[0][1:])
                    except:
                        self.points_visited = np.append(self.points_visited, l[1:])

        self.x_visited = [self.x_nom[i] for i in self.points_visited]
        self.y_visited = [self.y_nom[i] for i in self.points_visited]
        self.x_visited = np.array(self.x_visited)
        self.y_visited = np.array(self.y_visited)

        path = self.get_path_length(self.points_visited)
        rev = self.get_revisits(self.points_visited)
        self.max_point = max(self.points_visited)

        return self.x_visited, self.y_visited, rev, path, self.max_point, self.points_visited


    def get_alt_path_contact(self):
        self.generate_graph()
        # i = self.get_subgraph(self.points_visited)
        self.generate_subgraph(self.points_visited)

        while not self.get_adjacency_list(self.points_visited, self.obstacle_detected) == []:
            min_adj = min(self.get_adjacency_list(self.points_visited, self.obstacle_detected))
            self.generate_subgraph(np.append(self.points_visited, min_adj))
            if min_adj == self.points_visited[-1] + 1:
                if min_adj in self.obstacle:
                    self.obstacle_detected = np.append(self.obstacle_detected, min_adj)
                else:
                    self.points_visited = np.append(self.points_visited, min_adj)
            else:
                l = self.get_req_path(
                    self.points_visited, self.subgraph.get_all_shortest_paths(
                        self.points_visited[-1], to=min_adj)
                )
                if min_adj in self.obstacle:
                    try:
                        self.obstacle_detected = np.append(self.obstacle_detected, l[0][-1])
                        self.points_visited = np.append(self.points_visited, l[0][1:-1])
                    except:
                        self.obstacle_detected = np.append(self.obstacle_detected, l[-1])
                        self.points_visited = np.append(self.points_visited, l[1:-1])
                else:
                    try:
                        self.points_visited = np.append(self.points_visited, l[0][1:])
                    except:
                        self.points_visited = np.append(self.points_visited, l[1:])

        self.x_visited = [self.x_nom[i] for i in self.points_visited]
        self.y_visited = [self.y_nom[i] for i in self.points_visited]
        self.x_visited = np.array(self.x_visited)
        self.y_visited = np.array(self.y_visited)

        path = self.get_path_length(self.points_visited)
        rev = self.get_revisits(self.points_visited)
        self.max_point = max(self.points_visited)

        return self.x_visited, self.y_visited, rev, path, self.max_point, self.points_visited

    def plot_workspace(self):
        self.x_bound = [min(self.x_nom) - self.grid / 2, min(self.x_nom) - self.grid / 2, max(self.x_nom) +
                        self.grid / 2, max(self.x_nom) + self.grid / 2, min(self.x_nom) - self.grid / 2]
        self.y_bound = [min(self.y_nom) - self.grid / 2, max(self.y_nom) + self.grid / 2, max(self.y_nom) +
                        self.grid / 2, min(self.y_nom) - self.grid / 2, min(self.y_nom) - self.grid / 2]

        for i in range(self.size):
            if i in self.obstacle_detected:
                rectangle = plt.Rectangle(
                    (self.x_nom[i] - self.grid / 2, self.y_nom[i] - self.grid / 2),
                    1,
                    1,
                    fc="red",
                    alpha=0.2,
                    ec="black",
                )
                plt.gca().add_patch(rectangle)
            else:
                rectangle = plt.Rectangle(
                    (self.x_nom[i] - self.grid / 2, self.y_nom[i] - self.grid / 2),
                    1,
                    1,
                    fc="grey",
                    alpha=0.04,
                    ec="black",
                )
                plt.gca().add_patch(rectangle)

    def plot(self):
        self.plot_workspace()

        self.node_coord = [[self.x_nom[i], self.y_nom[i]] for i in range(self.size)]
        self.visual_style = {}
        self.g.vs["name"] = [str(i) for i in range(self.size)]

        self.visual_style["edge_width"] = [0.3]
        self.visual_style["vertex_size"] = 10
        self.visual_style["edge_color"] = "orange"
        self.layout_subgraph = ig.Layout(coords=self.node_coord)

        ig.plot(self.g, target=ax, layout=self.layout_subgraph, **self.visual_style)

        plt.plot(self.x_visited[0], self.y_visited[0], marker="o", markersize=10, markeredgecolor="blue",
                 markerfacecolor="blue")
        plt.plot(self.x_visited[-1], self.y_visited[-1], marker="o", markersize=10, markeredgecolor="gold",
                 markerfacecolor="gold")
        plt.plot(self.x_bound, self.y_bound, linestyle="solid", color="black", linewidth=1)

        if self.plot_flag:
            plt.plot(self.x_visited, self.y_visited, linestyle="solid", color="green", linewidth=1.0)
            plt.show()

        if self.animate_flag:
            ani = animation.FuncAnimation(fig, self.motion_update, frames=len(self.x_visited),
                                          interval=100, blit=True)
            ani.save(os.path.join(doc_anim_dir, 'animation.gif'), writer='ffmpeg', fps=5, dpi=300)

    def init(self):
        self.agent.set_data([], [])
        self.path.set_data([], [])
        return self.agent, self.path

    def motion_update(self, i):
        self.agent.set_data(self.x_visited[i], self.y_visited[i])
        self.path.set_data(self.x_visited[:i + 1], self.y_visited[:i + 1])
        return self.agent, self.path
