#!/usr/bin/python3.11
# from hilbertcurve.hilbertcurve import HilbertCurve
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import igraph as ig

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.set_axis_off()


class BAStarRoute:

    def __init__(self, side_size):
        self.side_size = side_size
        self.grid_size = self.side_size * self.side_size

        self.obstacle_detected = np.array([])

        # Generate Graph
        self.g = ig.Graph(n=self.grid_size)

        # Area covered by the Hilbert's curve
        self.xmin = 0
        self.ymin = 0
        # Grid size
        self.grid = 1
        # # Maximum x and y
        # self.xmax = self.grid * (self.side_size - 1)
        # self.ymax = self.grid * (self.side_size - 1)

        # Points
        self.points = {}

        # Populate point object
        self.get_coordinates()
        self.coordinate = np.array(list(self.points.values()))

        self.x_nom = self.coordinate[:, 0]
        self.y_nom = self.coordinate[:, 1]

        print(self.points)
        print(self.x_nom)
        print(self.y_nom)

        self.generate_graph()

    def get_coordinates(self):
        self.points[0] = np.array([self.xmin, self.ymin])

        for i in range(1, self.grid_size):
            col_count = int(i / self.side_size)
            row_count = i % self.side_size

            if row_count == 0:
                self.points[i] = np.array([self.points[i - 1][0] + self.grid, self.points[i - 1][1]])
            else:
                if col_count % 2 == 0:
                    self.points[i] = np.array([self.points[i - 1][0], self.points[i - 1][1] + self.grid])
                else:
                    self.points[i] = np.array([self.points[i - 1][0], self.points[i - 1][1] - self.grid])

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
            self.get_point_index(adjacent_points[i][0], adjacent_points[i][1])
            if adjacent_points[i][0] in self.x_nom and adjacent_points[i][1] in self.y_nom
            else [-1]
            for i in adjacent_points
        ]

        return adjacent_nodes

    def generate_graph(self):
        # graph Generation
        for i in range(self.grid_size):
            neigh = self.get_adjacent_nodes(i)
            # neigh = self.adj_dict[i]
            for j in range(4):
                if neigh[j][0] != -1 and self.g.are_connected(i, neigh[j][0]) == False:
                    self.g.add_edge(i, neigh[j][0])

    def plot_workspace(self):
        self.x_bound = [min(self.x_nom) - self.grid / 2, min(self.x_nom) - self.grid / 2, max(self.x_nom) +
                        self.grid / 2, max(self.x_nom) + self.grid / 2, min(self.x_nom) - self.grid / 2]
        self.y_bound = [min(self.y_nom) - self.grid / 2, max(self.y_nom) + self.grid / 2, max(self.y_nom) +
                        self.grid / 2, min(self.y_nom) - self.grid / 2, min(self.y_nom) - self.grid / 2]

        for i in range(self.grid_size):
            if i in self.obstacle_detected:
                rectangle = plt.Rectangle(
                    (self.x_nom[i] - self.grid / 2, self.y[i] - self.grid / 2),
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

        self.node_coord = [[self.x_nom[i], self.y_nom[i]] for i in range(self.grid_size)]
        self.visual_style = {}
        self.g.vs["name"] = [str(i) for i in range(self.grid_size)]

        self.visual_style["edge_width"] = [0.3]
        self.visual_style["vertex_size"] = 10
        self.visual_style["edge_color"] = "orange"
        self.layout_subgraph = ig.Layout(coords=self.node_coord)

        ig.plot(self.g, target=ax, layout=self.layout_subgraph, **self.visual_style)

        for i, (x, y) in self.points.items():
            ax.text(x, y, f'{i}', fontsize=24, ha='right', va='bottom', color='blue')
        plt.show()


if __name__ == "__main__":
    bastar = BAStarRoute(3)
    bastar.plot()
