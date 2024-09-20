import math
from hilbertcurve.hilbertcurve import HilbertCurve
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt

from SKC.workspace.obstacle import Obstacle

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.set_axis_off()


class Workspace:
    def __init__(self, style, iteration, sparsity_order, block_fraction):
        if style not in ["boustro", "hilbert"]:
            raise ValueError("Style must be one of 'boustro', 'hilbert'")
        else:
            self.style = style

        self.iteration = iteration
        self.sparsity_order = sparsity_order
        self.block_fraction = block_fraction

        self.grid_size = 2 ** (2*self.iteration)
        self.side_size = 2 ** self.iteration

        self.xmin = 0
        self.ymin = 0
        self.grid = 1

        self.points = {}
        self.x_nom = None
        self.y_nom = None

        # Generate Graph nominal waypoints
        self.g = ig.Graph(n=self.grid_size)
        self.subgraph = ig.Graph(n=self.grid_size)
        self.subgraph.vs["name"] = [str(i) for i in range(self.grid_size)]
        self.subgraph.vs["label"] = self.subgraph.vs["name"]

        if self.style == "boustro":
            self.generate_boustro_coordinates()
        elif self.style == "hilbert":
            self.generate_hilbert_coordinates()

        # Add edges to the graph
        self.generate_waypoint_graph()

        # Add obstacle to the space
        obstacle_create = Obstacle(self.side_size, self.block_fraction, self.sparsity_order)
        self.obstacle_grid = obstacle_create.generate_grid()
        self.obstacles = self.get_blocked_waypoints(self.obstacle_grid)

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

    def generate_waypoint_graph(self):
        for i in range(self.grid_size):
            neigh = self.get_adjacent_nodes(i)
            for j in range(4):
                if neigh[j][0] != -1 and self.g.are_connected(i, neigh[j][0]) == False:
                    self.g.add_edge(i, neigh[j][0])

    def generate_boustro_coordinates(self):
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

        coordinate = np.array(list(self.points.values()))

        self.x_nom = coordinate[:, 0]
        self.y_nom = coordinate[:, 1]

    def generate_hilbert_coordinates(self):
        hilbert_curve = HilbertCurve(self.iteration, 2)  # Dimension assumed to be 2
        distances = list(range(self.grid_size))
        points = hilbert_curve.points_from_distances(distances)

        self.x_nom = np.array([points[i][0] for i in range(self.grid_size)])
        self.y_nom = np.array([points[i][1] for i in range(self.grid_size)])

        for i in range(self.iteration):
            self.points[i] = np.array([self.x_nom, self.y_nom])

    def get_blocked_waypoints(self, grid):
        obstacle_ls = []
        for i in range(self.side_size):
            for j in range(self.side_size):
                if grid[i, j] == 1:
                    obstacle_ls.append(self.get_point_index(j, i)[0])
        return obstacle_ls

    def plot_workspace(self):
        """
        Visualizes the generated grid.
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(self.obstacle_grid, cmap='Greys', origin='lower')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()


if __name__ == "__main__":
    ## Hilbert test
    # hilbert_workspace = Workspace('hilbert', 2, 1.0, 0.4)
    # hilbert_workspace.plot_workspace()
    # print(hilbert_workspace.obstacles)

    ## Boustro Test
    boustro_workspace = Workspace('boustro', 2, 1.0, 0.4)
    boustro_workspace.plot_workspace()
    print(boustro_workspace.obstacles)
