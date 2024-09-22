import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import igraph as ig
from maps import Potential

# Current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
doc_anim_dir = os.path.abspath(os.path.join(current_dir, "../docs/animation"))

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.set_axis_off()

class EStar:
    def __init__(self, side_size, obstacle, plot_flag, animate_flag):
        self.y_bound = None
        self.x_bound = None
        self.side_size = side_size
        self.grid_size = self.side_size * self.side_size
        self.epsilon = 1  # ε for grid size
        self.obstacle_list = obstacle  # Initial known obstacles
        self.plot_flag = plot_flag
        self.animate_flag = animate_flag

        self.maps = Potential(self.side_size)

        self.grid = 1
        self.obstacle_detected = []

        self.xmin = 0
        self.ymin = 0

        # # Initialize MAPS with potential values
        # self.potential_map = {}

        # Points
        self.points = {}
        # self.create_map()

        # Populate point object
        self.get_coordinates()
        self.coordinate = np.array(list(self.points.values()))

        self.x_nom = self.coordinate[:, 0]
        self.y_nom = self.coordinate[:, 1]

        self.x_visited = []
        self.y_visited = []
        self.points_visited = [0]

        self.overall_path = []
        self.estar_complete = False

        self.get_path()

        self.agent, = ax.plot([], [], 'o', color='green')
        self.path, = ax.plot([], [], 'g-', linewidth=2)

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

    # def create_map(self):
    #     for i in range(self.grid_size):
    #         col_count = int(i / self.side_size)
    #         self.potential_map[i] = self.side_size + 1 - col_count

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

        unvisited_adjacent_node = []
        for i in adjacent_nodes:
            if i[0] not in self.points_visited and i[0] != -1 and i[0] not in self.points_visited:
                unvisited_adjacent_node.append(i[0])

        return unvisited_adjacent_node

    def check_obstacles(self):


    def get_upper_waypoint(self, waypoint_list):
        y = [i / self.side_size for i in waypoint_list]
        y_max = max(y)
        return waypoint_list[y.index(y_max)]


    def get_path(self):
        while self.estar_complete == False:
            current_neighbors = self.get_adjacent_nodes(self.points_visited[-1])

            if len(current_neighbors) == 0:
                print('Local minima')
                self.estar_complete == True
                break
            else:
                neigh_potential = self.maps.get_potential(current_neighbors)
                max_potential = max(neigh_potential)

                if len(max_potential) == 1:
                    self.points_visited.append(current_neighbors[neigh_potential.index(max_potential)])
                # The cost (Eq 5) remains the same when for lambda_up, lambda_down
                # Choose the waypoint which is higher up the y-axis
                else:
                    max_potential_waypoint = [current_neighbors[neigh_potential.index(i)] for i in max_potential]
                    self.points_visited.append(self.get_upper_waypoint(max_potential_waypoint))


                ## Check Obstacles
                ## Update maps

        print("EStar search completed!")

    def plot_workspace(self):
        self.x_bound = [min(self.x_nom) - self.grid / 2, min(self.x_nom) - self.grid / 2, max(self.x_nom) +
                        self.grid / 2, max(self.x_nom) + self.grid / 2, min(self.x_nom) - self.grid / 2]
        self.y_bound = [min(self.y_nom) - self.grid / 2, max(self.y_nom) + self.grid / 2, max(self.y_nom) +
                        self.grid / 2, min(self.y_nom) - self.grid / 2, min(self.y_nom) - self.grid / 2]

        for i in range(self.grid_size):
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
        for i in self.points_visited:
            self.x_visited.append(self.x_nom[i])
            self.y_visited.append(self.y_nom[i])

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



if __name__ == "__main__":
    obs = np.array([])
    bastar = EStar(8, obs, True, False)
    bastar.plot()
