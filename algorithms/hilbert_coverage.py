#!/usr/bin/python3.11
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Specify directory to save animation
doc_anim_dir = os.path.abspath(os.path.join(current_dir, "../docs/animation"))

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.set_axis_off()


class HilbertRoute:

    def __init__(self, workspace, detection_radius, plot_flag, animate_flag):
        self.node_coord = None
        self.layout_subgraph = None

        self.workspace = workspace
        self.detection_radius = detection_radius
        self.plot_flag = plot_flag
        self.animate_flag = animate_flag

        self.points_visited = [0]
        self.obstacle_detected = np.array([])

        # Initialize the alternate path suggested
        self.x_visited = np.array([])
        self.y_visited = np.array([])

        ## If detection level -1, do contact based obstacle detection
        if self.detection_radius == -1:
            self.get_alt_path_contact()
            pass
        else:
            # self.get_alt_path_detection_level(self.detection_radius)
            pass

        self.agent, = ax.plot([], [], 'o', color='blue')
        self.path, = ax.plot([], [], 'g-', linewidth=5)

    def get_adjacency_list(self, v_list, obs_list):
        """Gives the Adjacency list"""
        v_neigh = self.workspace.g.neighborhood(vertices=v_list)
        result_set = {v for neigh_list in v_neigh for v in neigh_list} - \
                     set(v_list) - set(obs_list)
        return list(result_set)

    def get_req_path(self, subgraph_list, l):
        """Gives the smallest path present within the subgraph"""
        k = np.array(l)
        if np.shape(k)[0] == 1:
            return k
        k = k[:, 0:-1]
        for i in range(self.workspace.grid_size):
            if set(k[i]).intersection(subgraph_list) == set(k[i]):
                return k[i]

    def generate_subgraph(self, v, new_vertex):
        """Generates subgraph for vertex list v"""
        neigh = self.workspace.get_adjacent_nodes(new_vertex)
        for j in range(4):
            if (
                    neigh[j][0] != -1
                    and not self.workspace.subgraph.are_connected(new_vertex, neigh[j][0])
                    and neigh[j][0] in v
            ):
                self.workspace.subgraph.add_edge(new_vertex, neigh[j][0])
    def update_detected_obstacle(self):
        neigh = self.workspace.get_adjacent_nodes(self.points_visited[-1])
        for v in neigh:
            if v in self.workspace.obstacles and v not in self.obstacle_detected.tolist():
                self.obstacle_detected = np.append(self.obstacle_detected, v)


    # def get_alt_path_detection_level(self):
    #     self.generate_graph()
    #     self.generate_subgraph(self.points_visited)
    #     while not self.get_adjacency_list(self.points_visited, self.obstacle_detected) == []:
    #         # self.update_detected_obstacle()
    #         min_adj = min(self.get_adjacency_list(self.points_visited, self.obstacle_detected))
    #         self.generate_subgraph(np.append(self.points_visited, min_adj))
    #         if min_adj == self.points_visited[-1] + 1:
    #             if min_adj in self.obstacle:
    #                 self.obstacle_detected = np.append(self.obstacle_detected, min_adj)
    #             else:
    #                 self.points_visited = np.append(self.points_visited, min_adj)
    #         else:
    #             l = self.get_req_path(
    #                 self.points_visited, self.subgraph.get_all_shortest_paths(
    #                     self.points_visited[-1], to=min_adj)
    #             )
    #             if min_adj in self.obstacle:
    #                 try:
    #                     self.obstacle_detected = np.append(self.obstacle_detected, l[0][-1])
    #                     self.points_visited = np.append(self.points_visited, l[0][1:-1])
    #                 except:
    #                     self.obstacle_detected = np.append(self.obstacle_detected, l[-1])
    #                     self.points_visited = np.append(self.points_visited, l[1:-1])
    #             else:
    #                 try:
    #                     self.points_visited = np.append(self.points_visited, l[0][1:])
    #                 except:
    #                     self.points_visited = np.append(self.points_visited, l[1:])
    #
    #     self.x_visited = [self.x_nom[i] for i in self.points_visited]
    #     self.y_visited = [self.y_nom[i] for i in self.points_visited]
    #     self.x_visited = np.array(self.x_visited)
    #     self.y_visited = np.array(self.y_visited)
    #
    #     path = self.get_path_length(self.points_visited)
    #     rev = self.get_revisits(self.points_visited)
    #     self.max_point = max(self.points_visited)
    #
    #     return self.x_visited, self.y_visited, rev, path, self.max_point, self.points_visited

    def get_alt_path_contact(self):
        self.generate_subgraph(self.points_visited, 0)
        self.update_detected_obstacle()

        while not self.get_adjacency_list(self.points_visited, self.obstacle_detected) == []:
            min_adj = min(self.get_adjacency_list(self.points_visited, self.obstacle_detected))
            self.generate_subgraph(self.points_visited, min_adj)
            if min_adj == self.points_visited[-1] + 1:
                if min_adj in self.workspace.obstacles:
                    if min_adj not in self.workspace.obstacles:
                        self.obstacle_detected = np.append(self.obstacle_detected, min_adj)
                    self.workspace.subgraph.delete_edges(self.workspace.subgraph.incident(min_adj))

                else:
                    self.points_visited = np.append(self.points_visited, min_adj)
            else:
                l = self.get_req_path(
                    self.points_visited, self.workspace.subgraph.get_all_shortest_paths(
                        self.points_visited[-1], to=min_adj)
                )
                if min_adj in self.workspace.obstacles:
                    try:
                        if min_adj not in self.workspace.obstacles:
                            self.obstacle_detected = np.append(self.obstacle_detected, min_adj)
                        self.points_visited = np.append(self.points_visited, l[0][1:-1])
                        self.workspace.subgraph.delete_edges(self.workspace.subgraph.incident(min_adj))
                    except:
                        if min_adj not in self.workspace.obstacles:
                            self.obstacle_detected = np.append(self.obstacle_detected, min_adj)
                        self.points_visited = np.append(self.points_visited, l[1:-1])
                        self.workspace.subgraph.delete_edges(self.workspace.subgraph.incident(min_adj))
                else:
                    try:
                        self.points_visited = np.append(self.points_visited, l[0][1:])
                    except:
                        self.points_visited = np.append(self.points_visited, l[1:])

            self.update_detected_obstacle()

        self.x_visited = [self.workspace.x_nom[i] for i in self.points_visited]
        self.y_visited = [self.workspace.y_nom[i] for i in self.points_visited]
        self.x_visited = np.array(self.x_visited)
        self.y_visited = np.array(self.y_visited)

    def plot_workspace(self):
        self.x_bound = [min(self.workspace.x_nom) - self.workspace.grid / 2,
                        min(self.workspace.x_nom) - self.workspace.grid / 2, max(self.workspace.x_nom) +
                        self.workspace.grid / 2, max(self.workspace.x_nom) + self.workspace.grid / 2,
                        min(self.workspace.x_nom) - self.workspace.grid / 2]
        self.y_bound = [min(self.workspace.y_nom) - self.workspace.grid / 2,
                        max(self.workspace.y_nom) + self.workspace.grid / 2, max(self.workspace.y_nom) +
                        self.workspace.grid / 2, min(self.workspace.y_nom) - self.workspace.grid / 2,
                        min(self.workspace.y_nom) - self.workspace.grid / 2]

        rectangles = []
        for i in range(self.workspace.grid_size):
            color = "red" if i in self.obstacle_detected else "grey"
            alpha = 0.2 if i in self.obstacle_detected else 0.04
            rectangles.append(plt.Rectangle(
                (self.workspace.x_nom[i] - self.workspace.grid / 2,
                 self.workspace.y_nom[i] - self.workspace.grid / 2),
                1, 1, fc=color, alpha=alpha, ec="black"
            ))
        for rect in rectangles:
            plt.gca().add_patch(rect)

    def plot(self):
        self.plot_workspace()

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
            ani.save(os.path.join(doc_anim_dir, 'hilbert_animation.gif'), writer='ffmpeg', fps=5, dpi=300)

    def init(self):
        self.agent.set_data([], [])
        self.path.set_data([], [])
        return self.agent, self.path

    def motion_update(self, i):
        self.agent.set_data(self.x_visited[i], self.y_visited[i])
        self.path.set_data(self.x_visited[:i + 1], self.y_visited[:i + 1])
        return self.agent, self.path
