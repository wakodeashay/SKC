#!/usr/bin/python3.11
import os
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


class BAStarRoute:
    def __init__(self, side_size, obstacle, plot_flag, animate_flag):
        self.side_size = side_size
        self.grid_size = self.side_size * self.side_size
        self.obstacle_list = obstacle
        self.plot_flag = plot_flag
        self.animate_flag = animate_flag

        # Generate Graph
        self.g = ig.Graph(n=self.grid_size)
        self.subgraph = ig.Graph(n=self.grid_size)
        self.subgraph.vs["name"] = [str(i) for i in range(self.grid_size)]
        self.subgraph.vs["label"] = self.subgraph.vs["name"]

        self.obstacle_detected = []
        self.all_blocked = False

        # Area covered by the Hilbert's curve
        self.xmin = 0
        self.ymin = 0
        # Grid size
        self.grid = 1

        # Points
        self.points = {}

        # Populate point object
        self.get_coordinates()
        self.coordinate = np.array(list(self.points.values()))

        self.x_nom = self.coordinate[:, 0]
        self.y_nom = self.coordinate[:, 1]

        self.x_visited = np.array([self.x_nom[0]])
        self.y_visited = np.array([self.y_nom[0]])
        self.points_visited = [0]

        self.overall_path = []
        self.bastar_complete = False
        self.generate_graph()
        self.get_alternate_path()

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

    def get_point_index(self, xl, yl):
        """Calculates the index of the point on the hilbert's curve"""
        x_index = np.where(self.x_nom == xl)[0]
        y_index = np.where(self.y_nom == yl)[0]
        return list(np.intersect1d(x_index, y_index))[0]

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
            else -1
            for i in adjacent_points
        ]

        return adjacent_nodes

    def generate_graph(self):
        # graph Generation
        for i in range(self.grid_size):
            neigh = self.get_adjacent_nodes(i)
            # neigh = self.adj_dict[i]
            for j in range(4):
                if neigh[j] != -1 and self.g.are_connected(i, neigh[j]) == False:
                    self.g.add_edge(i, neigh[j])

    def get_rel_direction(self, origin_point, adjacent_points):
        rel_directions = {}

        for point in adjacent_points:
            if point != -1 and point not in self.obstacle_detected:
                if point not in self.obstacle_list:
                    if self.y_nom[point] > self.y_nom[origin_point]:
                        rel_directions['north'] = point
                    elif self.y_nom[point] < self.y_nom[origin_point]:
                        rel_directions['south'] = point
                    elif self.x_nom[point] < self.x_nom[origin_point]:
                        rel_directions['west'] = point
                    elif self.x_nom[point] > self.x_nom[origin_point]:
                        rel_directions['east'] = point
                else:
                    self.obstacle_detected.append(point)

        return rel_directions

    def get_next_point(self, last_point_visited):
        adjacent_points = self.get_adjacent_nodes(last_point_visited)
        directed_adjacent_points = self.get_rel_direction(last_point_visited, adjacent_points)

        # Priority check: north, south, east, west
        movement_priority = ['north', 'south', 'east', 'west']

        for direction in movement_priority:
            if direction in directed_adjacent_points:
                next_point = directed_adjacent_points[direction]
                if next_point not in self.points_visited:
                    return next_point
        return None

    def get_all_neighbors(self, point):
        """Outputs all the adjacent nodes of a given node"""
        dir = ['east', 'north-east', 'north', 'north-west', 'west', 'south-west', 'south', 'south-east']

        x_i = self.x_nom[point]
        y_i = self.y_nom[point]

        adjacent_points = {
            dir[0]: [x_i + self.grid, y_i],
            dir[1]: [x_i + self.grid, y_i + self.grid],
            dir[2]: [x_i, y_i + self.grid],
            dir[3]: [x_i - self.grid, y_i + self.grid],
            dir[4]: [x_i - self.grid, y_i],
            dir[5]: [x_i - self.grid, y_i - self.grid],
            dir[6]: [x_i, y_i - self.grid],
            dir[7]: [x_i + self.grid, y_i - self.grid],
        }

        adjacent_nodes = {
            direction: self.get_point_index(coord[0], coord[1])
            if coord[0] in self.x_nom and coord[1] in self.y_nom
            else -1
            for direction, coord in adjacent_points.items()
        }

        return adjacent_nodes

    def b_value(self, point1, point2):
        if point1 not in self.obstacle_detected and point1 not in self.points_visited and point1 != -1 and (
                point2 in self.obstacle_detected or point2 == -1):
            return 1
        else:
            return 0

    def sum_function(self, point):
        all_neighbors = self.get_all_neighbors(point)

        sum_function = self.b_value(all_neighbors['east'], all_neighbors['south-east']) + self.b_value(
            all_neighbors['east'], all_neighbors['north-east']) + self.b_value(all_neighbors['west'], all_neighbors[
            'south-west']) + self.b_value(all_neighbors['west'], all_neighbors['north-west']) + self.b_value(
            all_neighbors['south'], all_neighbors['south-west']) + self.b_value(all_neighbors['south'],
                                                                                all_neighbors['south-east'])

        return sum_function

    def get_subgraph(self):
        for i in self.points_visited:
            if i not in self.subgraph.vs["name"]:
                neigh = self.get_adjacent_nodes(i)
                # neigh = self.adj_dict[i]
                for j in range(4):
                    if (
                            neigh[j] != -1
                            and self.subgraph.are_connected(i, neigh[j]) == False
                            and neigh[j] in self.points_visited
                    ):
                        self.subgraph.add_edge(i, neigh[j])

    def get_shortest_dist(self, start, end):
        self.get_subgraph()
        shortest_path = self.subgraph.get_shortest_path(start, to=end, output='vpath')
        return shortest_path, len(shortest_path) - 1

    def get_backtrack_path(self, last_bpath):
        poss_backtracks = []
        poss_backtracks_shortest_path = []
        poss_backtracks_dist = []

        for point in last_bpath:
            if self.sum_function(point) >= 1:
                poss_backtracks.append(point)
                shortest_path, shortest_path_len = self.get_shortest_dist(last_bpath[-1], point)
                poss_backtracks_shortest_path.append(shortest_path)
                poss_backtracks_dist.append(shortest_path_len)

        if not poss_backtracks_dist:
            return None

        min_index = poss_backtracks_dist.index(min(poss_backtracks_dist))
        return poss_backtracks_shortest_path[min_index]

    def get_free_adjacent_point(self, point):
        adjacent_points = self.get_adjacent_nodes(point)
        for point in adjacent_points:
            if point != -1 and point not in self.points_visited and point not in self.obstacle_detected:
                return point

        return None

    def get_alternate_path(self):
        while not self.bastar_complete:
            while not self.all_blocked:
                if self.get_next_point(self.points_visited[-1]) is None:
                    break
                self.points_visited.append(self.get_next_point(self.points_visited[-1]))

            prev_len = sum(len(sublist) for sublist in self.overall_path)
            self.overall_path.append(self.points_visited[prev_len:])

            back_track_path = self.get_backtrack_path(self.points_visited)

            if back_track_path is None:
                print('BASTAR trajectory is complete!!')
                break

            back_track_path = back_track_path[1:]

            self.overall_path.append(back_track_path)
            self.points_visited.extend(back_track_path)

            free_adjacent_point = self.get_free_adjacent_point(self.points_visited[-1])

            self.points_visited.append(free_adjacent_point)
            if free_adjacent_point is None:
                print('Backtracking point dont have free adjacent points')
                break

        prev_len = sum(len(sublist) for sublist in self.overall_path)
        self.overall_path.append(self.points_visited[prev_len:])

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

        self.node_coord = [[self.x_nom[i], self.y_nom[i]] for i in range(self.grid_size)]
        self.visual_style = {}
        self.g.vs["name"] = [str(i) for i in range(self.grid_size)]

        self.visual_style["edge_width"] = [0.3]
        self.visual_style["vertex_size"] = 10
        self.visual_style["edge_color"] = "orange"
        self.layout_subgraph = ig.Layout(coords=self.node_coord)

        ig.plot(self.g, target=ax, layout=self.layout_subgraph, **self.visual_style)

        x = list([self.x_nom[i] for i in self.points_visited])
        y = list([self.y_nom[i] for i in self.points_visited])
        print(self.points_visited)
        for i in range(len(self.overall_path)):
            if i % 2 == 0:
                ax.text(self.x_nom[self.overall_path[i][0]], self.y_nom[self.overall_path[i][0]], f'{int(i / 2) + 1},S',
                        fontsize=24, ha='left', va='top',
                        color='red')

                ax.text(self.x_nom[self.overall_path[i][-1]], self.y_nom[self.overall_path[i][-1]],
                        f'{int(i / 2) + 1},E',
                        fontsize=24, ha='left', va='top',
                        color='blue')

        for i, (x, y) in self.points.items():
            ax.text(x, y, f'{i}', fontsize=24, ha='right', va='bottom', color='blue')

        if self.plot_flag:
            plt.plot(x, y, linestyle="solid", color="green", linewidth=1.0)
            plt.show()

        if self.animate_flag:
            ani = animation.FuncAnimation(fig, self.motion_update, frames=len(self.points_visited),
                                          interval=100, blit=True)
            ani.save(os.path.join(doc_anim_dir, 'animation.gif'), writer='ffmpeg', fps=5, dpi=300)

    def init(self):
        self.agent.set_data([], [])
        self.path.set_data([], [])
        return self.agent, self.path

    def motion_update(self, i):
        self.agent.set_data(self.x_nom[self.points_visited[i]], self.y_nom[self.points_visited[i]])
        self.path.set_data(self.x_nom[:self.points_visited[i] + 1], self.y_nom[:self.points_visited[i] + 1])
        return self.agent, self.path


if __name__ == "__main__":
    obs = np.array([9, 22, 25, 34, 45, 50, 61, 46, 47, 33, 32, 26, 27, 28])
    # obs = np.array([])
    bastar = BAStarRoute(10, obs, True, False)
    bastar.plot()
