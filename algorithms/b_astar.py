#!/usr/bin/python3.11
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from SKC.workspace.workspace import Workspace
from SKC.workspace.obstacle import Obstacle

# Current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Specify directory to save animation
doc_anim_dir = os.path.abspath(os.path.join(current_dir, "../docs/animation"))

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.set_axis_off()


class BAStarRoute:
    def __init__(self, workspace, plot_flag, animate_flag):
        self.workspace = workspace
        self.plot_flag = plot_flag
        self.animate_flag = animate_flag

        self.obstacle_detected = []
        self.all_blocked = False

        self.x_visited = [self.workspace.x_nom[0]]
        self.y_visited = [self.workspace.y_nom[0]]
        self.points_visited = [0]

        self.overall_path = []
        self.bastar_complete = False
        self.get_alternate_path()

        self.agent, = ax.plot([], [], 'o', color='green')
        self.path, = ax.plot([], [], 'g-', linewidth=2)

    def get_point_index(self, xl, yl):
        """Calculates the index of the point on the hilbert's curve"""
        x_index = np.where(self.workspace.x_nom == xl)[0]
        y_index = np.where(self.workspace.y_nom == yl)[0]
        return list(np.intersect1d(x_index, y_index))[0]

    def get_adjacent_nodes(self, i):
        """Outputs the adjacent nodes of a given node"""
        x_i = self.workspace.x_nom[i]
        y_i = self.workspace.y_nom[i]

        adjacent_points = {
            "p1": [x_i + self.workspace.grid, y_i],
            "p2": [x_i, y_i + self.workspace.grid],
            "p3": [x_i - self.workspace.grid, y_i],
            "p4": [x_i, y_i - self.workspace.grid],
        }

        adjacent_nodes = [
            self.get_point_index(adjacent_points[i][0], adjacent_points[i][1])
            if adjacent_points[i][0] in self.workspace.x_nom and adjacent_points[i][1] in self.workspace.y_nom
            else -1
            for i in adjacent_points
        ]

        return adjacent_nodes

    def get_rel_direction(self, origin_point, adjacent_points):
        rel_directions = {}

        for point in adjacent_points:
            if point != -1 and point not in self.obstacle_detected:
                if point not in self.workspace.obstacles:
                    if self.workspace.y_nom[point] > self.workspace.y_nom[origin_point]:
                        rel_directions['north'] = point
                    elif self.workspace.y_nom[point] < self.workspace.y_nom[origin_point]:
                        rel_directions['south'] = point
                    elif self.workspace.x_nom[point] < self.workspace.x_nom[origin_point]:
                        rel_directions['west'] = point
                    elif self.workspace.x_nom[point] > self.workspace.x_nom[origin_point]:
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

        x_i = self.workspace.x_nom[point]
        y_i = self.workspace.y_nom[point]

        adjacent_points = {
            dir[0]: [x_i + self.workspace.grid, y_i],
            dir[1]: [x_i + self.workspace.grid, y_i + self.workspace.grid],
            dir[2]: [x_i, y_i + self.workspace.grid],
            dir[3]: [x_i - self.workspace.grid, y_i + self.workspace.grid],
            dir[4]: [x_i - self.workspace.grid, y_i],
            dir[5]: [x_i - self.workspace.grid, y_i - self.workspace.grid],
            dir[6]: [x_i, y_i - self.workspace.grid],
            dir[7]: [x_i + self.workspace.grid, y_i - self.workspace.grid],
        }

        adjacent_nodes = {
            direction: self.get_point_index(coord[0], coord[1])
            if coord[0] in self.workspace.x_nom and coord[1] in self.workspace.y_nom
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
            if i not in self.workspace.subgraph.vs["name"]:
                neigh = self.get_adjacent_nodes(i)
                for j in range(4):
                    if (
                            neigh[j] != -1
                            and self.workspace.subgraph.are_connected(i, neigh[j]) == False
                            and neigh[j] in self.points_visited
                    ):
                        self.workspace.subgraph.add_edge(i, neigh[j])

    def get_shortest_dist(self, start, end):
        self.get_subgraph()
        shortest_path = self.workspace.subgraph.get_shortest_path(start, to=end, output='vpath')
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
        max_waypoint = max(self.points_visited)
        final_path, _ = self.get_shortest_dist(self.points_visited[-1], max_waypoint)
        if len(final_path) != 1:
            self.overall_path.append(final_path[1:])
            self.points_visited.extend(final_path[1:])
        else:
            pass

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

        for i in self.points_visited:
            self.x_visited.append(self.workspace.x_nom[i])
            self.y_visited.append(self.workspace.y_nom[i])

        plt.plot(self.x_visited[0], self.y_visited[0], marker="o", markersize=10, markeredgecolor="blue",
                 markerfacecolor="blue")
        plt.plot(self.x_visited[-1], self.y_visited[-1], marker="o", markersize=10, markeredgecolor="gold",
                 markerfacecolor="gold")

        for i in range(len(self.overall_path)):
            if i % 2 == 0:
                plt.text(self.workspace.x_nom[self.overall_path[i][0]], self.workspace.y_nom[self.overall_path[i][0]],
                        f'{int(i / 2) + 1},S',
                        fontsize=24, ha='left', va='top',
                        color='red')

                plt.text(self.workspace.x_nom[self.overall_path[i][-1]], self.workspace.y_nom[self.overall_path[i][-1]],
                        f'{int(i / 2) + 1},E',
                        fontsize=24, ha='left', va='top',
                        color='blue')

        # for i, (x, y) in self.workspace.points.items():
        #     plt.text(x, y, f'{i}', fontsize=24, ha='right', va='bottom', color='blue')

        if self.plot_flag:
            plt.plot(self.x_visited, self.y_visited, linestyle="solid", color="green", linewidth=1.0)
            plt.show()

        if self.animate_flag:
            ani = animation.FuncAnimation(fig, self.motion_update, frames=len(self.points_visited),
                                          interval=100, blit=True)
            ani.save(os.path.join(doc_anim_dir, 'bstar_animation.gif'), writer='ffmpeg', fps=5, dpi=300)

    def init(self):
        self.agent.set_data([], [])
        self.path.set_data([], [])
        return self.agent, self.path

    def motion_update(self, i):
        self.agent.set_data(self.workspace.x_nom[self.points_visited[i]], self.workspace.y_nom[self.points_visited[i]])
        self.path.set_data(self.workspace.x_nom[:self.points_visited[i] + 1],
                           self.workspace.y_nom[:self.points_visited[i] + 1])
        return self.agent, self.path


if __name__ == "__main__":
    iteration = 3
    side_size = 2 ** iteration
    obstacle = Obstacle(side_size, 0.2, 0.5)
    boustro_workspace = Workspace('boustro', iteration, obstacle)
    bastar = BAStarRoute(boustro_workspace, True, False)
    bastar.plot()
