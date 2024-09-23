import numpy as np
import matplotlib.pyplot as plt

class ObstacleSensor:
    def __init__(self, sensor_radius, grid):
        self.sensor_radius = sensor_radius
        self.grid = grid
        self.grid_height, self.grid_width = self.grid.shape

    def line_of_sight(self, robot_position, obstacle_position):
        """
        Determines if there is a clear line of sight between the robot and the obstacle.
        """
        x0, y0 = robot_position
        x1, y1 = obstacle_position

        dx = x1 - x0
        dy = y1 - y0

        steps = int(max(abs(dx), abs(dy)) * 2)  # Increase steps for better sampling

        if steps == 0:
            return True  # Same point

        x_inc = dx / steps
        y_inc = dy / steps

        x, y = x0, y0

        for _ in range(steps):
            x += x_inc
            y += y_inc

            xi = int(round(x))
            yi = int(round(y))

            # Skip the starting point and the obstacle point itself
            if (xi, yi) == robot_position or (xi, yi) == obstacle_position:
                continue

            # Check grid bounds
            if xi < 0 or xi >= self.grid_width or yi < 0 or yi >= self.grid_height:
                continue

            # Check if cell contains an obstacle
            if self.grid[yi, xi] == 1:
                return False  # Line of sight is blocked

        return True  # Line of sight is clear

    def get_visible_obstacles(self, robot_position):
        """
        Determines which obstacles are visible to the robot from its position,
        using a visual line of sight method with a circular sensor footprint.
        """
        x0, y0 = robot_position

        # Create meshgrid of coordinates within sensor range
        x_min = max(0, int(np.floor(x0 - self.sensor_radius)))
        x_max = min(self.grid_width - 1, int(np.floor(x0 + self.sensor_radius)))
        y_min = max(0, int(np.floor(y0 - self.sensor_radius)))
        y_max = min(self.grid_height - 1, int(np.floor(y0 + self.sensor_radius)))

        xi = np.arange(x_min, x_max + 1)
        yi = np.arange(y_min, y_max + 1)
        xi_grid, yi_grid = np.meshgrid(xi, yi, indexing='xy')

        # Flatten the grids for vectorized operations
        xi_flat = xi_grid.flatten()
        yi_flat = yi_grid.flatten()

        # Compute distances from the robot position to all points
        distances = np.hypot(xi_flat - x0, yi_flat - y0)

        # Filter points within sensor radius
        within_radius = distances <= self.sensor_radius

        # Filter out the robot's own position
        not_robot = (xi_flat != x0) | (yi_flat != y0)

        # Combine filters
        valid_points = within_radius & not_robot

        # Get indices of obstacles in the grid
        obstacle_indices = np.where(self.grid[yi_flat[valid_points], xi_flat[valid_points]] == 1)[0]

        visible_obstacles = []

        for idx in obstacle_indices:
            x_obs = xi_flat[valid_points][idx]
            y_obs = yi_flat[valid_points][idx]
            obstacle_position = (x_obs, y_obs)
            if self.line_of_sight(robot_position, obstacle_position):
                visible_obstacles.append(obstacle_position)

        return visible_obstacles

    def plot_grid(self, robot_position, visible_obstacles):
        """
        Plots the grid environment, robot position, sensor range, and visible obstacles.
        """
        plt.figure(figsize=(8,8))
        # Adjust the extent to align the grid correctly
        extent = [-0.5, self.grid_width - 0.5, -0.5, self.grid_height - 0.5]
        plt.imshow(self.grid, cmap='Greys', origin='lower', extent=extent)

        x0, y0 = robot_position
        plt.plot(x0, y0, 'bo', label='Robot')

        # Draw sensor range (circle)
        circle = plt.Circle((x0, y0), self.sensor_radius, color='blue', fill=False, linestyle='--', label='Sensor Range')
        plt.gca().add_patch(circle)

        # Plot visible obstacles and lines of sight
        for obs in visible_obstacles:
            x, y = obs
            plt.plot(x, y, 'rx', markersize=12, label='Visible Obstacle' if obs == visible_obstacles[0] else "")
            # Draw line of sight
            lx = [x0, x]
            ly = [y0, y]
            plt.plot(lx, ly, 'g-', linewidth=1)

        plt.legend(loc='upper right')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(range(self.grid_width))
        plt.yticks(range(self.grid_height))
        plt.xlim(-0.5, self.grid_width - 0.5)
        plt.ylim(-0.5, self.grid_height - 0.5)
        plt.gca().set_aspect('equal')
        plt.title('Robot Sensor View with Visible Obstacles (Grid Points)')
        plt.show()

# Main script
if __name__ == "__main__":
    # Define the grid size
    grid_width = 15
    grid_height = 15
    grid = np.zeros((grid_height, grid_width), dtype=int)

    # Define the sensor radius (in grid units)
    sensor_radius = 7  # Sensor radius in units of grid cells

    # Place some obstacles (1 represents an obstacle)
    obstacles_indices = [(7, 4), (7, 5), (7, 6), (8, 7), (9, 7), (10, 7), (5, 10), (6, 10), (7, 10)]
    for xi, yi in obstacles_indices:
        grid[yi, xi] = 1  # Note that numpy arrays are indexed as [row, column] => [y, x]

    sensor = ObstacleSensor(sensor_radius, grid)

    # Obstacle positions are now at integer grid coordinates
    obstacles_positions = [(xi, yi) for xi, yi in obstacles_indices]

    # Define the robot position at a grid point
    robot_index = (7, 7)  # (x, y) indices in the grid
    robot_position = (robot_index[0], robot_index[1])

    visible_obstacles = sensor.get_visible_obstacles(robot_position)
    print("Visible obstacles from robot at position {}: {}".format(robot_position, visible_obstacles))

    sensor.plot_grid(robot_position, visible_obstacles)
