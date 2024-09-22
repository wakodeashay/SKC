import numpy as np
import matplotlib.pyplot as plt

def line_of_sight(robot_position, obstacle_position, grid):
    """
    Determines if there is a clear line of sight between the robot and the obstacle.

    Parameters:
    - robot_position: Tuple (x0, y0) representing the robot's position (at grid cell centers)
    - obstacle_position: Tuple (x1, y1) representing the obstacle's position (at grid cell centers)
    - grid: 2D numpy array representing the grid environment

    Returns:
    - True if the line of sight is clear (no obstacles blocking the view), False otherwise.
    """
    x0, y0 = robot_position
    x1, y1 = obstacle_position

    dx = x1 - x0
    dy = y1 - y0

    steps = int(max(abs(dx), abs(dy)) * 2)  # Increase steps for better sampling

    if steps == 0:
        return True  # Same cell

    x_inc = dx / steps
    y_inc = dy / steps

    x, y = x0, y0

    for _ in range(steps):
        x += x_inc
        y += y_inc

        xi = int(np.floor(x))
        yi = int(np.floor(y))

        # Skip the starting point and the obstacle cell itself
        if (xi + 0.5, yi + 0.5) == robot_position or (xi + 0.5, yi + 0.5) == obstacle_position:
            continue

        # Check grid bounds
        if xi < 0 or xi >= grid.shape[1] or yi < 0 or yi >= grid.shape[0]:
            continue

        # Check if cell contains an obstacle
        if grid[yi, xi] == 1:
            return False  # Line of sight is blocked

    return True  # Line of sight is clear

def get_visible_obstacles(grid, robot_position, sensor_radius):
    """
    Determines which obstacles are visible to the robot from its position,
    using a visual line of sight method with a circular sensor footprint.

    Parameters:
    - grid: 2D numpy array, with 0 for free space, 1 for obstacles
    - robot_position: Tuple (x0, y0) representing the robot's position (at grid cell centers)
    - sensor_radius: Integer, radius of the sensor's circular footprint in grid units

    Returns:
    - List of tuples (x, y) representing visible obstacle positions (at grid cell centers)
    """
    visible_obstacles = []
    x0, y0 = robot_position
    grid_height, grid_width = grid.shape

    # Define the bounds of the sensor range (square that contains the circle)
    x_min = max(0, int(np.floor(x0 - sensor_radius)))
    x_max = min(grid_width - 1, int(np.floor(x0 + sensor_radius)))
    y_min = max(0, int(np.floor(y0 - sensor_radius)))
    y_max = min(grid_height - 1, int(np.floor(y0 + sensor_radius)))

    # Loop over all cells within the square bounds
    for xi in range(x_min, x_max + 1):
        for yi in range(y_min, y_max + 1):
            x = xi + 0.5
            y = yi + 0.5

            # Skip the robot's own position
            if (x, y) == robot_position:
                continue

            # Calculate Euclidean distance to check if within the circular sensor range
            distance = np.hypot(x - x0, y - y0)
            if distance > sensor_radius:
                continue

            if grid[yi, xi] == 1:
                # There is an obstacle at this cell
                obstacle_position = (x, y)
                if line_of_sight(robot_position, obstacle_position, grid):
                    visible_obstacles.append(obstacle_position)
    return visible_obstacles

def plot_grid(grid, robot_position, sensor_radius, visible_obstacles):
    """
    Plots the grid environment, robot position, sensor range, and visible obstacles.
    """
    plt.figure(figsize=(8,8))
    plt.imshow(grid, cmap='Greys', origin='lower', extent=[0, grid.shape[1], 0, grid.shape[0]])

    x0, y0 = robot_position
    plt.plot(x0, y0, 'bo', label='Robot')

    # Draw sensor range (circle)
    circle = plt.Circle((x0, y0), sensor_radius, color='blue', fill=False, linestyle='--', label='Sensor Range')
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
    plt.xticks(range(grid.shape[1] + 1))
    plt.yticks(range(grid.shape[0] + 1))
    plt.xlim(0, grid.shape[1])
    plt.ylim(0, grid.shape[0])
    plt.gca().set_aspect('equal')
    plt.title('Robot Sensor View with Visible Obstacles (Centers of Grid Cells)')
    plt.show()

# Main script
if __name__ == "__main__":
    # Define the grid size
    grid_width = 15
    grid_height = 15
    grid = np.zeros((grid_height, grid_width), dtype=int)

    # Place some obstacles (1 represents an obstacle)
    obstacles_indices = [(7, 4), (7, 5), (7, 6), (8, 7), (9, 7), (10, 7), (5, 10), (6, 10), (7, 10)]
    for xi, yi in obstacles_indices:
        grid[yi, xi] = 1  # Note that numpy arrays are indexed as [row, column] => [y, x]

    # Convert obstacle indices to positions at grid cell centers
    obstacles_positions = [(xi + 0.5, yi + 0.5) for xi, yi in obstacles_indices]

    # Define the robot position at the center of a grid cell
    robot_index = (3, 3)  # (x, y) indices in the grid
    robot_position = (robot_index[0] + 0.5, robot_index[1] + 0.5)

    # Define the sensor radius (in grid units)
    sensor_radius = 7 # Sensor radius in units of grid cells

    visible_obstacles = get_visible_obstacles(grid, robot_position, sensor_radius)

    print("Visible obstacles from robot at position {}: {}".format(robot_position, visible_obstacles))

    plot_grid(grid, robot_position, sensor_radius, visible_obstacles)
