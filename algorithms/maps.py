import numpy as np
import matplotlib.pyplot as plt


def split_axis_even_odd(n):
    """Split an axis of length n into two regions, handling odd and even cases."""
    return n // 2


def average_subgrid(arr, x, y):
    """Calculate the average potential over a subgrid."""
    return (arr[2 * x, 2 * y] + arr[2 * x, 2 * y + 1] + arr[2 * x + 1, 2 * y] + arr[2 * x + 1, 2 * y + 1]) / 4


def hierarchical_tiling(arr):
    """
    Create hierarchical tiling grids by recursively dividing the grid.
    Returns a list of grids from the finest to the coarsest.
    """
    grids = [arr]
    n, m = arr.shape

    while n > 2:
        n_new = split_axis_even_odd(n)
        coarser_grid = np.zeros((n_new, n_new))

        for i in range(n_new):
            for j in range(n_new):
                coarser_grid[i, j] = average_subgrid(grids[-1], i, j)

        grids.append(coarser_grid)
        n = n_new

    return grids


def plot_3d_bars(ax, data, title):
    """
    Plots a 3D bar plot for the potential values to create a step-like appearance.
    """
    n, m = data.shape
    x, y = np.meshgrid(np.arange(n), np.arange(m))

    x = x.flatten()
    y = y.flatten()
    z = np.zeros_like(x)

    dx = dy = 1.0
    dz = data.flatten()

    ax.bar3d(x, y, z, dx, dy, dz, shade=True, color=plt.cm.viridis(dz / dz.max()))
    ax.set_title(title)


class Potential:
    def __init__(self, side_size):
        self.side_size = side_size
        self.finest_grid = np.zeros((self.side_size, self.side_size))
        self.initialize_grid()
        self.potential = hierarchical_tiling(self.finest_grid)
        # self.plot_maps()

    def initialize_grid(self):
        for i in range(self.side_size):
            for j in range(self.side_size):
                self.finest_grid[i, j] = self.side_size + 1 - i

    def update_grid(self, i, j, new_potential):
        self.finest_grid[i, j] = new_potential
        self.potential = hierarchical_tiling(self.finest_grid)
        # self.plot_maps()

    def get_potential(self, point_list):
        potential_list = []
        for i in point_list:
            potential_list.append(self.finest_grid[i // self.side_size, i % self.side_size])
        return potential_list

    def plot_maps(self):
        tiling = hierarchical_tiling(self.finest_grid)

        num_levels = len(tiling)
        fig = plt.figure(figsize=(15, 5))

        for i in range(num_levels):
            ax = fig.add_subplot(1, num_levels, i + 1, projection='3d')
            self.grid = tiling[i]
            plot_3d_bars(ax, self.grid, f'Level {i} ({self.grid.shape[0]}x{self.grid.shape[1]})')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    pot = Potential(8)
    pot.update_grid(4, 4,  -100)
    pot.update_grid(3, 5,  -100)
    pot.update_grid(3, 6,  -100)
    pot.plot_maps()
