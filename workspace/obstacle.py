import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

class Obstacle:
    def __init__(self, N, blocked_fraction, sparsity):
        """
        Initializes the GridGenerator.

        Parameters:
        - N: Size of the grid (NxN).
        - blocked_fraction: Fraction of grid nodes to be blocked (0 to 1).
        - sparsity: Desired sparsity level (0 to 1).
        """
        self.N = N
        self.blocked_fraction = blocked_fraction
        self.sparsity = sparsity
        self.actual_sparsity = sparsity

        self.total_nodes = N * N
        self.B = int(round(self.total_nodes * blocked_fraction))  # Total number of blocked nodes
        self.grid = np.zeros((N, N), dtype=int)
        self.components = []

        self.grid = self.generate_grid()
        self.calculate_actual_sparsity()

    def generate_grid(self):
        """
        Generates the grid with blocked nodes according to the specified sparsity.
        """
        N = self.N
        B = self.B
        S = self.sparsity

        # Calculate the number of connected components (C)
        if B == 1:
            C = 1
        else:
            C = max(1, min(B, round(1 + (B - 1) * S)))

        self.grid = np.zeros((N, N), dtype=int)
        self.components = []

        # Create a list of all grid positions
        all_positions = [(i, j) for i in range(N) for j in range(N)]
        random.shuffle(all_positions)

        # Step 3: Select starting nodes for each component
        for _ in range(C):
            while all_positions:
                x, y = all_positions.pop()
                if self.grid[x, y] == 0:
                    self.grid[x, y] = 1  # Mark as blocked
                    self.components.append([(x, y)])  # Start new component
                    break

        # Step 4: Distribute remaining blocked nodes
        R = B - C  # Remaining blocked nodes to place
        while R > 0:
            progress_made = False
            for component in self.components:
                if R == 0:
                    break
                # Get potential neighbors
                potential_neighbors = []
                for x, y in component:
                    neighbors = self.get_valid_neighbors(x, y, component)
                    potential_neighbors.extend(neighbors)
                if potential_neighbors:
                    # Add a new node to the component
                    x_new, y_new = random.choice(potential_neighbors)
                    self.grid[x_new, y_new] = 1
                    component.append((x_new, y_new))
                    R -= 1
                    progress_made = True
                else:
                    continue  # No valid neighbors, skip to next component
            if not progress_made:
                # No progress can be made, cannot place remaining blocked nodes
                print("Cannot place all blocked nodes with the given sparsity and 8-cell adjacency.")
                break

        # Ensure the left-bottom-most element is unblocked
        self.grid[0, 0] = 0
        return self.grid

    def get_valid_neighbors(self, x, y, component):
        """
        Get unblocked neighbors of (x, y) that are not adjacent to other components.

        Parameters:
        - x, y: Current node coordinates.
        - component: The current component being expanded.

        Returns:
        - valid_neighbors: List of valid neighbor positions (x_new, y_new).
        """
        N = self.N
        grid = self.grid
        components = self.components

        # 8-connectivity moves (including diagonals)
        moves = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),          (0, 1),
                 (1, -1),  (1, 0), (1, 1)]
        valid_neighbors = []
        for dx, dy in moves:
            x_new, y_new = x + dx, y + dy
            if 0 <= x_new < N and 0 <= y_new < N:
                if grid[x_new, y_new] == 0:
                    # Check adjacency to other components
                    adjacent_to_other = False
                    for other_component in components:
                        if other_component != component:
                            for x_c, y_c in other_component:
                                # Check 8-connectivity for adjacency
                                if max(abs(x_c - x_new), abs(y_c - y_new)) <= 1:
                                    adjacent_to_other = True
                                    break
                            if adjacent_to_other:
                                break
                    if not adjacent_to_other:
                        valid_neighbors.append((x_new, y_new))
        return valid_neighbors

    def visualize_grid(self):
        """
        Visualizes the generated grid.
        """
        N = self.N
        plt.figure(figsize=(8, 8))
        plt.imshow(self.grid, cmap='Greys', origin='lower')
        plt.title(f'Blocked Nodes Grid (Sparsity {self.sparsity}, Blocked Fraction {self.blocked_fraction})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(range(0, N+1, max(1, N // 10)))
        plt.yticks(range(0, N+1, max(1, N // 10)))
        plt.show()

    def calculate_actual_sparsity(self):
        """
        Calculates and returns the actual sparsity of the generated grid.

        Returns:
        - actual_sparsity: The calculated sparsity value.
        """
        B = np.sum(self.grid)
        grid = self.grid
        # Label connected components of blocked nodes
        structure = np.ones((3, 3), dtype=int)  # 8-connectivity
        labeled_grid, num_features = label(grid, structure=structure)

        self.actual_sparsity = (num_features - 1) / (B - 1) if B > 1 else 1.0
        # print(f"Actual number of components: {num_features}")
        # print(f"Actual sparsity: {self.actual_sparsity:.2f}")
        return self.actual_sparsity

# Example usage
if __name__ == "__main__":
    # Parameters
    N = 5                # Grid size (NxN)
    blocked_fraction = 0.25 # Fraction of grid nodes to be blocked (0 to 1)
    sparsity = 0.9         # Desired sparsity level (between 0 to 1)

    # Initialize the grid generator
    grid_gen = Obstacle(N, blocked_fraction, sparsity)

    # The grid is already generated in __init__
    grid = grid_gen.grid
    print(grid)
    # Visualize the grid
    grid_gen.visualize_grid()

    # Calculate actual sparsity
    actual_sparsity = grid_gen.calculate_actual_sparsity()
