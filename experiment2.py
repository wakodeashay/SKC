from algorithms.hilbert_coverage import HilbertRoute
from algorithms.b_astar import BAStarRoute
from workspace.workspace import Workspace
from workspace.obstacle import Obstacle
import numpy as np
import csv
import time
import gc

iteration_list = [2, 3]
# iteration_list = [4, 5]
# iteration_list = [2]
iteration_multiple = 10
blocked_fraction_range = np.linspace(0.01, 0.15, 15)
# blocked_fraction_range = np.linspace(0.01, 0.2, 2)
sparsity_range = np.linspace(0, 1.0, 21)
# sparsity_range = np.linspace(0, 1.0, 2)

with open('experiment2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["iteration", "iteration run", "sparsity", "blocked fraction", "actual_sparsity", "hilbert length",
             "bastar length", "unblocked cells"]

    for iter in iteration_list:
        t0 = time.time()
        for blocked_fraction in blocked_fraction_range:
            for i in range(iter * iteration_multiple):
                for sparsity in sparsity_range:
                    iteration = iter
                    side_size = 2 ** iteration
                    obstacle = Obstacle(side_size, blocked_fraction, sparsity)

                    hilbert_workspace = Workspace('hilbert', iteration, obstacle)
                    hilbert_route = HilbertRoute(hilbert_workspace, 1, False, False)
                    hilbert_visits = len(hilbert_route.points_visited)
                    free_grid_num = len(set(hilbert_route.points_visited))
                    actual_sparsity = obstacle.actual_sparsity

                    del hilbert_route, hilbert_workspace
                    gc.collect()

                    boustro_workspace = Workspace('boustro', iteration, obstacle)
                    bastar_route = BAStarRoute(boustro_workspace, False, False)
                    bstar_visits = len(bastar_route.points_visited)

                    del obstacle, boustro_workspace, bastar_route
                    gc.collect()

                    writer.writerow([iter, i, sparsity, blocked_fraction, actual_sparsity,
                                     hilbert_visits, bstar_visits, free_grid_num])
            gc.collect()

        t1 = time.time()
        print("Time required for iteration: ", iter, "=", t1 - t0)
