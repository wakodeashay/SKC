import numpy as np
from algorithms.hilbert_coverage import HilbertRoute
from algorithms.b_astar import BAStarRoute
from workspace.workspace import Workspace
from workspace.obstacle import Obstacle


iteration = 3
side_size = 2 ** iteration
obstacle = Obstacle(side_size, 0.2,0.8)

hilbert_workspace = Workspace('hilbert', iteration, obstacle)
boustro_workspace = Workspace('boustro', iteration, obstacle)

hilbert_route = HilbertRoute(hilbert_workspace, 2, True, False)
hilbert_route.plot()

bastar = BAStarRoute(boustro_workspace, True, False)
bastar.plot()

print(hilbert_route.points_visited)
print(bastar.points_visited)
print(len(hilbert_route.points_visited))
print(len(bastar.points_visited))
