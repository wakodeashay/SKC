import numpy as np
from algorithms.hilbert_coverage import HilbertRoute
# import time


obs = np.array([3, 4, 9, 34])
hilbert_route = HilbertRoute(2, obs, True, True)
hilbert_route.plot()
