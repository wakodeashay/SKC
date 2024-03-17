import math
import random
import numpy as np
import matplotlib.pyplot as plt
import statistics 
from main import Route
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 600
plt.rcParams["font.family"] = "Times New Roman"
csfont = {'fontname':'Times New Roman'}

def get_sparse_obstacle(start, end, size):
    """Generates sparse_obstacle with given start, end and number of obstacles required"""
    r = []
    for _ in range(size):
        r.append(random.randint(start, end))
    return r

def metric(iter, runs, max_blocked_per):
    iter = iter
    runs = runs
    size = 2**(2*iter)
    side_size = size/2
    max_blocked_per = max_blocked_per 
    max_blocked = math.floor(max_blocked_per*(2**(2*iter)))
    
    path_array = np.zeros(shape=(max_blocked + 1, runs))

    for j in range(runs): 
        for k in range(1, max_blocked + 1):
            # Exclude the first waypoint from being blocked
            obs = get_sparse_obstacle(1, size - 1, k)
            skc = Route(iter, obs, 0)
            _, _, rev, path, _ = skc.get_alt_path()
            path_array[k][j] = path

    path_mean = [sum(path_array[i])/len(path_array[i])/side_size for i in range(1, max_blocked + 1)]
    path_mean.insert(0, (size-1)/side_size)
    path_err = [statistics.pstdev(path_array[i])/side_size for i in range(1, max_blocked + 1)]
    path_err.insert(0, 0.0)

    str_label = 'Iteration : ' + str(iter) 
    plt.errorbar(range(0, max_blocked+1), path_mean, linestyle='dashed', linewidth=2, marker='o', markersize=15, yerr=path_err, label=str_label)
        
metric(2, 100, 0.5)

plt.grid(linestyle='dotted')
plt.tick_params(axis='x', labelsize=25, pad=10)
plt.tick_params(axis='y', labelsize=25, pad=10)

# naming the x axis
plt.xlabel('Number of Waypoints Blocked', fontsize=30, labelpad=30, **csfont)
# naming the y axis
plt.ylabel('Path Length', fontsize=30, labelpad=30, **csfont)
plt.legend(fontsize = 30, borderpad=1.0, labelspacing=1.5)
plt.show()
