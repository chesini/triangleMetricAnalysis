import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

metric_values = np.fromfile("metric_andreashaus-1.txt", dtype=float, count=-1, sep='\n')

print("Tamanho da amostra: " + np.size(metric_values))
# metric_values *= 1.0/metric_values.max() #normalize vector into [0,1]

# metric_values = np.setdiff1d(metric_values, [0.0])

# hist, bin_edges = np.histogram(metric_values, range=(0, 1), density=True)

# filter_arr = []
# for element in hist:
#     if element < 100000:
#         filter_arr.append(True)
#     else:
#         filter_arr.append(False)

# hist = hist[filter_arr]

# print(bin_edges)

# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = plt.hist(x=metric_values, bins=1000, color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Triangle Mesh Quality Histogram')
plt.text(23, 45, r'$\mu=15, b=3$')
plt.text(50, 50, np.count_nonzero(metric_values == 1))
maxfreq = n.max() # or defined max value
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

plt.show()
