import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()
print("Open metric file: " + file_path)

x = np.fromfile(file_path, dtype=float, count=-1, sep='\n')
# x *= 1.0/x.max() #normalize vector into [0,1]

mu = np.mean(x)  # mean of distribution
print("mean: " + str(mu))

sigma = np.std(x)  # standard deviation of distribution
print("sigma: " + str(sigma))

fig, ax = plt.subplots()

num_bins = 1024
# the histogram of the data
n, bins, patches = ax.hist(x, num_bins, density=True)

print("len(bins): " + str(len(bins)))
print("len(n): " + str(len(n)))

# bins = np.delete(bins, int(len(bins) - 1))
# n = np.delete(n, str(len(n) - 1))

# print("len(bins): " + str(len(bins)))
# print("len(n): " + str(len(n)))

xaxis = np.zeros(bins.shape)
for idx, val in enumerate(bins):
    if(idx + 1 < len(bins)):
        xaxis[idx+1] = np.mean([bins[idx], bins[idx+1]])
    else:
        xaxis[idx] = np.mean([bins[idx-1], bins[idx]])

print(xaxis)

n = np.append([0], n)
ax.plot(xaxis, n, '-p')


px = n / sum(n)
print("px: ")
print(px)
ax.plot(xaxis, px, '-ok')


print("Bins: ")
print(xaxis)
print("n: ")
print(n)

# add an 'exponential' line
y = n * np.exp(np.e, (-n * xaxis)) #lambda * e ^ (-lambda*x)
     
ax.plot(xaxis, y, '--')
ax.set_xlabel('Radius-ratio value')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of IQ: $\mu=' +  str(mu) + r'$, $\sigma=' + str(sigma) + r'$')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()
