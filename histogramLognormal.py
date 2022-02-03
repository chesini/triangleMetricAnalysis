from os import major
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
from scipy.stats import lognorm

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(title='Selecione o primeiro arquivo')
print("Open metric file: " + file_path)

x = np.fromfile(file_path, dtype=float, count=-1, sep='\n')
# x *= 1.0/x.max() #normalize vector into [0,1]

# remove the major metric value (1000)
# while True:
#     majorIdx = np.where(x == x.max())
#     print("x.max(): " + str(x.max()))
#     np.delete(x, majorIdx[0], 0)
#     print("len(x)" + str(len(x)))
#     if x.max() < 1:
#         print("x.max() < 1")
#         break



mu = np.mean(x)  # mean of distribution
print("mean: " + str(mu))

sigma = np.std(x)  # standard deviation of distribution
print("sigma: " + str(sigma))

# dist=lognorm([sigma],loc=mu)

fig, ax = plt.subplots()

num_bins = 1024
# the histogram of the data
n, bins, patches = ax.hist(x, num_bins, density=True)
n = np.append([0], n)

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

# print(xaxis)

# ax.plot(xaxis, n, '-p')


px = n / sum(n)
# print("px: ")
# print(px)
# ax.plot(xaxis, px, '-ok')


# add a 'lognormal' line
ax.plot(xaxis, lognorm.pdf(xaxis, sigma)*100, '--')
ax.set_xlabel('Radius-ratio value')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of IQ: $\mu=' +  str(mu) + r'$, $\sigma=' + str(sigma) + r'$')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()
