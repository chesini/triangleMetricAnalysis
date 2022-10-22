import json
import sys
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
from scipy.stats import lognorm, shapiro, jarque_bera, anderson

import tkinter as tk
from tkinter import filedialog

from dataNormalizer import getMaxNormalized, getMirrored, getNormNormalized

root = tk.Tk()
root.withdraw()

dot_style = ['-', '--', '-.', ':']
color = ['k', 'y', 'r', 'c']

file_path = filedialog.askopenfilenames(title='Selecione os arquivos de métrica')
print(file_path)

# fig3, cx = plt.subplots()
# fig4, dx = plt.subplots()

def getQualityPlot(ax, file_path, file_n):
    x = np.fromfile(file_path, dtype=float, count=-1, sep='\n')
    x_infos = {}

    x_infos.update({
        'invalid_count': len( x[x == (-1)] )
    })
    x = x[x!=(-1)]

    x_infos.update({
        'outlier_count': len( x[x >= 1000] )
    })
    # x = x[x < 1000]
    # x_log = np.log10(x)
    # x = np.log(x)
    # x = np.sqrt(x)
    # x = np.cbrt(x)
    # x = x + np.abs(np.min(x))
    # x -= 1
    # x = getMaxNormalized(x)

    x_infos.update({
        'len': len(x),
        'min': min(x),
        'max': max(x),
        'sum': np.sum(x),
        'mean': np.mean(x),  # mean of distribution
        'std': np.std(x),  # standard deviation of distribution
        'var': np.var(x)  # variance of distribution
    })




    # boxes =  cx.boxplot(x_log, showmeans=True)
    # for key in boxes:
    #     if key != 'fliers' and key != 'whiskers':
    #         print(f'{key}: {[item.get_ydata() for item in boxes[key]]}\n')

    # upper_outlier = boxes['caps'][1].get_ydata()[0]
    # outliers = boxes['fliers'][0].get_ydata()
    # whiskers = [boxes['whiskers'][0].get_ydata()[1], boxes['whiskers'][1].get_ydata()[1]]
    # print('whiskers: ', whiskers)

    # filter_arr = []
    # for element in x_log:
    #     # if the element is higher than 42, set the value to True, otherwise False:
    #     if element >= whiskers[0] and element <= whiskers[1]:
    #         filter_arr.append(True)
    #     else:
    #         filter_arr.append(False)

    # x_log = x_log[filter_arr]
    
    # print('len(outliers): {}'.format(len(outliers)))
    # # print(boxes['whiskers'][0].get_ydata())
    # x_log = x_log[np.isin(x_log[:], outliers)]

    # boxes =  dx.boxplot(x, showmeans=True)


    # x_log_infos = {
    #     'len': len(x_log),
    #     'min': min(x_log),
    #     'max': max(x_log),
    #     'sum': np.sum(x_log),
    #     'mean': np.mean(x_log),  # mean of distribution
    #     'std': np.std(x_log),  # standard deviation of distribution
    #     'var': np.var(x_log)  # variance of distribution
    # }



    # remove the major metric value (1000)
    # while True:
    #     majorIdx = np.where(x == x.max())
    #     print("x.max(): " + str(x.max()))
    #     np.delete(x, majorIdx[0], 0)
    #     print("len(x)" + str(len(x)))
    #     if x.max() < 1:
    #         print("x.max() < 1")
    #         break



    # x_lognorm = lognorm(s=x_infos.get('std'), loc=0, scale=x_infos.get('mean'))
    # var = x_lognorm.var()
    # what = x_lognorm.expect()
    # print('x_lognorm.expect(', type(what), '): ', what)
    # print('x_lognorm.variance: ', var)

    # mean, var, skew, kurt = x_lognorm.stats(moments='mvsk')
    # print('x_lognorm.mean: ', mean)
    # print('x_lognorm.var: ', var)
    # print('x_lognorm.skew: ', skew)
    # print('x_lognorm.kurt: ', kurt)


    # x_log_lognorm = lognorm(s=x_log_infos.get('std'), loc=0, scale=x_log_infos.get('mean'))
    # log_var = x_log_lognorm.var()
    # log_what = x_log_lognorm.expect()
    # print('x_log_lognorm.expect(', type(log_what), '): ', log_what)
    # print('x_log_lognorm.variance: ', log_var)

    # log_mean, log_var, log_skew, log_kurt = x_log_lognorm.stats(moments='mvsk')
    # print('x_log_lognorm.mean: ', log_mean)
    # print('x_log_lognorm.var: ', log_var)
    # print('x_log_lognorm.skew: ', log_skew)
    # print('x_log_lognorm.kurt: ', log_kurt)

    # fitted_shape, fitted_location, fitted_scale = lognorm.fit(x)
    # # print("fitted_shape: {}, fitted_location: {}, fitted_scale: {}".format(fitted_shape, fitted_location, fitted_scale))
    # x_infos.update({
    #     'fitted_std': fitted_shape,
    #     'fitted_location': fitted_location,
    #     'fitted_mean': fitted_scale
    # })


    # stat, critical_values, significance_level = anderson(x_log, dist='norm')
    # print('Statistics={}, critical_values={}, significance_level={}'.format(stat, critical_values, significance_level))
    



    _label = file_path.split("/").pop()
    label = '{} {}'.format(_label, '')
    num_bins = 16


    # fig, ax = plt.subplots(1, 2)
    color_index = file_n % len(color)
    print('color_index: {} = {} % {}'.format(color_index, file_n, len(color)))


    dot_index = file_n % len(dot_style)
    print('dot_index: {} = {} % {}'.format(dot_index, file_n, len(dot_style)))

    # the histogram of the data
    # ax[0].set_title('Histogram (mu: {}; sigma: {}'.format(mu, sigma))
    ax.set_title(f'Histograma com {num_bins} divisões')
    ax.set_xlabel(r'Valor de $\rho$/2')
    # ax.set_ylabel('Probability density')
    # n, bins, patches = ax.hist(x, num_bins, density=True, label=label, histtype='step', color=color[color_index])
    ax.set_ylabel('Ocorrências do valor')
    n, bins, patches = ax.hist(x, num_bins, density=False, label=label, histtype='bar', color=color[color_index])
    ax.set_yscale('log')
    ax.grid(True, which='both')
    print(np.arange(0, 1100, 100))
    ax.set_xticks(np.arange(0, 1100, 100))

    # # the histogram of the logarithm of data
    # # ax[1].set_title('Histogram (mu: {}; sigma: {}'.format(mu, sigma))
    # ax[1].set_title('Histograma com 1024 divisões')
    # ax[1].set_xlabel(r'Valor de log($\rho$/2)')
    # ax[1].set_ylabel('Probability density')
    # n_log, bins_log, patches = ax[1].hist(x_log, num_bins, density=True, label=label, histtype='step')
    # # ax[1].set_ylabel('Ocorrências do valor')
    # # n_log, bins_log, patches = ax[1].hist(x_log, num_bins, density=False, label=label, histtype='step', color='k')
    # ax[1].grid(True, which='both')


    # n = np.append([0], n)

    print("len(bins): " + str(len(bins)))
    print("len(n): " + str(len(n)))

    # bins = np.delete(bins, int(len(bins) - 1))
    # n = np.delete(n, str(len(n) - 1))

    # print("len(bins): " + str(len(bins)))
    # print("len(n): " + str(len(n)))


    stat, p = shapiro(n)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret results
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')



    xaxis = np.zeros(bins.shape)
    for idx, val in enumerate(bins):
        if(idx + 1 < len(bins)):
            xaxis[idx+1] = np.mean([bins[idx], bins[idx+1]])
        else:
            xaxis[idx] = np.mean([bins[idx-1], bins[idx]])


    # x_log_axis = np.zeros(bins_log.shape)
    # for idx, val in enumerate(bins_log):
    #     if(idx + 1 < len(bins_log)):
    #         x_log_axis[idx+1] = np.mean([bins_log[idx], bins_log[idx+1]])
    #     else:
    #         x_log_axis[idx] = np.mean([bins_log[idx-1], bins_log[idx]])

    # print(xaxis)

    ax.plot(xaxis, np.append(n[0], n), dot_style[dot_index])


    print('sum(n[:]): {}'.format(label), sum(n[:]))

    px = n / sum(n)
    # print("px: ")
    # print(px)
    # ax.plot(xaxis, px, '-ok')


    # fig2, bx = plt.subplots()

    # # add a 'lognormal' line
    # bx.set_title('Lognormal PDF (mu: {}; sigma: {}'.format(x_infos.get('mean'), x_infos.get('std')))
    # bx.set_xlabel('Radius-ratio value')
    # bx.set_ylabel('Probability density')


    # ax[0].plot(xaxis, lognorm.pdf(xaxis, x_infos.get('std')), dot_style[dot_index], label='scipy.stats.lognorm.pdf_{}'.format(label))
    # ax.plot(bins, x_lognorm.pdf(bins), dot_style[dot_index], label='scipy.stats.lognorm_{}'.format(label))
    
    # # ax[1].plot(x_log_axis, lognorm.pdf(x_log_axis, x_log_infos.get('std')), dot_style[dot_index], label='scipy.stats.lognorm.pdf_{}'.format(label))
    # ax[1].plot(bins_log, x_log_lognorm.pdf(bins_log), dot_style[dot_index], label='scipy.stats.lognorm_{}'.format(label))
    
    # ax.plot(xaxis, lognorm(s=sigma, loc=fitted_location, scale=fitted_scale).pdf(xaxis), dot_style[dot_index], label='scipy.stats.lognorm_{}'.format(label))
    # ax.plot(xaxis, norm.pdf(xaxis, sigma)*1000, dot_style[dot_index], label=label)

    # sum of PDF, it would be 1
    # pdf_lognorm = x_lognorm.pdf(bins)
    # print('sum(pdf_lognorm[:]): {}'.format(label), sum(pdf_lognorm[:]))
    # print(pdf_lognorm)

    ax.legend(loc="upper right")
    # ax[1].legend(loc="upper right")
    # bx.legend(loc="upper right")

    print('x_infos: ', json.dumps(x_infos, indent=2))
    # print('x_log_infos: ', json.dumps(x_log_infos, indent=2))

    return n, x_infos.get('mean'), x_infos.get('std'), bins



def getQualityPlot2(ax, file_path, file_n):
    x = np.fromfile(file_path, dtype=float, count=-1, sep='\n')
    x_infos = {}

    plots = [[0,0], [0,1], [1,0], [1,1]]

    x_infos.update({
        'invalid_count': len( x[x == (-1)] )
    })
    x = x[x!=(-1)]

    x_infos.update({
        'outlier_count': len( x[x >= 1000] )
    })
    # x = x[x < 1000]
    # x_log = np.log10(x)
    # x = np.log(x)
    # x = np.sqrt(x)
    # x = np.cbrt(x)
    # x = x + np.abs(np.min(x))
    # x -= 1
    # x = getMaxNormalized(x)

    x_infos.update({
        'len': len(x),
        'min': min(x),
        'max': max(x),
        'sum': np.sum(x),
        'mean': np.mean(x),  # mean of distribution
        'std': np.std(x),  # standard deviation of distribution
        'var': np.var(x)  # variance of distribution
    })

    _label = file_path.split("/").pop()
    label = '{} {}'.format(_label, '')
    num_bins = 1024


    # fig, ax = plt.subplots(1, 2)
    # color_index = file_n % len(color)
    color_index = 0
    print('color_index: {} = {} % {}'.format(color_index, file_n, len(color)))


    dot_index = file_n % len(dot_style)
    print('dot_index: {} = {} % {}'.format(dot_index, file_n, len(dot_style)))

    # the histogram of the data
    # ax[0].set_title('Histogram (mu: {}; sigma: {}'.format(mu, sigma))
    ax[plots[file_n][0]][plots[file_n][1]].set_title(f'Histograma com {num_bins} divisões')
    ax[plots[file_n][0]][plots[file_n][1]].set_xlabel(r'Valor de $\rho$/2')
    # ax.set_ylabel('Probability density')
    # n, bins, patches = ax.hist(x, num_bins, density=True, label=label, histtype='step', color=color[color_index])
    ax[plots[file_n][0]][plots[file_n][1]].set_ylabel('Ocorrências do valor')
    n, bins, patches = ax[plots[file_n][0]][plots[file_n][1]].hist(x, num_bins, density=False, label=label, histtype='step', color=color[color_index])
    ax[plots[file_n][0]][plots[file_n][1]].set_yscale('log')
    ax[plots[file_n][0]][plots[file_n][1]].set_xscale('log')
    ax[plots[file_n][0]][plots[file_n][1]].grid(True, which='both')
    # print(np.arange(0, 1100, 100))
    max_tick_x = np.ceil(np.log10(np.max(bins)))
    print('max_tick_x: ', max_tick_x)
    x_ticks = np.logspace(0, max_tick_x, int(max_tick_x + 1))
    print('x_ticks: ', x_ticks)
    
    max_tick_y = np.ceil(np.log10(np.max(n)))
    print('max_tick_y: ', max_tick_y)
    y_ticks = np.logspace(0, max_tick_y, int(max_tick_y + 1))
    print('y_ticks: ', y_ticks)
    # ax[plots[file_n][0]][plots[file_n][1]].set_xticks(np.arange(0, 1100, 100))
    ax[plots[file_n][0]][plots[file_n][1]].set_xticks(x_ticks)
    ax[plots[file_n][0]][plots[file_n][1]].set_yticks(y_ticks)


    # n = np.append([0], n)

    print("len(bins): " + str(len(bins)))
    print("len(n): " + str(len(n)))

    xaxis = np.zeros(bins.shape)
    for idx, val in enumerate(bins):
        if(idx + 1 < len(bins)):
            xaxis[idx+1] = np.mean([bins[idx], bins[idx+1]])
        else:
            xaxis[idx] = np.mean([bins[idx-1], bins[idx]])

    # ax[plots[file_n][0]][plots[file_n][1]].plot(xaxis, np.append(n[0], n), dot_style[dot_index])


    ax[plots[file_n][0]][plots[file_n][1]].legend(loc="upper right")

    print('x_infos: ', json.dumps(x_infos, indent=2))

    return n, x_infos.get('mean'), x_infos.get('std'), bins



def getQualityPlotUnified(x, ax, file_path, file_label, file_n):
    x[file_n] = np.fromfile(file_path[file_n], dtype=float, count=-1, sep='\n')
    x_infos = {}

    x_infos.update({
        'invalid_count': len( x[file_n][x[file_n] == (-1)] )
    })
    x[file_n] = x[file_n][x[file_n]!=(-1)]
    x[file_n] = x[file_n][x[file_n] < 1000]

    x_infos.update({
        'outlier_count': len( x[file_n][x[file_n] >= 1000] )
    })

    x_infos.update({
        'len': len(x[file_n]),
        'min': min(x[file_n]),
        'max': max(x[file_n]),
        'sum': np.sum(x[file_n]),
        'mean': np.mean(x[file_n]),  # mean of distribution
        'std': np.std(x[file_n]),  # standard deviation of distribution
        'var': np.var(x[file_n])  # variance of distribution
    })


    _label = file_path[file_n].split("/").pop()
    file_label[file_n] = '{} {}'.format(_label, '')
    num_bins = 1024


    # fig, ax = plt.subplots(1, 2)
    color_index = file_n % len(color)
    print('color_index: {} = {} % {}'.format(color_index, file_n, len(color)))


    dot_index = file_n % len(dot_style)
    print('dot_index: {} = {} % {}'.format(dot_index, file_n, len(dot_style)))

    # the histogram of the data
    # ax[0].set_title('Histogram (mu: {}; sigma: {}'.format(mu, sigma))
    ax.set_title(f'Histograma com {num_bins} divisões')
    ax.set_xlabel(r'Valor de $\rho$/2')
    # ax.set_ylabel('Probability density')
    # n, bins, patches = ax.hist(x, num_bins, density=True, label=label, histtype='step', color=color[color_index])
    ax.set_ylabel('Ocorrências do valor')
    if file_n == 3:
        n, bins, patches = ax.hist(x, num_bins, density=False, label=file_label, histtype='bar', color=color, align='left')
        # print('n.shape: ', n.shape, 'bins.shape: ', bins.shape)
        xaxis = np.zeros(bins.shape)
        for idx, val in enumerate(bins):
            if(idx + 1 < len(bins)):
                xaxis[idx+1] = np.mean([bins[idx], bins[idx+1]])
            else:
                xaxis[idx] = np.mean([bins[idx-1], bins[idx]])

        fig2, bx = plt.subplots()
        bx.set_yscale('log')
        bx.set_xscale('log')
        for idx, val in enumerate(n):
            bx.set_title(f'Valores do histograma')
            bx.set_xlabel(r'Valor de $\rho$/2')
            bx.set_ylabel('Ocorrências do valor')

            max_tick_x = np.ceil(np.log10(np.max(bins)))
            print('max_tick_x: ', max_tick_x)
            x_ticks = np.logspace(0, max_tick_x, int(max_tick_x + 1))
            print('x_ticks: ', x_ticks)
            
            # max_tick_y = np.ceil(np.log10(np.max(n)))
            # print('max_tick_y: ', max_tick_y)
            # y_ticks = np.logspace(0, max_tick_y, int(max_tick_y + 1))
            # print('y_ticks: ', y_ticks)
            # bx.set_xticks(np.arange(0, 1100, 100))
            bx.set_xticks(x_ticks)
            # bx.set_yticks(y_ticks)

            bx.plot(bins[0:-1], n[idx], dot_style[idx], label=file_label[idx], color=color[idx])
            bx.legend(loc="upper right")

            print('{}*{} + {}*{} + ... + {}*{}'.format(n[idx][0], bins[0], n[idx][1], bins[1], n[idx][-1], bins[0:-1][-1]))

            w_avg = np.average(n[idx], weights=bins[0:-1])
            print('WEIGHTED_AVG-{}: '.format(file_label[idx]), w_avg)
            # print(str(w_avg).replace('.', ','))

        
    else:
        n = np.array([])
        bins = np.array([])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid(True, which='both')
    # print(np.arange(0, 1100, 100))
    # ax.set_xticks(np.arange(0, 1100, 100))

    # # the histogram of the logarithm of data
    # # ax[1].set_title('Histogram (mu: {}; sigma: {}'.format(mu, sigma))
    # ax[1].set_title('Histograma com 1024 divisões')
    # ax[1].set_xlabel(r'Valor de log($\rho$/2)')
    # ax[1].set_ylabel('Probability density')
    # n_log, bins_log, patches = ax[1].hist(x_log, num_bins, density=True, label=label, histtype='step')
    # # ax[1].set_ylabel('Ocorrências do valor')
    # # n_log, bins_log, patches = ax[1].hist(x_log, num_bins, density=False, label=label, histtype='step', color='k')
    # ax[1].grid(True, which='both')


    # n = np.append([0], n)

    # print("len(bins): " + str(len(bins)))
    # print("len(n): " + str(len(n)))

    # bins = np.delete(bins, int(len(bins) - 1))
    # n = np.delete(n, str(len(n) - 1))

    # print("len(bins): " + str(len(bins)))
    # print("len(n): " + str(len(n)))


    # stat, p = shapiro(n)
    # print('Statistics=%.3f, p=%.3f' % (stat, p))
    # # interpret results
    # alpha = 0.05
    # if p > alpha:
    #     print('Sample looks Gaussian (fail to reject H0)')
    # else:
    #     print('Sample does not look Gaussian (reject H0)')





    # print('sum(n[:]): {}'.format(label), sum(n[:]))

    # px = n / sum(n)
    # print("px: ")
    # print(px)
    # ax.plot(xaxis, px, '-ok')


    # fig2, bx = plt.subplots()

    # # add a 'lognormal' line
    # bx.set_title('Lognormal PDF (mu: {}; sigma: {}'.format(x_infos.get('mean'), x_infos.get('std')))
    # bx.set_xlabel('Radius-ratio value')
    # bx.set_ylabel('Probability density')


    # ax[0].plot(xaxis, lognorm.pdf(xaxis, x_infos.get('std')), dot_style[dot_index], label='scipy.stats.lognorm.pdf_{}'.format(label))
    # ax.plot(bins, x_lognorm.pdf(bins), dot_style[dot_index], label='scipy.stats.lognorm_{}'.format(label))
    
    # # ax[1].plot(x_log_axis, lognorm.pdf(x_log_axis, x_log_infos.get('std')), dot_style[dot_index], label='scipy.stats.lognorm.pdf_{}'.format(label))
    # ax[1].plot(bins_log, x_log_lognorm.pdf(bins_log), dot_style[dot_index], label='scipy.stats.lognorm_{}'.format(label))
    
    # ax.plot(xaxis, lognorm(s=sigma, loc=fitted_location, scale=fitted_scale).pdf(xaxis), dot_style[dot_index], label='scipy.stats.lognorm_{}'.format(label))
    # ax.plot(xaxis, norm.pdf(xaxis, sigma)*1000, dot_style[dot_index], label=label)

    # sum of PDF, it would be 1
    # pdf_lognorm = x_lognorm.pdf(bins)
    # print('sum(pdf_lognorm[:]): {}'.format(label), sum(pdf_lognorm[:]))
    # print(pdf_lognorm)

    ax.legend(loc="upper right")
    # ax[1].legend(loc="upper right")

    print('x_infos: ', json.dumps(x_infos, indent=2))
    # print('x_log_infos: ', json.dumps(x_log_infos, indent=2))

    return n, x_infos.get('mean'), x_infos.get('std'), bins



def getQualityLogPlot(ax, file_path, file_n):
    x = np.fromfile(file_path, dtype=float, count=-1, sep='\n')
    x_infos = {}

    x_infos.update({
        'invalid_count': len( x[x == (-1)] )
    })
    x = x[x!=(-1)]

    x_infos.update({
        'outlier_count': len( x[x >= 1000] )
    })
    x = x[x < 1000]
    x_log = np.log10(x)
    # x = np.log(x)
    # x = np.sqrt(x)
    # x = np.cbrt(x)
    # x = x + np.abs(np.min(x))
    # x -= 1
    # x = getMaxNormalized(x)

    x_infos.update({
        'len': len(x),
        'min': min(x),
        'max': max(x),
        'sum': np.sum(x),
        'mean': np.mean(x),  # mean of distribution
        'std': np.std(x),  # standard deviation of distribution
        'var': np.var(x)  # variance of distribution
    })




    # boxes =  cx.boxplot(x_log, showmeans=True)
    # for key in boxes:
    #     if key != 'fliers' and key != 'whiskers':
    #         print(f'{key}: {[item.get_ydata() for item in boxes[key]]}\n')

    # upper_outlier = boxes['caps'][1].get_ydata()[0]
    # outliers = boxes['fliers'][0].get_ydata()
    # whiskers = [boxes['whiskers'][0].get_ydata()[1], boxes['whiskers'][1].get_ydata()[1]]
    # print('whiskers: ', whiskers)

    # filter_arr = []
    # for element in x_log:
    #     # if the element is higher than 42, set the value to True, otherwise False:
    #     if element >= whiskers[0] and element <= whiskers[1]:
    #         filter_arr.append(True)
    #     else:
    #         filter_arr.append(False)

    # x_log = x_log[filter_arr]
    
    # print('len(outliers): {}'.format(len(outliers)))
    # # print(boxes['whiskers'][0].get_ydata())
    # x_log = x_log[np.isin(x_log[:], outliers)]

    # boxes =  dx.boxplot(x, showmeans=True)


    x_log_infos = {
        'len': len(x_log),
        'min': min(x_log),
        'max': max(x_log),
        'sum': np.sum(x_log),
        'mean': np.mean(x_log),  # mean of distribution
        'std': np.std(x_log),  # standard deviation of distribution
        'var': np.var(x_log)  # variance of distribution
    }



    # remove the major metric value (1000)
    # while True:
    #     majorIdx = np.where(x == x.max())
    #     print("x.max(): " + str(x.max()))
    #     np.delete(x, majorIdx[0], 0)
    #     print("len(x)" + str(len(x)))
    #     if x.max() < 1:
    #         print("x.max() < 1")
    #         break



    x_lognorm = lognorm(s=x_infos.get('std'), loc=0, scale=x_infos.get('mean'))
    # var = x_lognorm.var()
    # what = x_lognorm.expect()
    # print('x_lognorm.expect(', type(what), '): ', what)
    # print('x_lognorm.variance: ', var)

    mean, var, skew, kurt = x_lognorm.stats(moments='mvsk')
    print('x_lognorm.mean: ', mean)
    print('x_lognorm.var: ', var)
    print('x_lognorm.skew: ', skew)
    print('x_lognorm.kurt: ', kurt)


    x_log_lognorm = lognorm(s=x_log_infos.get('std'), loc=0, scale=x_log_infos.get('mean'))
    # log_var = x_log_lognorm.var()
    # log_what = x_log_lognorm.expect()
    # print('x_log_lognorm.expect(', type(log_what), '): ', log_what)
    # print('x_log_lognorm.variance: ', log_var)

    log_mean, log_var, log_skew, log_kurt = x_log_lognorm.stats(moments='mvsk')
    print('x_log_lognorm.mean: ', log_mean)
    print('x_log_lognorm.var: ', log_var)
    print('x_log_lognorm.skew: ', log_skew)
    print('x_log_lognorm.kurt: ', log_kurt)

    fitted_shape, fitted_location, fitted_scale = lognorm.fit(x)
    # print("fitted_shape: {}, fitted_location: {}, fitted_scale: {}".format(fitted_shape, fitted_location, fitted_scale))
    x_infos.update({
        'fitted_std': fitted_shape,
        'fitted_location': fitted_location,
        'fitted_mean': fitted_scale
    })


    # stat, critical_values, significance_level = anderson(x_log, dist='norm')
    # print('Statistics={}, critical_values={}, significance_level={}'.format(stat, critical_values, significance_level))
    



    _label = file_path.split("/").pop()
    label = '{} {}'.format(_label, '')
    num_bins = 1024


    # the histogram of the data
    # ax[0].set_title('Histogram (mu: {}; sigma: {}'.format(mu, sigma))
    ax[0].set_title('Histograma com 1024 divisões')
    ax[0].set_xlabel(r'Valor de $\rho$/2')
    ax[0].set_ylabel('Probability density')
    n, bins, patches = ax[0].hist(x, num_bins, density=True, label=label, histtype='step')
    # ax[0].set_ylabel('Ocorrências do valor')
    # n, bins, patches = ax[0].hist(x, num_bins, density=False, label=label, histtype='step', color='k')
    ax[0].set_yscale('log')
    ax[0].grid(True, which='both')

    # the histogram of the logarithm of data
    # ax[1].set_title('Histogram (mu: {}; sigma: {}'.format(mu, sigma))
    ax[1].set_title('Histograma com 1024 divisões')
    ax[1].set_xlabel(r'Valor de log($\rho$/2)')
    ax[1].set_ylabel('Probability density')
    n_log, bins_log, patches = ax[1].hist(x_log, num_bins, density=True, label=label, histtype='step')
    # ax[1].set_ylabel('Ocorrências do valor')
    # n_log, bins_log, patches = ax[1].hist(x_log, num_bins, density=False, label=label, histtype='step', color='k')
    ax[1].grid(True, which='both')


    n = np.append([0], n)
    n_log = np.append([0], n_log)

    print("len(bins): " + str(len(bins)))
    print("len(n): " + str(len(n)))

    # bins = np.delete(bins, int(len(bins) - 1))
    # n = np.delete(n, str(len(n) - 1))

    # print("len(bins): " + str(len(bins)))
    # print("len(n): " + str(len(n)))


    stat, p = shapiro(n_log)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret results
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')



    xaxis = np.zeros(bins.shape)
    for idx, val in enumerate(bins):
        if(idx + 1 < len(bins)):
            xaxis[idx+1] = np.mean([bins[idx], bins[idx+1]])
        else:
            xaxis[idx] = np.mean([bins[idx-1], bins[idx]])


    x_log_axis = np.zeros(bins_log.shape)
    for idx, val in enumerate(bins_log):
        if(idx + 1 < len(bins_log)):
            x_log_axis[idx+1] = np.mean([bins_log[idx], bins_log[idx+1]])
        else:
            x_log_axis[idx] = np.mean([bins_log[idx-1], bins_log[idx]])

    # print(xaxis)

    # ax.plot(xaxis, n, '-p')


    print('sum(n[:]): {}'.format(label), sum(n[:]))

    px = n / sum(n)
    # print("px: ")
    # print(px)
    # ax.plot(xaxis, px, '-ok')


    # fig2, bx = plt.subplots()

    # # add a 'lognormal' line
    # bx.set_title('Lognormal PDF (mu: {}; sigma: {}'.format(x_infos.get('mean'), x_infos.get('std')))
    # bx.set_xlabel('Radius-ratio value')
    # bx.set_ylabel('Probability density')


    dot_index = file_n % len(dot_style)
    print('dot_index: {} = {} % {}'.format(dot_index, file_n, len(dot_style)))
    ax[0].plot(xaxis, lognorm.pdf(xaxis, x_infos.get('std')), dot_style[dot_index], label='scipy.stats.lognorm.pdf_{}'.format(label))
    # ax[0].plot(bins, x_lognorm.pdf(bins), dot_style[dot_index], label='scipy.stats.lognorm_{}'.format(label))
    
    # ax[1].plot(x_log_axis, lognorm.pdf(x_log_axis, x_log_infos.get('std')), dot_style[dot_index], label='scipy.stats.lognorm.pdf_{}'.format(label))
    ax[1].plot(bins_log, x_log_lognorm.pdf(bins_log), dot_style[dot_index], label='scipy.stats.lognorm_{}'.format(label))
    
    # ax.plot(xaxis, lognorm(s=sigma, loc=fitted_location, scale=fitted_scale).pdf(xaxis), dot_style[dot_index], label='scipy.stats.lognorm_{}'.format(label))
    # ax.plot(xaxis, norm.pdf(xaxis, sigma)*1000, dot_style[dot_index], label=label)

    # sum of PDF, it would be 1
    pdf_lognorm = x_lognorm.pdf(bins)
    print('sum(pdf_lognorm[:]): {}'.format(label), sum(pdf_lognorm[:]))
    # print(pdf_lognorm)

    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper right")
    # bx.legend(loc="upper right")

    print('x_infos: ', json.dumps(x_infos, indent=2))
    print('x_log_infos: ', json.dumps(x_log_infos, indent=2))

    return n, x_infos.get('mean'), x_infos.get('std'), bins



def getQualityHistograms(file_path, file_n):
    x = np.fromfile(file_path, dtype=float, count=-1, sep='\n')
    x = x[x!=(-1)]
    # x = x[x < 1000]
    # x = np.log(x)

    x_infos = {
        'len': len(x),
        'min': min(x),
        'max': max(x),
        'sum': np.sum(x),
        'mean': np.mean(x),  # mean of distribution
        'std': np.std(x),  # standard deviation of distribution
        'var': np.var(x)  # variance of distribution
    }

    fig, ax = plt.subplots(2, 2)


    _label = file_path.split("/").pop()
    label = '{} {}'.format(_label, '(com outlier 1000)')
    num_bins = 1024

    # the histogram of the data
    ax[0, 0].set_title('Histograma a)')
    ax[0, 0].set_xlabel(r'Valor de $\rho$/2')
    # ax[0, 0].set_ylabel('Probability density')
    # n, bins, patches = ax[0, 0].hist(x, num_bins, density=True, label=label, histtype='step')
    ax[0, 0].set_ylabel('Ocorrências do valor')
    n, bins, patches = ax[0, 0].hist(x, num_bins, density=False, label=label, histtype='step', color='k')
    ax[0, 0].grid(True, which='both')
    ax[0, 0].legend(loc="upper right")

    # n = np.append([0], n)

    print("len(bins): " + str(len(bins)))
    print("len(n): " + str(len(n)))

    xaxis = np.zeros(bins.shape)
    for idx, val in enumerate(bins):
        if(idx + 1 < len(bins)):
            xaxis[idx+1] = np.mean([bins[idx], bins[idx+1]])
        else:
            xaxis[idx] = np.mean([bins[idx-1], bins[idx]])

    # print(xaxis)

    # ax[0, 0].plot(xaxis, n, '-p')



    label = '{} {}'.format(_label, '(com outlier 1000)')
    # the histogram of the data
    ax[0, 1].set_title('Histograma b)')
    ax[0, 1].set_xlabel(r'Valor de $\rho$/2')
    # ax.set_ylabel('Probability density')
    # n, bins, patches = ax.hist(x, num_bins, density=True, label=label, histtype='step')
    ax[0, 1].set_ylabel('Ocorrências do valor (log)')
    n, bins, patches = ax[0, 1].hist(x, num_bins, density=False, label=label, histtype='step', color='k')
    ax[0, 1].set_yscale('log')
    ax[0, 1].grid(True, which='both')
    ax[0, 1].legend(loc="upper right")

    # n = np.append([0], n)

    print("len(bins): " + str(len(bins)))
    print("len(n): " + str(len(n)))

    xaxis = np.zeros(bins.shape)
    for idx, val in enumerate(bins):
        if(idx + 1 < len(bins)):
            xaxis[idx+1] = np.mean([bins[idx], bins[idx+1]])
        else:
            xaxis[idx] = np.mean([bins[idx-1], bins[idx]])

    # print(xaxis)

    # ax[0, 1].plot(xaxis, n, '-p')



    label = '{} {}'.format(_label, '(sem outlier 1000)')
    # the histogram of the data
    ax[1, 0].set_title('Histograma c)')
    ax[1, 0].set_xlabel(r'Valor de $\rho$/2')
    # ax.set_ylabel('Probability density')
    # n, bins, patches = ax.hist(x, num_bins, density=True, label=label, histtype='step')
    ax[1, 0].set_ylabel('Ocorrências do valor')
    n, bins, patches = ax[1, 0].hist(x[x < 1000], num_bins, density=False, label=label, histtype='step', color='k')
    ax[1, 0].grid(True, which='both')
    ax[1, 0].legend(loc="upper right")

    # n = np.append([0], n)

    print("len(bins): " + str(len(bins)))
    print("len(n): " + str(len(n)))

    xaxis = np.zeros(bins.shape)
    for idx, val in enumerate(bins):
        if(idx + 1 < len(bins)):
            xaxis[idx+1] = np.mean([bins[idx], bins[idx+1]])
        else:
            xaxis[idx] = np.mean([bins[idx-1], bins[idx]])

    # print(xaxis)

    # ax[1, 0].plot(xaxis, n, '-p')



    label = '{} {}'.format(_label, '(sem outlier 1000)')
    # the histogram of the data
    ax[1, 1].set_title('Histograma d)')
    ax[1, 1].set_xlabel(r'Valor de $\rho$/2')
    # ax.set_ylabel('Probability density')
    # n, bins, patches = ax.hist(x, num_bins, density=True, label=label, histtype='step')
    ax[1, 1].set_ylabel('Ocorrências do valor (log)')
    n, bins, patches = ax[1, 1].hist(x[x < 1000], num_bins, density=False, label=label, histtype='step', color='k')
    ax[1, 1].set_yscale('log')
    ax[1, 1].grid(True, which='both')
    ax[1, 1].legend(loc="upper right")
    
    # n = np.append([0], n)

    print("len(bins): " + str(len(bins)))
    print("len(n): " + str(len(n)))

    xaxis = np.zeros(bins.shape)
    for idx, val in enumerate(bins):
        if(idx + 1 < len(bins)):
            xaxis[idx+1] = np.mean([bins[idx], bins[idx+1]])
        else:
            xaxis[idx] = np.mean([bins[idx-1], bins[idx]])

    # print(xaxis)

    # ax[1, 1].plot(xaxis, n, '-p')



    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()

    print('x_infos: ', json.dumps(x_infos, indent=4))

    return n, x_infos.get('mean'), x_infos.get('std'), bins



def getQualityHistogramsLogLog(file_path, file_n):
    x = np.fromfile(file_path, dtype=float, count=-1, sep='\n')
    x = x[x!=(-1)]
    # x = x[x < 1000]
    # x = np.log(x)

    x_infos = {
        'len': len(x),
        'min': min(x),
        'max': max(x),
        'sum': np.sum(x),
        'mean': np.mean(x),  # mean of distribution
        'std': np.std(x),  # standard deviation of distribution
        'var': np.var(x)  # variance of distribution
    }

    fig, ax = plt.subplots(2, 2)


    _label = file_path.split("/").pop()
    label = '{} {}'.format(_label, '(com outlier 1000)')
    num_bins = 1024

    # the histogram of the data
    ax[0, 0].set_title('Histograma a)')
    ax[0, 0].set_xlabel(r'Valor de $\rho$/2')
    # ax[0, 0].set_ylabel('Probability density')
    # n, bins, patches = ax[0, 0].hist(x, num_bins, density=True, label=label, histtype='step')
    ax[0, 0].set_ylabel('Ocorrências do valor')
    n, bins, patches = ax[0, 0].hist(x, num_bins, density=False, label=label, histtype='bar', color='k')
    ax[0, 0].grid(True, which='both')
    ax[0, 0].legend(loc="upper right")

    # n = np.append([0], n)

    print("len(bins): " + str(len(bins)))
    print("len(n): " + str(len(n)))

    xaxis = np.zeros(bins.shape)
    for idx, val in enumerate(bins):
        if(idx + 1 < len(bins)):
            xaxis[idx+1] = np.mean([bins[idx], bins[idx+1]])
        else:
            xaxis[idx] = np.mean([bins[idx-1], bins[idx]])

    # print(xaxis)

    # ax[0, 0].plot(xaxis, n, '-p')



    label = '{} {}'.format(_label, '(com outlier 1000)')
    # the histogram of the data
    ax[0, 1].set_title('Histograma b)')
    ax[0, 1].set_xlabel(r'Valor de $\rho$/2')
    # ax.set_ylabel('Probability density')
    # n, bins, patches = ax.hist(x, num_bins, density=True, label=label, histtype='step')
    ax[0, 1].set_ylabel('Ocorrências do valor (log)')
    n, bins, patches = ax[0, 1].hist(x, num_bins, density=False, label=label, histtype='bar', color='k')
    ax[0, 1].set_yscale('log')
    ax[0, 1].grid(True, which='both')
    ax[0, 1].legend(loc="upper right")

    # n = np.append([0], n)

    print("len(bins): " + str(len(bins)))
    print("len(n): " + str(len(n)))

    xaxis = np.zeros(bins.shape)
    for idx, val in enumerate(bins):
        if(idx + 1 < len(bins)):
            xaxis[idx+1] = np.mean([bins[idx], bins[idx+1]])
        else:
            xaxis[idx] = np.mean([bins[idx-1], bins[idx]])

    # print(xaxis)

    # ax[0, 1].plot(xaxis, n, '-p')



    label = '{} {}'.format(_label, '(com outlier 1000)')
    # the histogram of the data
    ax[1, 0].set_title('Histograma c)')
    ax[1, 0].set_xlabel(r'Valor de $\rho$/2 (escala log)')
    # ax.set_ylabel('Probability density')
    # n, bins, patches = ax.hist(x, num_bins, density=True, label=label, histtype='step')
    ax[1, 0].set_ylabel('Ocorrências do valor')
    n, bins, patches = ax[1, 0].hist(x, num_bins, density=False, label=label, histtype='bar', color='k')
    ax[1, 0].set_xscale('log')
    ax[1, 0].grid(True, which='both')
    ax[1, 0].legend(loc="upper right")

    # n = np.append([0], n)

    print("len(bins): " + str(len(bins)))
    print("len(n): " + str(len(n)))

    xaxis = np.zeros(bins.shape)
    for idx, val in enumerate(bins):
        if(idx + 1 < len(bins)):
            xaxis[idx+1] = np.mean([bins[idx], bins[idx+1]])
        else:
            xaxis[idx] = np.mean([bins[idx-1], bins[idx]])

    # print(xaxis)

    # ax[1, 0].plot(xaxis, n, '-p')



    label = '{} {}'.format(_label, '(com outlier 1000)')
    # the histogram of the data
    ax[1, 1].set_title('Histograma d)')
    ax[1, 1].set_xlabel(r'Valor de $\rho$/2 (escala log)')
    # ax.set_ylabel('Probability density')
    # n, bins, patches = ax.hist(x, num_bins, density=True, label=label, histtype='step')
    ax[1, 1].set_ylabel('Ocorrências do valor (escala log)')
    n, bins, patches = ax[1, 1].hist(x, num_bins, density=False, label=label, histtype='bar', color='k')
    ax[1, 1].set_yscale('log')
    ax[1, 1].set_xscale('log')
    ax[1, 1].grid(True, which='both')
    ax[1, 1].legend(loc="upper right")
    
    # n = np.append([0], n)

    print("len(bins): " + str(len(bins)))
    print("len(n): " + str(len(n)))

    xaxis = np.zeros(bins.shape)
    for idx, val in enumerate(bins):
        if(idx + 1 < len(bins)):
            xaxis[idx+1] = np.mean([bins[idx], bins[idx+1]])
        else:
            xaxis[idx] = np.mean([bins[idx-1], bins[idx]])

    # print(xaxis)

    # ax[1, 1].plot(xaxis, n, '-p')



    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()

    print('x_infos: ', json.dumps(x_infos, indent=4))

    return n, x_infos.get('mean'), x_infos.get('std'), bins




i = 0
x = np.array([None, None, None, None])
y = []
file_label = [None, None, None, None]
mean = []
std = []
bins = []
fig1, ax = plt.subplots(1, 1)
for file in file_path:
    # a, b, c, d = getQualityHistograms(file, i)
    # a, b, c, d = getQualityHistogramsLogLog(file, i)
    # a, b, c, d = getQualityPlot(ax, file, i)
    # a, b, c, d = getQualityPlot2(ax, file, i)
    # a, b, c, d = getQualityLogPlot(ax, file, i)
    getQualityPlotUnified(x, ax, file_path, file_label, i)

    # y.append(a)
    # mean.append(b)
    # std.append(c)
    # bins.append(d)
    i = i + 1



# # print('y: ', len(y))
# print('y[0][0]: ', y[0][0])
# print('y[0][len(y)-1]: ', y[0][len(y[0])-1])
# # print('mean: ', len(mean))
# # print('std: ', len(std))

# print('\n\n\n===============\n\n\n')

# # for k in range(0, len(y)):
# #     MSE = np.square(np.subtract(y[k], np.zeros(len(y[k])))).mean()
# #     print('MSE(0){}: '.format(file_path[k]), MSE)

# for k in range(0, len(y)):
#     w_avg = np.average(y[k], weights=bins[k])
#     print('WEIGHTED_AVG-{}: '.format(file_path[k]), w_avg)
#     # print(str(w_avg).replace('.', ','))




plt.show()
