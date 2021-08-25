import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(rc={'figure.figsize':(15,8)})
sns.set_style("darkgrid", {"axes.facecolor": ".85"})

sizes = {}
for file in os.listdir('fpga_modeling_reports/models_sizes'):
    if file.endswith(".txt"):
        model_name = file.split('_')[0]
        sizes[model_name] = {'ifmaps': [], 'filters': []}
        with open(os.path.join('fpga_modeling_reports/models_sizes', file)) as f:
            lines = f.read().splitlines()
            for l in lines:
                ifmap = float(l.split(',')[0])
                filter = float(l.split(',')[1])
                sizes[model_name]['ifmaps'].append(ifmap)
                sizes[model_name]['filters'].append(filter)

for k in sizes.keys():
    x_axis = np.arange(0, len(sizes[k]['ifmaps']))
    sns.lineplot(x=x_axis, y=sizes[k]['ifmaps'], label=k+' ifmaps', color='tab:orange')
    sns.lineplot(x=x_axis, y=sizes[k]['filters'], label=k+' filters', color='b')
    
    plt.hlines(y=1.375, xmin=x_axis[0], xmax=x_axis[-1], label='ZCU104', linestyles='dashed', colors='r')
    plt.hlines(y=4.0125, xmin=x_axis[0], xmax=x_axis[-1], label='ZCU102', linestyles='dashed', colors='g')

    plt.title(k + ' memory footprint')
    plt.legend()
    plt.xlabel('Conv Layers')
    plt.ylabel('Memory Footprint (MBs)')
    plt.savefig('fpga_modeling_reports/models_sizes/' + k + '_mem_footprint.png')

    plt.clf()