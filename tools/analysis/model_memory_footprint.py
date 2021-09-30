import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(rc={'figure.figsize':(15,8)})
sns.set_style("whitegrid")
FONT_SIZE = 22
FONT_SIZE_TICKS = 17
FONT_SIZE_LEGEND = 15

mode = 'ifmaps' # one of ['ifmaps', 'filters']
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
    
    if k == 'resnet50-v1-7' or k == 'vgg16-7':
        if mode == 'filters':
            sns.lineplot(x=x_axis, y=sizes[k]['filters'], label=k, linewidth = 1.75, marker='X')
        elif mode == 'ifmaps':
            sns.lineplot(x=x_axis, y=sizes[k]['ifmaps'], label=k, linewidth = 1.75, marker='X')
    else:
        if mode == 'filters':
            sns.lineplot(x=x_axis, y=sizes[k]['filters'], label=k, linewidth = 1.75, marker='8')
        elif mode == 'ifmaps':
            sns.lineplot(x=x_axis, y=sizes[k]['ifmaps'], label=k, linewidth = 1.75, marker='8')
    
    # Old code
    # sns.lineplot(x=x_axis, y=sizes[k]['ifmaps'], label=k+' ifmaps', color='tab:orange')
    # sns.lineplot(x=x_axis, y=sizes[k]['filters'], label=k+' filters', color='b')

plt.hlines(y=1.375, xmin=x_axis[0], xmax=115, label='ZCU104 BRAM', linestyles='dashed', colors='r')
plt.hlines(y=4.0125, xmin=x_axis[0], xmax=115, label='ZCU102 BRAM', linestyles='dashed', colors='g')
# Old code
# plt.hlines(y=1.375, xmin=x_axis[0], xmax=x_axis[-1], label='ZCU104', linestyles='dashed', colors='r')
# plt.hlines(y=4.0125, xmin=x_axis[0], xmax=x_axis[-1], label='ZCU102', linestyles='dashed', colors='g')
# plt.title(k + ' memory footprint')

if mode == 'filters':
    plt.title('Weights memory footprint per convolutional layer', fontsize=FONT_SIZE)
elif mode == 'ifmaps':
    plt.title('Feature maps memory footprint per convolutional layer', fontsize=FONT_SIZE)

plt.legend(framealpha=0.5, fontsize=FONT_SIZE_LEGEND)
plt.xlabel('Conv Layers', fontsize=FONT_SIZE)
plt.ylabel('Memory Footprint (MBs)', fontsize=FONT_SIZE)

plt.xticks(fontsize=FONT_SIZE_TICKS)
plt.yticks(fontsize=FONT_SIZE_TICKS)

if mode == 'filters':
    plt.savefig('fpga_modeling_reports/models_sizes/weights_mem_footprint.png')
elif mode == 'ifmaps':
    plt.savefig('fpga_modeling_reports/models_sizes/fmaps_mem_footprint.png')

# Old code
# plt.savefig('fpga_modeling_reports/models_sizes/' + k + '_mem_footprint.png')

plt.clf()