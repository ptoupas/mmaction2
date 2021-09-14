import argparse
import csv
import os
import math
import coloredlogs
import logging
import torch
import mmcv

import numpy as np
import seaborn as sns
import pandas as pd

from mmcv.parallel import collate, scatter
from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose
from collections import deque
from matplotlib import pyplot as plt
from functools import reduce

from numpy.lib.function_base import append
from numpy.testing._private.utils import assert_equal


coloredlogs.install(level='INFO')
np.set_printoptions(precision=5, suppress=True, linewidth=120)

EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode', 'FrameSelector'
]

class ModelFeatureMaps():
    def __init__(self, model, word_length, clock_freq, bram, dsp):
        self.model = model
        self.layers = {}
        self.activation = {}
        self.layers_break_down = {}
        self.wl = word_length
        self.wb = word_length//8
        self.clock_freq = clock_freq # in MHz
        self.cycles_per_sec = clock_freq*1e6
        self.clock_period = (1/(clock_freq*1e6))*1e9 # in nanosec
        self.bram_mem = 18 # Bram size is 18 Kbits or 2.25 KBytes
        self.fpga_bram = bram # each can hold a total of 18 Kbits or 2.25 KBytes
        self.fpga_dsps = dsp

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = [output.detach(), input[0].detach()]
        return hook

    def get_inter_feature_maps(self):
        for name, layer in self.model.named_modules():
            # print(name, layer)
            if isinstance(layer, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
                                torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
                                torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d,
                                torch.nn.AdaptiveAvgPool1d, torch.nn.AdaptiveAvgPool2d, torch.nn.AdaptiveAvgPool3d,
                                torch.nn.ReLU, torch.nn.ReLU6, torch.nn.Softmax, torch.nn.Sigmoid, torch.nn.Tanh,
                                torch.nn.Linear, mmcv.cnn.bricks.swish.Swish)):
                layer.register_forward_hook(self.get_activation(name))
            if hasattr(layer, 'weight'):
                if hasattr(layer, 'bias') and layer.bias is not None:
                    # print("name: {}. Layer {}\nWeights + Bias shape = {} + {}. Weights + Bias Parameters = {} + {} and MBs = {:.5f} + {:.5f}".format(name, layer, list(layer.weight.shape), list(layer.bias.shape), layer.weight.numel(), layer.bias.numel(), (layer.weight.numel()*self.wb)/1024**2, (layer.bias.numel()*self.wb)/1024**2))
                    self.layers[name] = [list(layer.weight.shape), layer.weight.numel(), (layer.weight.numel()*self.wb)/1024**2, list(layer.bias.shape), layer.bias.numel(), (layer.bias.numel()*self.wb)/1024**2]
                else:
                    # print("name: {}. Layer {}\nWeights shape = {}. Weights Parameters = {} and MBs = {:.5f}".format(name, layer, list(layer.weight.shape), layer.weight.numel(), (layer.weight.numel()*self.wb)/1024**2))
                    self.layers[name] = [list(layer.weight.shape), layer.weight.numel(), (layer.weight.numel()*self.wb)/1024**2]
                if isinstance(layer, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                    self.layers[name].append(layer.stride)
                    self.layers[name].append(layer.padding)
                    self.layers[name].append(layer.groups)
                    self.layers[name].append(layer.dilation)
            # print("="*40)
    def get_info(self):

        for n in self.activation.keys():
            layers_num = n.split('layer')[-1][:3]
            if len(layers_num.split(".")) <= 1:
                layers_num = n
            print(layers_num)
            if layers_num not in self.layers_break_down.keys():
                self.layers_break_down[layers_num] = {}
            if n in self.layers.keys():
                if ('conv' in n or 'se_module' in n) and 'bn' not in n:
                  macs = reduce((lambda x, y: x * y), self.layers[n][0]) * reduce((lambda x, y: x * y), list(self.activation[n][0].shape)[2:])
                elif 'fc' in n and 'bn' not in n:
                  macs = reduce((lambda x, y: x * y), self.layers[n][0])
                else:
                  macs = 0
                if 'bn' not in n and len(self.layers[n]) > 7:
                    print("Layer {} -> Input: {} - Output: {}, MBs {:.5f}\nFilters shape: {}. Params: {}, MBs: {:.5f}.\nBiases shape: {}. Params: {}, MBs: {:.5f}, MACs: {}.".format(n, list(self.activation[n][1].shape), list(self.activation[n][0].shape), (self.activation[n][0].numel()*self.wb)/1024**2, self.layers[n][0], self.layers[n][1], self.layers[n][2], self.layers[n][3], self.layers[n][4], self.layers[n][5], macs))
                    self.layers_break_down[layers_num][n] = {'input shape': list(self.activation[n][1].shape), 'output shape': list(self.activation[n][0].shape), 'intermediate feature map': (self.activation[n][0].numel()*self.wb)/1024**2, 'kernel shape': self.layers[n][0], 'kernel params': self.layers[n][1], 'kernel MBs': self.layers[n][2], 'bias shape': self.layers[n][3], 'bias params': self.layers[n][4], 'bias MBs': self.layers[n][5], 'MACs': macs}
                elif 'bn' not in n and len(self.layers[n]) <= 7:
                    print("Layer {} -> Input: {} - Output: {}, MBs {:.5f}\nFilters shape: {}. Params: {}, MBs: {:.5f}, MACs: {}.".format(n, list(self.activation[n][1].shape), list(self.activation[n][0].shape), (self.activation[n][0].numel()*self.wb)/1024**2, self.layers[n][0], self.layers[n][1], self.layers[n][2], macs))
                    self.layers_break_down[layers_num][n] = {'input shape': list(self.activation[n][1].shape), 'output shape': list(self.activation[n][0].shape), 'intermediate feature map': (self.activation[n][0].numel()*self.wb)/1024**2, 'kernel shape': self.layers[n][0], 'kernel params': self.layers[n][1], 'kernel MBs': self.layers[n][2], 'bias shape': [], 'bias params': -1, 'bias MBs': -1, 'MACs': macs}
                else:
                    print("Layer {} -> Input: {} - Output: {}, MBs {:.5f}\nFilters shape: {}. Params: {}, MBs: {:.5f}.\nBiases shape: {}. Params: {}, MBs: {:.5f}, MACs: {}.".format(n, list(self.activation[n][1].shape), list(self.activation[n][0].shape), (self.activation[n][0].numel()*self.wb)/1024**2, self.layers[n][0], self.layers[n][1], self.layers[n][2], self.layers[n][3], self.layers[n][4], self.layers[n][5], macs))
                    self.layers_break_down[layers_num][n] = {'input shape': list(self.activation[n][1].shape), 'output shape': list(self.activation[n][0].shape), 'intermediate feature map': (self.activation[n][0].numel()*self.wb)/1024**2, 'kernel shape': self.layers[n][0], 'kernel params': self.layers[n][1], 'kernel MBs': self.layers[n][2], 'bias shape': self.layers[n][3], 'bias params': self.layers[n][4], 'bias MBs': self.layers[n][5], 'MACs': macs}
            else:
                print("Layer {} -> Input: {} - Output: {}, MBs {:.5f}.".format(n, list(self.activation[n][1].shape), list(self.activation[n][0].shape), (self.activation[n][0].numel()*self.wb)/1024**2))
                self.layers_break_down[layers_num][n] = {'input shape': list(self.activation[n][1].shape), 'output shape': list(self.activation[n][0].shape), 'intermediate feature map': (self.activation[n][0].numel()*self.wb)/1024**2, 'kernel shape': [], 'kernel params': -1, 'kernel MBs': -1, 'bias shape': [], 'bias params': -1, 'bias MBs': -1, 'MACs': -1}
            print("="*40)

        print("="*40)
        print("="*40)
        total_macs = 0
        total_params = 0
        for key in self.layers_break_down.keys():
            feature_maps_size = 0
            feature_maps_size_opt = 0
            kernel_params = 0
            kernel_size = 0
            bias_params = 0
            bias_size = 0
            macs = 0
            prev_input_shape = self.layers_break_down[key][list(self.layers_break_down[key].keys())[0]]['input shape']
            for k in self.layers_break_down[key].keys():
                if ('conv' in k or 'pool' in k or 'fc' in k or 'bn' in k) and 'activate' not in k:
                    feature_maps_size += self.layers_break_down[key][k]['intermediate feature map']
                    macs += self.layers_break_down[key][k]['MACs']
                    if self.layers_break_down[key][k]['kernel params'] != -1:
                        kernel_params += self.layers_break_down[key][k]['kernel params']
                    if self.layers_break_down[key][k]['kernel MBs'] != -1:
                        kernel_size += self.layers_break_down[key][k]['kernel MBs']
                    if self.layers_break_down[key][k]['bias params'] != -1:
                        bias_params += self.layers_break_down[key][k]['bias params']
                    if self.layers_break_down[key][k]['bias MBs'] != -1:
                        bias_size += self.layers_break_down[key][k]['bias MBs']
                if ('conv' in k or 'pool' in k or 'fc' in k) and 'activate' not in k and 'bn' not in k:
                    feature_maps_size_opt += self.layers_break_down[key][k]['intermediate feature map']
            print("Layer {}. Input shape: {}. Feature maps size: {:.5f}({:.5f}) MBs. Kernel params: {}. Kernel size: {:.5f} MBs. Bias params: {}. Bias size: {:.5f} MBs. MACs: {}.".format(key, prev_input_shape, feature_maps_size, feature_maps_size_opt, kernel_params, kernel_size, bias_params, bias_size, macs/1e9))
            total_macs += macs
            total_params += kernel_params + bias_params
        print("="*40)
        print("="*40)
        print("Total MACs: {}.\nTotal parameters: {}.".format(total_macs/1e9, total_params))

    def modeling(self, csv_writer, layer_name, fold_setup, pb, kernel_size, cout, dout, hout, wout, cin, kd, kh, kw, coarse_in, coarse_out, fine):

        muls_unrl = math.ceil(cout * cin * kd * kh * kw * fine)
        adds_unrl_1 = cout * cin * (math.ceil(kd * kh * kw * fine) - 1)
        adds_unrl_2 = cout * cin
        bw_in_w_unrl = cin
        bw_out_w_unrl = cout
        thr_w_unrl = cout

        layer = layer_name
        folding = fold_setup
        mem_kb = ((pb*cin+kernel_size)*self.wb)/1e3
        mem_bram = math.ceil(mem_kb/(self.bram_mem/8))
        mem_util = (mem_bram/self.fpga_bram)*100

        bw_in_w = (bw_in_w_unrl / (coarse_in*coarse_out)) * fine
        bw_out_w = (bw_out_w_unrl / (coarse_in*coarse_out)) * fine
        muls = muls_unrl / (coarse_in*coarse_out)
        adds = adds_unrl_1 / (coarse_in*coarse_out) + adds_unrl_2 / (coarse_in*coarse_out)
        dsps = muls
        thr_w = (thr_w_unrl / (coarse_in*coarse_out)) * fine

        bw_in_gb = (self.cycles_per_sec*bw_in_w*self.wb)/1e9	
        bw_out_gb = (self.cycles_per_sec*bw_out_w*self.wb)/1e9
        dsps_util = (dsps/self.fpga_dsps)*100
        thr_v = (self.cycles_per_sec*thr_w)/(cout*dout*hout*wout)
        thr_go = ((muls + adds)*self.cycles_per_sec)/1e9

        csv_writer.writerow([layer, folding, mem_util, dsps_util, thr_v, bw_in_w, bw_out_w, mem_kb, mem_bram, bw_in_gb, bw_out_gb, muls, adds, dsps, thr_w, thr_go])

        print("Layer: {:<35} Folding: {}\nOn Chip Mem(KB) = {:<15.3f} On Chip Mem(BRAM) = {:<20.3f} On Chip Mem (BRAM %) = {:<20.3f}\nMem BW In(words/cycle) = {:<20.3f} Mem BW In(GBs/sec) = {:<20.3f} Mem BW Out(words/cycle) = {:<20.3f} Mem BW Out(GBs/sec) = {:<20.3f}\nMuls = {:<20.3f} Adds = {:<20.3f} DSPS = {:<20.3f} DSPS % = {:<20.3f}\nThroughtput(words/cycle) = {:<20.3f} Throughtput(outputs/sec) = {:<20.3f} Throughtput(GOps/sec) = {:.3f}".format(layer, folding, mem_kb, mem_bram, mem_util, bw_in_w, bw_in_gb, bw_out_w, bw_out_gb, muls, adds, dsps, dsps_util, thr_w, thr_v, thr_go))
        print("="*50)

    def get_conv_layers(self, file_name):
        if not os.path.exists(os.path.join(os.getcwd(), 'fpga_modeling_reports')):
            os.makedirs(os.path.join(os.getcwd(), 'fpga_modeling_reports'))
        csv_file = os.path.join(os.getcwd(), 'fpga_modeling_reports', file_name + '.csv')
        with open(csv_file, mode='w') as model_results:
            csv_writer = csv.writer(model_results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            csv_writer.writerow(["Layer", "Folding", "On-Chip Memory(BRAM %)", "DSPS %", "Throughtput(outputs/sec)", "Memory Bandwidth In(words/cycle)", "Memory Bandwidth Out(words/cycle)", "On-Chip Memory(KB)", "On-Chip Memory(BRAM)", "Memory Bandwidth In(GBs/sec)", "Memory Bandwidth Out(GBs/sec)", "Multipliers", "Adders", "DSPS", "Throughtput(words/cycle)", "Throughtput(GOps/sec)"])

            for n in self.activation.keys():
                name = n.split('.')
                if 'conv' in name or 'se_module.fc' in n:
                    in_shape = list(self.activation[n][1].shape)
                    din = in_shape[2]
                    hin = in_shape[3]
                    win = in_shape[4]

                    out_shape = list(self.activation[n][0].shape)
                    dout = out_shape[2]
                    hout = out_shape[3]
                    wout = out_shape[4]

                    kernel_shape = self.layers[n][0]
                    kd = kernel_shape[2]
                    kh = kernel_shape[3]
                    kw = kernel_shape[4]

                    kd_minus = kd-1 if kd > 1 else 1
                    kh_minus = kh-1 if kh > 1 else 1
                    kw_minus = kw-1 if kw > 1 else 1
                    
                    cin = kernel_shape[1]
                    cout = kernel_shape[0]

                    # In case of pointwise convolution we dont need the plane buffer anymore
                    if kd == 1 and kh == 1 and kw == 1:
                        pb = 1
                    else:
                        # Plane buffer + Line buffer (needed in conjuction with plane buffer) + Accumulator Buffer
                        pb = min((din*win*kh_minus), (win*hin*kd_minus)) + min((din*kw_minus), (win*kh_minus)) + cin

                    kernel_size = self.layers[n][1]

                    depthwise = False
                    if len(self.layers[n]) > 7:
                        # print("Layer: {}. Input Shape = {}. Output Shape = {}. Kernel Shape = {}. Stride = {}. Padding = {}. Groups = {}. Dilation = {}.".format(n, in_shape, out_shape, kernel_shape, self.layers[n][6], self.layers[n][7], self.layers[n][8], self.layers[n][9]))
                        # In case of depthwise convolution
                        if cout == self.layers[n][8]:
                            depthwise = True
                    else:
                        # print("Layer: {}. Input Shape = {}. Output Shape = {}. Kernel Shape = {}. Stride = {}. Padding = {}. Groups = {}. Dilation = {}.".format(n, in_shape, out_shape, kernel_shape, self.layers[n][3], self.layers[n][4], self.layers[n][5], self.layers[n][6]))
                        if cout == self.layers[n][5]:
                            depthwise = True

                    if depthwise and (not pb == 1):
                        pb = pb - cin

                        if cin == 1:
                            coarse_in_config = [1]
                        else:
                            coarse_in_config = [1, cin]
                        if cout == 1:
                            coarse_out_config = [1]
                        else:
                            coarse_out_config = [1, cout]
                        fine_config = [0.25, 0.5, 0.75, 1]

                        for coarse_in in coarse_in_config:
                            for coarse_out in coarse_out_config:
                                for fine in fine_config:
                                    if coarse_in == cin and coarse_out == cout and (not fine == 1):
                                        continue
                                    coarse_in_name = str(coarse_in) if coarse_in == 1 else 'Cin'
                                    coarse_out_name = str(coarse_out) if coarse_out == 1 else 'Cout'
                                    fine_name = str(fine)
                                    folding_name = "Coarse({}/{}) - Fine({})".format(coarse_in_name, coarse_out_name, fine_name)
                                    
                                    self.modeling(csv_writer, name, folding_name, pb, kernel_size, cout, dout, hout, wout, cin, kd, kh, kw, coarse_in, coarse_out, fine)

                    csv_writer.writerow(["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"])
                    # print("**"*100)

def find_pareto(scores):
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            # print("[{}][0] = {}. [{}][0] = {}. [{}][1] = {}. [{}][1] = {}.".format(j, scores[j][0], i, scores[i][0], j, scores[j][1], i, scores[i][1]))
            if (scores[j][0] >= scores[i][0] and scores[j][1] <= scores[i][1]) and (scores[j][0] > scores[i][0] or scores[j][1] < scores[i][1]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]

def plot_graph(x, y, leg, name, type, model_name):
    se_layer = True if "Se" in name.split("_") else False
    sns.set(rc={'figure.figsize':(15,8)})
    sns.set_style("whitegrid")
    
    dsps_dir = os.path.join(os.getcwd(), 'fpga_modeling_reports', 'graphs', model_name, 'throughput_dsps')
    if not os.path.exists(dsps_dir):
        os.makedirs(dsps_dir)
    mem_bw_dir = os.path.join(os.getcwd(), 'fpga_modeling_reports', 'graphs', model_name, 'throughput_mem_bw')
    if not os.path.exists(mem_bw_dir):
        os.makedirs(mem_bw_dir)
    if type == 'DSPS':

        scores = np.zeros((len(x), 2))
        scores[:,0] = x
        scores[:,1] = y[0]
        pareto = find_pareto(scores)
        pareto_front = scores[pareto]

        pareto_front_df = pd.DataFrame(pareto_front)
        pareto_front_df.sort_values(0, inplace=True)
        pareto_front = pareto_front_df.values

        sns.scatterplot(x=np.array(x), y=np.array(y[0]), hue=leg, style=leg, s=50)
        sns.lineplot(x=pareto_front[:, 0], y=pareto_front[:, 1], color='red')

        plt.title(name)
        plt.xlabel('Throughtput(outputs/sec)')
        plt.ylabel('DSPS %')
        if max(y[0]) > 100:
            plt.yscale("log")
        else:
            plt.ylim([-5, max(100, max(y[0]) + 0.1*max(y[0]))])
        if max(x) > 100:
            plt.xscale("log")
        if se_layer:
            legd = []
            for l in pareto:
                legd.append(leg[l])
            plt.legend(legd, frameon=False, prop={"size":8}, loc='upper right', bbox_to_anchor=(1.11, 1.12), borderaxespad=0.)
        else:
            plt.legend(frameon=False, prop={"size":8}, loc='upper right', bbox_to_anchor=(1.11, 1.12), borderaxespad=0.)

        file_name = name.replace('.', '_') + '.jpg'
        plt.savefig(os.path.join(dsps_dir, file_name))
        plt.clf()
    elif type == 'Memory Bandwidth':

        sns.scatterplot(x=np.array(x), y=np.array(y[0]), hue=leg, style=leg, s=50)

        plt.title(name)
        plt.xlabel('Throughtput(outputs/sec)')
        plt.ylabel('Memory Bandwidth IN (GBs/sec)')
        if max(x) > 100:
            plt.xscale("log")
        plt.legend(frameon=False, prop={"size":8}, loc='upper right', bbox_to_anchor=(1.11, 1.12), borderaxespad=0.)

        file_name = name.replace('.', '_') + '_in.jpg'
        plt.savefig(os.path.join(mem_bw_dir, file_name))
        plt.clf()


        sns.scatterplot(x=np.array(x), y=np.array(y[1]), hue=leg, style=leg, s=50)

        plt.title(name)
        plt.xlabel('Throughtput(outputs/sec)')
        plt.ylabel('Memory Bandwidth OUT (GBs/sec)')
        if max(x) > 100:
            plt.xscale("log")
        plt.legend(frameon=False, prop={"size":5}, loc='upper right', bbox_to_anchor=(1.05, 1.05), borderaxespad=0.)

        file_name = name.replace('.', '_') + '_out.jpg'
        plt.savefig(os.path.join(mem_bw_dir, file_name))
        plt.clf()

def performance_graphs(file_name="x3d_m", layer_to_plot=None):
    
    csv_file = os.path.join(os.getcwd(), 'fpga_modeling_reports', file_name + '.csv')
    with open(csv_file, mode='r') as model_results:
        csv_reader = csv.reader(model_results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # csv_reader = csv.DictReader(model_results)

        cols = {}
        for i, c in enumerate(next(csv_reader)):
            cols[c] = i
        print(cols)

        folding = []
        dsp_util = []
        mem_bw_in = []
        mem_bw_out = []
        throughput = []

        first_layer = True
        for i, row in enumerate(csv_reader):
            if "-" in row or (layer_to_plot is not None and row[cols['Layer']] not in layer_to_plot):
                continue

            if first_layer and (i == 0 or (layer_to_plot is not None and row[cols['Layer']] in layer_to_plot)):
                prev_layer = row[cols['Layer']]
                first_layer = False

            if row[cols['Layer']] == prev_layer:
                folding.append(row[cols['Folding']])
                dsp_util.append(float(row[cols['DSPS %']]))
                mem_bw_in.append(float(row[cols['Memory Bandwidth In(GBs/sec)']]))
                mem_bw_out.append(float(row[cols['Memory Bandwidth Out(GBs/sec)']]))
                throughput.append(float(row[cols['Throughtput(outputs/sec)']]))
            else:
                plot_graph(throughput, [dsp_util], folding, prev_layer, 'DSPS', file_name)
                plot_graph(throughput, [mem_bw_in, mem_bw_out], folding, prev_layer, 'Memory Bandwidth', file_name)

                folding.clear()
                dsp_util.clear()
                mem_bw_in.clear()
                mem_bw_out.clear()
                throughput.clear()
                
                folding.append(row[cols['Folding']])
                dsp_util.append(float(row[cols['DSPS %']]))
                mem_bw_in.append(float(row[cols['Memory Bandwidth In(GBs/sec)']]))
                mem_bw_out.append(float(row[cols['Memory Bandwidth Out(GBs/sec)']]))
                throughput.append(float(row[cols['Throughtput(outputs/sec)']]))

            prev_layer = row[cols['Layer']]

        plot_graph(throughput, [dsp_util], folding, prev_layer, 'DSPS', file_name)
        plot_graph(throughput, [mem_bw_in, mem_bw_out], folding, prev_layer, 'Memory Bandwidth', file_name)

def get_paretto(file_name="x3d_m"):
    
    csv_file_par = os.path.join(os.getcwd(), 'fpga_modeling_reports', file_name + '_pareto.csv')
    with open(csv_file_par, mode='w') as pareto_results:
        csv_writer_par = csv.writer(pareto_results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csv_writer_par.writerow(["Layer", "Folding", "On-Chip Memory(BRAM %)", "DSPS %", "Consumption(inputs/sec)", "Throughtput(outputs/sec)", "Memory Bandwidth In(words/cycle)", "Memory Bandwidth Out(words/cycle)", "On-Chip Memory(KB)", "On-Chip Memory(BRAM)", "Memory Bandwidth In(GBs/sec)", "Memory Bandwidth Out(GBs/sec)", "Multipliers", "Adders", "DSPS", "Throughtput(words/cycle)", "Throughtput(GOps/sec)"])

        csv_file = os.path.join(os.getcwd(), 'fpga_modeling_reports', file_name + '.csv')
        with open(csv_file, mode='r') as model_results:
            csv_reader = csv.reader(model_results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            cols = {}
            for i, c in enumerate(next(csv_reader)):
                cols[c] = i
            print(cols)

            rows = []

            first_layer = True
            for i, row in enumerate(csv_reader):
                if "-" in row :
                    continue

                if first_layer and i == 0:
                    prev_layer = row[cols['Layer']]
                    first_layer = False

                if row[cols['Layer']] == prev_layer:
                    rows.append(row)
                else:

                    through = [r[5] for r in rows]
                    dsps = [r[3] for r in rows]

                    scores = np.zeros((len(through), 2))
                    scores[:,0] = through
                    scores[:,1] = dsps
                    pareto = find_pareto(scores)
                    
                    for p in pareto:
                        csv_writer_par.writerow(rows[p])
                    csv_writer_par.writerow(["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"])
                    rows.clear()
                    
                    rows.append(row)

                prev_layer = row[cols['Layer']]

            through = [r[5] for r in rows]
            dsps = [r[3] for r in rows]

            scores = np.zeros((len(through), 2))
            scores[:,0] = through
            scores[:,1] = dsps
            pareto = find_pareto(scores)
            
            for p in pareto:
                csv_writer_par.writerow(rows[p])
            csv_writer_par.writerow(["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"])

def get_partition_layers(layers, model_name):
    final_layers = []
    if model_name == 'x3d_m':
        layer_type_1 = ['Relu', 'Conv', 'BatchNormalization', 'Relu', 'Conv', 'BatchNormalization', 'SqueezeExcitation', 'Swish', 'Conv', 'BatchNormalization', 'Conv', 'BatchNormalization', 'Add']
        layer_type_2 = ['Relu', 'Conv', 'BatchNormalization', 'Relu', 'Conv', 'BatchNormalization', 'SqueezeExcitation', 'Swish', 'Conv', 'BatchNormalization', 'Add']
        layer_type_3 = ['Relu', 'Conv', 'BatchNormalization', 'Relu', 'Conv', 'BatchNormalization', 'Swish', 'Conv', 'BatchNormalization', 'Add']    
        layer_queue = deque(maxlen=13)
        layer_queue_operations = deque(maxlen=13)
        for k in layers.keys():
            layer_queue_operations.append(layers[k]['operation'])
            layer_queue.append(k)
            if list(layer_queue_operations) == layer_type_1:
                final_layers.append(list(layer_queue))
            if list(layer_queue_operations)[:-2] == layer_type_2:
                final_layers.append(list(layer_queue)[:-2])
            if list(layer_queue_operations)[:-3] == layer_type_3:
                final_layers.append(list(layer_queue)[:-3])
    elif model_name == 'i3d':
        layer_type_1 = ['Conv', 'BatchNormalization', 'Relu', 'Conv', 'BatchNormalization', 'Relu', 'Conv', 'BatchNormalization', 'Conv', 'BatchNormalization', 'Add']
        layer_type_2 = ['Conv', 'BatchNormalization', 'Relu', 'Conv', 'BatchNormalization', 'Relu', 'Conv', 'BatchNormalization', 'Add']
        layer_queue = deque(maxlen=11)
        layer_queue_operations = deque(maxlen=11)
        for k in layers.keys():
            layer_queue_operations.append(layers[k]['operation'])
            layer_queue.append(k)
            if list(layer_queue_operations) == layer_type_1:
                final_layers.append(list(layer_queue))
            if list(layer_queue_operations)[:-2] == layer_type_2:
                final_layers.append(list(layer_queue)[:-2])
    return final_layers


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 parse model')
    parser.add_argument('model_name', help='name of the har model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--use_frames', action='store_true', help='whether to use video decoder or raw frame decoder')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--label', default=None, help='label file')
    parser.add_argument('--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument('--imshape', type=int, nargs="+", default=[224, 224, 3], help='image size for inference')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    device = torch.device(args.device)
    use_frames = False
    if args.use_frames == 'False':
        use_frames = False
    if args.use_frames == 'True':
        use_frames = True
    model = init_recognizer(args.config, device=device, use_frames=use_frames)
    
    # Target FPGA Zynq UltraScale+ MPSoC ZCU104. Assuming clock frequency of 100 MHz.
    # The actual BRAM size is 11 Mbits (1.375 MBytes). This divided by the 18 Kbits size of each BRAM gives a total of 624 BRAM units.
    # The ZCU104 has also 27 Mbits (3.375 MBytes) of URAM. This divided by the 288 Kbits size of each URAM gives a total of 96 URAM units.
    # The ZCU104 has 20 GTH gigabit transceivers (16.3 Gb/s or 2.03 GB/s) on the PL-size
    feature_maps = ModelFeatureMaps(model=model, word_length=16, clock_freq=100, bram=624, dsp=1728)
    feature_maps.get_inter_feature_maps()

    random_img = np.random.randn(args.imshape[0], args.imshape[1], args.imshape[2])

    data = dict(img_shape=None, modality='RGB', label=-1)

    # prepare test pipeline from non-camera pipeline
    cfg = model.cfg
    sample_length = 0
    pipeline = cfg.test_pipeline
    pipeline_ = pipeline.copy()
    for step in pipeline:
        if 'SampleFrames' in step['type']:
            step['num_clips'] = 1
            sample_length = step['clip_len'] * step['num_clips']
            data['num_clips'] = step['num_clips']
            data['clip_len'] = step['clip_len']
            pipeline_.remove(step)
        if step['type'] in EXCLUED_STEPS:
            # remove step to decode frames
            pipeline_.remove(step)
    test_pipeline = Compose(pipeline_)
    print(test_pipeline)
    assert sample_length > 0

    data_in = []
    for _ in range(data['clip_len']):
        data_in.append(random_img)

    data['imgs'] = data_in
    if data['img_shape'] is None:
        data['img_shape'] = random_img.shape[:2]

    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        data = scatter(data, [device])[0]

    with torch.no_grad():
        scores = model(return_loss=False, **data)[0]

    feature_maps.get_info()
    
    feature_maps.get_conv_layers(file_name=args.model_name)

    # performance_graphs(file_name=args.model_name, layer_to_plot=None)

if __name__ == '__main__':
    main()