import argparse
import csv
import os
import sys
import math
import coloredlogs
import logging
import onnx
import json
import time
import ray
import itertools
import concurrent.futures

import numpy as np
import seaborn as sns
import pandas as pd

from collections import deque
from matplotlib import pyplot as plt
from tqdm import tqdm

from numpy.lib.function_base import append
from numpy.testing._private.utils import assert_equal
from functools import reduce

from multiprocessing import Pool
from timebudget import timebudget

timebudget.set_quiet()       # don't show measurements as they happen
timebudget.report_at_exit()  # Generate report when the program exits

coloredlogs.install(level='WARNING')
logging.basicConfig(level=logging.WARNING)
np.set_printoptions(precision=5, suppress=True, linewidth=150)
sns.set(rc={'figure.figsize':(15,8)})
sns.set_style("whitegrid")

@timebudget
def multithreaded_modeling(operation, input, pool):
    pool.starmap(operation, input)

class ModelFeatureMapsOnnx():

    def __init__(self, model, word_length, clock_freq, bram, dsp, mem_bw):
        self.model_name = model
        self.model_path = model + ".onnx"
        self.layers = {}
        self.modules = {}
        self.wl = word_length
        self.wb = word_length//8
        self.clock_freq = clock_freq # in MHz
        self.cycles_per_sec = clock_freq*1e6
        self.clock_period = (1/(self.cycles_per_sec))*1e9 # in nanosec
        self.bram_mem = 18 # Bram size is 18 Kbits or 2.25 KBytes
        self.fpga_bram = bram # each can hold a total of 18 Kbits or 2.25 KBytes
        self.fpga_dsps = dsp
        self.mem_bandwidth = mem_bw * 1e9 # in b/s (bits per second)
        self.max_words_per_cycle = (self.mem_bandwidth / self.wl) / self.cycles_per_sec

        self.op_list = ['Conv', 'BatchNormalization', 'Relu', 'GlobalAveragePool', 'AveragePool', 'MaxPool', 'Sigmoid', 'Mul', 'Add', 'Div', 'MatMul', 'Gemm', 'Elu', 'Flatten', 'GRU', 'HardSigmoid', 'LSTM', 'LeakyRelu', 'PRelu', 'RNN', 'Selu', 'Tanh', 'Celu', 'HardSwish', 'Softmax']
        self.onnx_model = onnx.load(self.model_path)
        self.onnx_model = onnx.shape_inference.infer_shapes(self.onnx_model)
        onnx.checker.check_model(self.onnx_model)

        # print(onnx.helper.printable_graph(self.onnx_model.graph))

    @timebudget
    def balance_module_rates_new(self, rate_graph):
        
        rate_ratio = [ abs(rate_graph[i,i]/rate_graph[i-1,i]) for i in range(1,rate_graph.shape[1]-1) ]
        # print("RATE RATIO")
        # print(rate_ratio)
        # print("-"*50)
        for i in range(1,rate_graph.shape[1]):
            # start from end
            layer = rate_graph.shape[1]-i

            if abs(rate_graph[layer-1,layer]) > abs(rate_graph[layer-1,layer-1]):
                # propogate forward
                for j in range(layer,rate_graph.shape[1]):
                        if(abs(rate_graph[j-1,j]) <= abs(rate_graph[j-1,j-1])):
                            break
                        rate_graph[j-1,j]   = abs(rate_graph[j-1,j-1])
                        if j < rate_graph.shape[0]:
                            rate_graph[j,j] = -rate_graph[j-1,j-1]*rate_ratio[j-1]

            elif abs(rate_graph[layer-1,layer]) < abs(rate_graph[layer-1,layer-1]):
                # propogate backward
                for j in range(0,layer):
                        if(abs(rate_graph[layer-j-1,layer-j]) >= abs(rate_graph[layer-j-1,layer-j-1])):
                            break
                        rate_graph[layer-j-1,layer-j-1]  = -abs(rate_graph[layer-j-1,layer-j])
                        if layer-j-1 > 0:
                            rate_graph[layer-j-2,layer-j-1] = -rate_graph[layer-j-1,layer-j-1]/rate_ratio[layer-1-j-1]
        return rate_graph

    def balance_module_rates(self, rate_graph):
        
        rate_ratio = [ abs(rate_graph[i,i+1]/rate_graph[i,i]) for i in range(rate_graph.shape[0]) ]
        # print("RATE RATIO")
        # print(rate_ratio)
        # print("-"*50)
        for i in range(1,rate_graph.shape[0]):
            # start from end
            layer = rate_graph.shape[0]-i

            if abs(rate_graph[layer,layer]) > abs(rate_graph[layer-1,layer]):
                # propogate forward
                for j in range(layer,rate_graph.shape[0]):
                        if(abs(rate_graph[j,j]) <= abs(rate_graph[j-1,j])):
                            break
                        rate_graph[j,j]   = abs(rate_graph[j-1,j])
                        rate_graph[j,j+1] = -rate_graph[j,j]*rate_ratio[j]

            elif abs(rate_graph[layer,layer]) < abs(rate_graph[layer-1,layer]):
                # propogate backward
                for j in range(0,layer):
                        if(abs(rate_graph[layer-j,layer-j]) >= abs(rate_graph[layer-j-1,layer-j])):
                            break
                        rate_graph[layer-j-1,layer-j]   = -abs(rate_graph[layer-j,layer-j])
                        rate_graph[layer-j-1,layer-j-1] = -rate_graph[layer-j-1,layer-j]/rate_ratio[layer-j-1]
        return rate_graph

    def get_shape_onnx(self, input, is_initializer=False):
        if is_initializer:
            return list(input.dims)

        tensor_type = input.type.tensor_type
        tensor_shape = []
        if (tensor_type.HasField("shape")):
            for d in tensor_type.shape.dim:
                if (d.HasField("dim_value")):
                    tensor_shape.append(d.dim_value)
                elif (d.HasField("dim_param")):
                    tensor_shape.append(d.dim_param)
                else:
                    logging.critical("Couldn't read the dimensions of tensor")
                    sys.exit() 
        return tensor_shape

    def is_in_inputs(self, input):
        exists = False
        shape = []
        for inp in self.onnx_model.graph.input:
            if input == inp.name:
                exists = True
                shape = self.get_shape_onnx(inp)
                break

        for inp in self.onnx_model.graph.initializer:
            if input == inp.name:
                exists = True
                shape = self.get_shape_onnx(inp, is_initializer=True)
                break

        return exists, shape

    def calculate_conv_out_shape(self, in_shape, groups, dilation, kernel, padding, stride):
        assert kernel[1]*groups == in_shape[1], 'filter channels and input channels does not match'
        out_shape = []

        dout = math.floor((in_shape[2] + 2 * padding[0] - dilation[0] * (kernel[2] - 1) - 1)/stride[0] + 1)
        hout = math.floor((in_shape[3] + 2 * padding[1] - dilation[1] * (kernel[3] - 1) - 1)/stride[1] + 1)
        wout = math.floor((in_shape[4] + 2 * padding[2] - dilation[2] * (kernel[4] - 1) - 1)/stride[2] + 1)
            
        out_shape.append(in_shape[0])
        out_shape.append(kernel[0])
        out_shape.append(dout)
        out_shape.append(hout)
        out_shape.append(wout)

        return out_shape

    def get_output_shape(self, out_node):
        for fmap in self.onnx_model.graph.value_info:
            if fmap.name == out_node:
                out_shape = []
                for dim in fmap.type.tensor_type.shape.dim:
                    out_shape.append(dim.dim_value)
                return out_shape
        return []

    @timebudget
    def from_onnx(self):

        layers_outputs = {}
        first_layer = True
        self.input_shape = self.get_shape_onnx(self.onnx_model.graph.input[0])
        # if self.model_name == "resnet3D":
        #     self.input_shape = self.get_shape_onnx(self.onnx_model.graph.input[0])
        logging.info("Model input shape = {}".format(self.input_shape))
        assert len(self.onnx_model.graph.input) == 1, "Model has multiple inputs or the initializers are duplicated to inputs as well. Aborting..."
        for n in self.onnx_model.graph.node:
            if n.op_type in self.op_list:

                logging.info("Node ({}) inputs: {}".format(n.name, n.input))
                
                skip_layer = False

                layer_input_shape = []
                layer_input_id = []

                dilation = []
                groups = 0
                kernel = []
                bias = []
                running_mean = []
                running_var = []
                padding = []
                stride = []

                if first_layer:
                    for i_num, layer_in in enumerate(n.input):
                        exists, shape = self.is_in_inputs(layer_in)
                        if exists and not i_num == 0:
                            logging.info("VARIABLE INPUT {} - {}".format(layer_in, shape))
                            if 'weight' in layer_in or i_num == 1:
                                kernel = shape.copy()
                            if 'bias' in layer_in or i_num == 2:
                                bias = shape.copy()
                            if 'running_mean' in layer_in or i_num == 3:
                                running_mean = shape.copy()
                            if 'running_var' in layer_in or i_num == 4:
                                running_var = shape.copy()
                        else:
                            layer_input_shape.append(self.input_shape.copy())
                            layer_input_id.append(layer_in)
                            logging.info("MODEL INPUT {} - {}".format(layer_in, self.input_shape))
                    first_layer = False
                else:
                    for i_num, layer_in in enumerate(n.input):
                        skip_unsupported_layer = False
                        if n.op_type =='MatMul' and self.model_name == 'x3d_m' and layer_in == '968':
                            logging.warning('Workaround to support final FC layers in x3d_m model. This code is specificaly written for this model and won\'t work in other models')
                            skip_unsupported_layer = True
                            layer_in = '960'
                        if n.op_type =='MatMul' and self.model_name == 'x3d_m' and layer_in == '969':
                            logging.warning('Workaround to support final FC layers in x3d_m model. This code is specificaly written for this model and won\'t work in other models')
                            layer_in = 'cls_head.fc1.weight'

                        exists, shape = self.is_in_inputs(layer_in)
                        if exists and not i_num == 0:
                            logging.info("VARIABLE INPUT {} - {}".format(layer_in, shape))
                            if 'weight' in layer_in or i_num == 1:
                                kernel = shape.copy()
                            if 'bias' in layer_in or i_num == 2:
                                bias = shape.copy()
                            if 'running_mean' in layer_in or i_num == 3:
                                running_mean = shape.copy()
                            if 'running_var' in layer_in or i_num == 4:
                                running_var = shape.copy()
                        else:
                            if layer_in not in layers_outputs.keys():
                                skip_layer = True
                                break

                            if skip_unsupported_layer:
                                layer_input_shape.append(layers_outputs[layer_in][:2])
                            else:
                                layer_input_shape.append(layers_outputs[layer_in])
                            layer_input_id.append(layer_in)
                            logging.info("INTERMEDIADE INPUT {} - {}".format(layer_in, layer_input_shape[-1]))
                if skip_layer:
                    logging.warning("Could not find the input of layer {}. This layer will be skipped in the analysis".format(n.name))
                    continue
                out_shape = []
                if n.op_type == 'Conv':
                    for attr in n.attribute:
                        if attr.name == "dilations":
                            dilation = list(attr.ints)
                        elif attr.name == "group":
                            groups = attr.i
                        elif attr.name == "pads":
                            padding = list(attr.ints[:3])
                        elif attr.name == "strides":
                            stride = list(attr.ints)
                    out_shape = self.get_output_shape(n.output[0])
                elif 'Pool' in n.op_type:
                    if n.op_type == 'GlobalAveragePool':
                        out_shape = self.get_output_shape(n.output[0])
                    else:
                        for attr in n.attribute:
                            if attr.name == "kernel_shape":
                                kernel_shape = list(attr.ints)
                            elif attr.name == "pads":
                                padding = list(attr.ints[:3])
                            elif attr.name == "strides":
                                stride = list(attr.ints)
                        out_shape = self.get_output_shape(n.output[0])
                elif n.op_type == 'Mul' or n.op_type == 'Add' or n.op_type == 'Div':
                    out_shape = self.get_output_shape(n.output[0])
                elif n.op_type == 'MatMul':
                    if len(kernel) > 0:
                        out_shape.append(layer_input_shape[0][0])
                        out_shape.append(kernel[0])
                    else:
                        logging.warning('Case not supported yet. Skipping...')
                        continue
                elif n.op_type == 'Gemm':
                    out_shape.append(layer_input_shape[0][0])
                    out_shape.append(kernel[0])
                else:
                    out_shape = layer_input_shape[0].copy()
                
                assert len(n.output) == 1, "More than one outputs for layer {}".format(n.name)
                logging.info("OUTPUT {} - {}".format(n.output, out_shape))
                layers_outputs[n.output[0]] = out_shape
                self.layers[n.name] = {"operation": n.op_type,
                                "input": layer_input_shape,
                                "input_id": layer_input_id,
                                "output": out_shape,
                                "output_id": n.output[0],
                                "kernel": kernel,
                                "padding": padding,
                                "stride": stride,
                                "groups": groups,
                                "dilation": dilation,
                                "bias": bias,
                                "running_mean": running_mean,
                                "running_var": running_var,}

    def get_layer_num_from_output(self, out_node):
        layer_num = -1
        for node in self.onnx_model.graph.node:
            if node.output[0] == out_node:
                if node.op_type == "Conv" or node.op_type == "BatchNormalization" or node.op_type == "Gemm":
                    layer_num = node.input[1].split(".")[1] + '.' + node.input[1].split(".")[2]
                    layer_num = layer_num.split("layer")[1] if len(layer_num.split("layer")) > 1 else layer_num 
        return layer_num

    @timebudget
    def get_info(self):
        file = open("fpga_modeling_reports/models_sizes/" + self.model_name + "_conv_sizes.txt", "w")
        if not os.path.exists(os.path.join(os.getcwd(), 'fpga_modeling_reports', 'roofline_modeling')):
                os.makedirs(os.path.join(os.getcwd(), 'fpga_modeling_reports', 'roofline_modeling'))
        csv_file = os.path.join(os.getcwd(), 'fpga_modeling_reports', 'roofline_modeling', self.model_name + '.csv')
        with open(csv_file, mode='w') as model_results:
            csv_writer = csv.writer(model_results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            csv_writer.writerow(["Layer", "Node", "Ifmaps", "Ofmaps", "params", "macs (G)"])
            for k in self.layers.keys():
                layer_num = self.get_layer_num_from_output(self.layers[k]['output_id'])
                ifmap = np.prod(np.array(self.layers[k]['input'][0]))
                ofmap = np.prod(np.array(self.layers[k]['output']))
                param = np.prod(np.array(self.layers[k]['kernel'])) if not len(self.layers[k]['kernel']) == 0 else 0 
                param += np.prod(np.array(self.layers[k]['bias'])) if not len(self.layers[k]['bias']) == 0 else 0
                macs = 0
                if self.layers[k]['operation'] == "Conv":
                    kernel = self.layers[k]['kernel'][1:]
                    macs = ofmap*np.prod(np.array(kernel))
                elif self.layers[k]['operation'] == "BatchNormalization":
                    param += np.prod(np.array(self.layers[k]['running_mean'])) if not len(self.layers[k]['running_mean']) == 0 else 0
                    param += np.prod(np.array(self.layers[k]['running_var'])) if not len(self.layers[k]['running_var']) == 0 else 0
                    macs = ifmap*3
                elif self.layers[k]['operation'] == "MatMul" or self.layers[k]['operation'] == "Gemm":
                    macs = np.prod(np.array(self.layers[k]['kernel']))
                elif self.layers[k]['operation'] == "GlobalAveragePool":
                    macs = ofmap*2
                logging.info("Layer Num: {}, Node {}: out fmaps {}-{}(MBs) params {}-{}(MBs) macs {}".format(layer_num, k, ofmap, (ofmap*self.wb)/(1e6), param, (param*self.wb)/(1e6), macs/1e9))
                csv_writer.writerow([layer_num, k, ifmap, ofmap, param, macs])

                if self.layers[k]['operation'] == "Conv": #or self.layers[k]['operation'] == "Gemm":
                    ifmap_size = self.layers[k]['input'][0]
                    kernel_size = self.layers[k]['kernel']
                    bias_size = self.layers[k]['bias']

                    ifmap_size = np.prod(np.array(ifmap_size))
                    kernel_size = np.prod(np.array(kernel_size)) if not len(kernel_size) == 0 else 0
                    bias_size = np.prod(np.array(bias_size)) if not len(bias_size) == 0 else 0

                    ifmap_mem_footprint = ((ifmap_size * self.wl) / 8) / 1e6
                    kernel_mem_footprint = ((kernel_size * self.wl) / 8) / 1e6 if not kernel_size == 0 else 0
                    bias_mem_footprint = ((bias_size * self.wl) / 8) / 1e6 if not bias_size == 0 else 0

                    txt_line = "{:.5f},{:.5f}\n".format(ifmap_mem_footprint,kernel_mem_footprint+bias_mem_footprint)
                    file.write(txt_line)
        file.close()

    def batchnorm_layer_config(self, in_shape, s_in=1, s_out=1):
        cin = in_shape[1]

        rate_in = 1 * s_in
        rate_out = 1 * s_in
        mem = cin * 4
        muls = max(3, math.ceil(3 * s_in))
        adds = max(3, math.ceil(3 * s_in))

        return rate_in, rate_out, muls, adds, mem
    
    def relu_layer_config(self, s_in=1, s_out=1):
        rate_in = 1 * s_in
        rate_out = 1 * s_in
        mem = 0
        muls = 0
        adds = 0

        return rate_in, rate_out, muls, adds, mem

    def sigmoid_layer_config(self, s_in=1, s_out=1):
        rate_in = 1 * s_in
        rate_out = 1 * s_in
        muls = max(3, math.ceil(3 * s_in))
        adds = max(1, math.ceil(1 * s_in))
        mem = 0

        return rate_in, rate_out, muls, adds, mem

    #TODO: Revise this layer
    def gap_layer_config(self, in_shape, coarse=None, s_in=1, s_out=1):
        cin = in_shape[1]
        din = in_shape[2]
        hin = in_shape[3]
        win = in_shape[4]

        if coarse is not None:
            gap_folding = coarse
        else:
            gap_folding = s_in

        rate_in = 1 * gap_folding
        rate_out = (1 * gap_folding)/(din * hin * win)
        mem = cin
        muls = max(2, math.ceil(2 * gap_folding))
        adds = max(1, math.ceil(1 * gap_folding))
        depth = din * hin * win

        mem_bounded_in = False
        if s_in < rate_in:
            in_module_ratio = rate_out / rate_in
            rate_in = s_in
            rate_out = rate_in * in_module_ratio
            assert math.isclose(in_module_ratio, rate_out / rate_in), "wrong calculation of ratio"
            mem_bounded_in = True 

        mem_bounded_out = False
        if s_out < rate_out:
            mem_bounded_out = True
            rate_out = s_out

        workload_in = cin*din*hin*win
        workload_out = cin
        latency = max(workload_in/rate_in, workload_out/rate_out)

        return rate_in, rate_out, muls, adds, mem, depth, latency, (mem_bounded_in, mem_bounded_out)

    def swish_layer_config(self, s_in=1, s_out=1):
        rate_in = 1 * s_in
        rate_out = 1 * s_in
        mem = 0
        muls = max(4, math.ceil(4 * s_in))
        adds = 0

        return rate_in, rate_out, muls, adds, mem

    def add_layer_config(self, s_in=1, s_out=1):
        rate_in = 1 * s_in
        rate_out = 1 * s_in
        mem = 0
        muls = 0
        adds = 1 * s_in

        return rate_in, rate_out, muls, adds, mem

    def mul_layer_config(self, s_in=1, s_out=1):
        rate_in = 1 * s_in
        rate_out = 1 * s_in
        mem = 0
        muls = 1 * s_in
        adds = 0

        return rate_in, rate_out, muls, adds, mem

    def fc_layer_config(self, in_shape, out_shape, fine, coarse_in, coarse_out, s_in=1, s_out=1):
        in_tensor_size = in_shape[1]
        out_tensor_size = out_shape[1]

        rate_in = 1 * coarse_in * coarse_out * fine
        rate_out = (1 * coarse_out * fine)/(in_tensor_size/coarse_in)
        mem = in_tensor_size * coarse_out
        if coarse_in < in_tensor_size:
            mem += coarse_in * coarse_out
        muls = max(coarse_in * coarse_out * fine, 1)
        adds = max(coarse_in * coarse_out * fine - 1, 1)

        mem_bounded_in = False
        if s_in < rate_in:
            in_module_ratio = rate_out / rate_in
            rate_in = s_in
            rate_out = rate_in * in_module_ratio
            assert math.isclose(in_module_ratio, rate_out / rate_in), "wrong calculation of ratio"
            mem_bounded_in = True 

        mem_bounded_out = False
        if s_out < rate_out:
            mem_bounded_out = True
            rate_out = s_out
        
        depth = coarse_in
        workload_in = in_tensor_size*out_tensor_size #we stream the weights as input
        workload_out = out_tensor_size
        latency = max(workload_in/rate_in, workload_out/rate_out)
        
        return rate_in, rate_out, muls, adds, mem, depth, latency, (mem_bounded_in, mem_bounded_out)

    def conv_layer_config(self, in_shape, out_shape, kernel_shape, padding, groups, fine, coarse_in, coarse_out, s_in=1, s_out=1):

        mem_bw_in = s_in
        mem_bw_out = s_out

        din = in_shape[2]
        hin = in_shape[3]
        win = in_shape[4]

        dout = out_shape[2]
        hout = out_shape[3]
        wout = out_shape[4]

        kd = kernel_shape[2]
        kh = kernel_shape[3]
        kw = kernel_shape[4]

        cin = in_shape[1]
        cout = out_shape[1]

        muls_unrl = kd * kh * kw * fine
        adds_unrl_1 = (kd * kh * kw - 1 ) * fine
        adds_unrl_2 = 1

        depthwise = False
        if cin == groups:
            depthwise = True

        if not depthwise:
            rates_graph = np.zeros( shape=(4,5) , dtype=float )
        else:
            rates_graph = np.zeros( shape=(3,4) , dtype=float )

        # The convolution operation is a Layer and is composed of the following modules: Sliding window, Conv, Accumulator 
        # Rates for the SW module
        rin_sw = 1
        rout_sw = (dout*hout*wout)/(din*hin*win)
        rates_graph[0,0] = rin_sw * coarse_in
        rates_graph[0,1] = rout_sw * (kd * kh * kw) * coarse_in

        rates_graph[1,1] = 1 * rates_graph[0,1]
        rates_graph[1,2] = 1 * rates_graph[0,1]

        # Rates for the Conv module
        rin_conv = (fine * groups * rates_graph[1,2] * coarse_out)/cout
        rout_conv = fine * rates_graph[1,2] * coarse_out
        rates_graph[2,2] = rin_conv
        rates_graph[2,3] = rout_conv / (kd * kh * kw)
        
        if not depthwise:
            # Rates for the Accumulator module
            rin_accum = 1 * rates_graph[2,3]
            rout_accum = (1 * groups * rates_graph[2,3])/cin
            rates_graph[3,3] = rin_accum
            rates_graph[3,4] = rout_accum

        # print("CONV RATE GRAPH")
        # print(rates_graph)
        # print("-"*50)
        rates_graph = self.balance_module_rates(rates_graph)
        # print(rates_graph)
        # print("=="*50)     

        mem_bounded_in = False
        if mem_bw_in < rates_graph[0,0]:
            in_module_ratio = rates_graph[0,1] / rates_graph[0,0]
            rates_graph[0,0] = mem_bw_in
            rates_graph[0,1] = rates_graph[0,0] * in_module_ratio
            assert math.isclose(in_module_ratio, rates_graph[0,1] / rates_graph[0,0]), "wrong calculation of ratio" 
            rates_graph = self.balance_module_rates(rates_graph)
            mem_bounded_in = True

        rate_in = abs(rates_graph[0,0])
        if not depthwise:
            rate_out = abs(rates_graph[3,4])
        else:
            rate_out = abs(rates_graph[2,3])

        mem_bounded_out = False
        if mem_bw_out < rate_out:
            mem_bounded_out = True
            rate_out = mem_bw_out

        if kd == 1 and kh == 1 and kw == 1:
            pb = 1
            sw_depth = min(((din*win+padding[0]+padding[1]+padding[2])*(cin/coarse_in)*kh)+(cin/coarse_in)*kh*kh, ((win*hin+padding[0]+padding[1]+padding[2])*(cin/coarse_in)*kd)+(cin/coarse_in)*kd*kd)
        else:
            # Plane buffer + Line buffer (needed in conjuction with plane buffer)
            pb = min((din*win*kh), (win*hin*kd)) + min((din*kw), (win*kh))
            sw_depth = min(((din*win+padding[0]+padding[1]+padding[2])*(cin/coarse_in)*(max(kh-1,1)))+(cin/coarse_in)*kh*(max(kh-1,1)), ((win*hin+padding[0]+padding[1]+padding[2])*(cin/coarse_in)*(max(kd-1,1)))+(cin/coarse_in)*kd*(max(kd-1,1)))
        kernel_size = int(np.prod(np.array(kernel_shape)))

        conv_depth = math.ceil(1/fine)
        acc_depth = (cin/coarse_in)*(cout/coarse_out)
        if not depthwise:
            depth = sw_depth + conv_depth + acc_depth
        else:
            depth = sw_depth + conv_depth
        mem = pb + kernel_size + cin

        muls = math.ceil(muls_unrl * coarse_in * coarse_out)
        #TODO: This calculations are not correct. Need revision.
        adds = math.ceil(adds_unrl_1 * coarse_in * coarse_out) + math.ceil(adds_unrl_2 * coarse_in * coarse_out)

        #TODO:Revise the case with channels and input shape onconvolutions with groups
        workload_in = cin*din*hin*win
        workload_out = cout*dout*hout*wout
        latency = max(workload_in/rate_in, workload_out/rate_out)

        return rate_in, rate_out, muls, adds, mem, depth, latency, (mem_bounded_in, mem_bounded_out)

    def se_layer_config(self, glavpool_in_shape, glavpool_coarse, conv1_in_shape, conv1_out_shape, conv1_kernel_shape, conv1_padding, conv1_groups, fine1, coarse_in_1, coarse_out_1, conv2_in_shape, conv2_out_shape, conv2_kernel_shape, conv2_padding, conv2_groups, fine2, coarse_in_2, coarse_out_2, bw_in=1, bw_total=1, se_on_bram=1):
        
        mem_bw_in = bw_in
        mem_bw_total = bw_total

        glavpool_rate_in, glavpool_rate_out, glavpool_muls, glavpool_adds, glavpool_mem, glavpool_depth, glavpool_latency, (glavpool_mem_bounded_in, glavpool_mem_bounded_out) = self.gap_layer_config(glavpool_in_shape, coarse=glavpool_coarse, s_in=mem_bw_in, s_out=10000)
        glavpool_thr_in = (self.cycles_per_sec*glavpool_rate_in)/int(np.prod(np.array(glavpool_in_shape[1:])))
        glavpool_thr_out = (self.cycles_per_sec*glavpool_rate_out)/int(glavpool_in_shape[1])
        assert math.isclose(glavpool_thr_in, glavpool_thr_out), "Input and Output Throughputs doesnt match on glavpool. Aborting..."

        conv1_rate_in, conv1_rate_out, conv1_muls, conv1_adds, conv1_mem, conv1_depth, conv1_latency, (conv1_mem_bounded_in, conv1_mem_bounded_out) = self.conv_layer_config(conv1_in_shape, conv1_out_shape, conv1_kernel_shape, conv1_padding, conv1_groups, fine1, coarse_in_1, coarse_out_1, s_in=glavpool_rate_out, s_out=10000)
        conv1_thr_in = (self.cycles_per_sec*conv1_rate_in)/int(np.prod(np.array(conv1_in_shape[1:])))
        conv1_thr_out = (self.cycles_per_sec*conv1_rate_out)/int(np.prod(np.array(conv1_out_shape[1:])))
        assert math.isclose(conv1_thr_in, conv1_thr_out), "Input and Output Throughputs doesnt match on conv1. Aborting..."

        relu_rate_in, relu_rate_out, _, _, _ = self.relu_layer_config(s_in=conv1_rate_out)

        conv2_rate_in, conv2_rate_out, conv2_muls, conv2_adds, conv2_mem, conv2_depth, conv2_latency, (conv2_mem_bounded_in, conv2_mem_bounded_out) = self.conv_layer_config(conv2_in_shape, conv2_out_shape, conv2_kernel_shape, conv2_padding, conv2_groups, fine2, coarse_in_2, coarse_out_2, s_in=relu_rate_out, s_out=10000)
        conv2_thr_in = (self.cycles_per_sec*conv2_rate_in)/int(np.prod(np.array(conv2_in_shape[1:])))
        conv2_thr_out = (self.cycles_per_sec*conv2_rate_out)/int(np.prod(np.array(conv2_out_shape[1:])))
        assert math.isclose(conv2_thr_in, conv2_thr_out), "Input and Output Throughputs doesnt match on conv2. Aborting..."

        sigmoid_rate_in, sigmoid_rate_out, sigmoid_muls, _, _ = self.sigmoid_layer_config(s_in=conv2_rate_out)
        
        rates_graph = np.zeros( shape=(5,6) , dtype=float )
        rates_graph[0,0] = glavpool_rate_in
        rates_graph[0,1] = glavpool_rate_out

        rates_graph[1,1] = conv1_rate_in
        rates_graph[1,2] = conv1_rate_out

        rates_graph[2,2] = relu_rate_in
        rates_graph[2,3] = relu_rate_out

        rates_graph[3,3] = conv2_rate_in
        rates_graph[3,4] = conv2_rate_out

        rates_graph[4,4] = sigmoid_rate_in
        rates_graph[4,5] = sigmoid_rate_out

        # print("SE RATE GRAPH")
        # print(rates_graph)
        # print("-"*50)
        rates_graph = self.balance_module_rates(rates_graph)
        # print(rates_graph)
        # print("=="*50)

        mem_bounded_in = False
        if mem_bw_in < rates_graph[0,0]:
            in_module_ratio = rates_graph[0,1] / rates_graph[0,0]
            rates_graph[0,0] = mem_bw_in
            rates_graph[0,1] = rates_graph[0,0] * in_module_ratio
            assert math.isclose(in_module_ratio, rates_graph[0,1] / rates_graph[0,0]), "wrong calculation of ratio" 
            rates_graph = self.balance_module_rates(rates_graph)
            mem_bounded_in = True
        if glavpool_mem_bounded_in:
            mem_bounded_in = True
            
        rate_in = abs(rates_graph[0,0])
        rate_out = abs(rates_graph[4,5])
        rate_broadcasting = rate_out * int(np.prod(np.array(glavpool_in_shape[2:])))
        se_thr_in = (self.cycles_per_sec*rate_in)/int(np.prod(np.array(glavpool_in_shape[1:])))
        se_thr_out = (self.cycles_per_sec*rate_out)/int(np.prod(np.array(conv2_out_shape[1:])))
        assert math.isclose(se_thr_in, se_thr_out), "Input and Output Throughputs doesnt match on se branch. Aborting..."

        mem_bw_left = mem_bw_total - rate_in
        if se_on_bram == 1:
            best_mem_bw_out = mem_bw_left
            best_mem_bw_branch = rate_broadcasting
        else:
            mem_bw_left -= rate_broadcasting
            if mem_bw_left > 0:
                best_mem_bw_out = mem_bw_left
                best_mem_bw_branch = rate_broadcasting
            else:
                mem_bw_left += rate_broadcasting
                best_mem_bw_out = mem_bw_left/2
                best_mem_bw_branch = mem_bw_left/2
                mem_bounded_in = True
        
        assert math.isclose(min(glavpool_thr_out, conv1_thr_out, conv2_thr_out), se_thr_out), "Invalid Throughput Balancing. Aborting..."

        elemwise_mul_rate_in, elemwise_mul_rate_out, elemwise_mul_rate_muls, _, _ = self.mul_layer_config(s_in=best_mem_bw_branch)
        
        rate_in = elemwise_mul_rate_in
        rate_out = elemwise_mul_rate_out

        assert mem_bw_in >= rate_in, "Memory bounded on SE input"
        mem_bounded_out = False
        if best_mem_bw_out < rate_out:
            mem_bounded_out = True
            rate_out = best_mem_bw_out

        muls = glavpool_muls + conv1_muls + conv2_muls + sigmoid_muls + elemwise_mul_rate_muls
        adds = glavpool_adds + conv1_adds + conv2_adds
        mem = glavpool_mem + conv1_mem + conv2_mem
        depth = glavpool_depth + conv1_depth + 1 + conv2_depth + 1

        if se_on_bram == 1:
            mem += depth
    
        workload_in = int(np.prod(np.array(glavpool_in_shape)))
        workload_out = int(np.prod(np.array(glavpool_in_shape)))
        latency = max(workload_in/rate_in, workload_out/rate_out)
        return rate_in, rate_out, muls, adds, mem, depth, latency, (mem_bounded_in, mem_bounded_out)

    def get_layer_from_id(self, layer_id):
        for k in self.layers.keys():
            if isinstance(layer_id, int):
                if layer_id == int(self.layers[k]['output_id']):
                    return self.layers[k], k
            else:
                if layer_id == self.layers[k]['output_id']:
                    return self.layers[k], k
        return None, None
    
    @timebudget
    def create_modules(self):
        se_module = deque(maxlen=6)
        swish_module = deque(maxlen=3)
        prev_output_id = -1
        for k in self.layers.keys():
            curr_output_id = int(self.layers[k]['output_id'])
            if not prev_output_id == -1:
                assert curr_output_id >= prev_output_id + 1, "Modules are not in the correct order. Revise the graph creation"
            prev_output_id = curr_output_id
        
            name = k
            operation = self.layers[k]['operation']
            input_shape = self.layers[k]['input'][0]
            output_shape = self.layers[k]['output']
            
            branching = False
            if len(self.layers[k]['input']) > 1:
                in1_layer, in1_layer_name = self.get_layer_from_id(int(self.layers[k]['input_id'][0]))
                in2_layer, in2_layer_name = self.get_layer_from_id(int(self.layers[k]['input_id'][1]))
                oldest_input = min(int(in1_layer['output_id']), int(in2_layer['output_id'])) if (in1_layer is not None and in2_layer is not None) else int(self.layers[k]['output_id'])
                logging.info("Layer name = {} ({}). Input_0 = {} ({}). Input_1 = {} ({})".format(name, self.layers[k]['output_id'], in1_layer_name, self.layers[k]['input_id'][0], in2_layer_name, self.layers[k]['input_id'][1]))
            else:
                if isinstance(self.layers[k]['input_id'][0], int):
                    in_id = int(self.layers[k]['input_id'][0])
                else:
                    in_id = self.layers[k]['input_id'][0]
                in1_layer, in1_layer_name = self.get_layer_from_id(in_id)
                oldest_input = int(in1_layer['output_id']) if in1_layer is not None else int(self.layers[k]['output_id'])
                logging.info("Layer name = {} ({}). Input = {} ({})".format(name, self.layers[k]['output_id'], in1_layer_name, self.layers[k]['input_id'][0]))
            #TODO: Should find a more generic way to detect and filter branching behaviours on networks.
            if int(self.layers[k]['output_id']) - oldest_input > 2 and not (operation == 'MatMul' and self.model_name == 'x3d_m'):
                branching = True
                logging.info("Identified a branching behaviour on layer {}".format(name))

            if operation == 'Mul' or operation == 'Add':
                input_shape = output_shape
            kernel = self.layers[k]['kernel']
            bias = self.layers[k]['bias']
            groups = self.layers[k]['groups']
            padding = self.layers[k]['padding']

            self.modules[name] = {"operation": operation,
                                  "shape_in": input_shape,
                                  "shape_out": output_shape,
                                  "kernel": kernel,
                                  "bias": bias,
                                  "groups": groups,
                                  "padding": padding,
                                  "branching": branching}

            swish_module.append([operation, name])
            if swish_module[0][0] == 'Sigmoid' and swish_module[1][0] == 'Mul' and swish_module[2][0] == 'Conv':
                logging.debug("Creating Swish Activation Module")

                sigmoid_name = swish_module[0][1]
                operation = self.layers[sigmoid_name]['operation']
                input_shape = self.layers[sigmoid_name]['input'][0]
                swish_input_shape = input_shape
                output_shape = self.layers[sigmoid_name]['output']
                if operation == 'Mul' or operation == 'Add':
                    input_shape = output_shape
                kernel = self.layers[sigmoid_name]['kernel']
                bias = self.layers[sigmoid_name]['bias']
                groups = self.layers[sigmoid_name]['groups']
                padding = self.layers[sigmoid_name]['padding']
                sigmoid = {"operation": operation,
                       "shape_in": input_shape,
                       "shape_out": output_shape,
                       "kernel": kernel,
                       "bias": bias,
                       "groups": groups,
                       "padding": padding}

                mul_name = swish_module[1][1]
                operation = self.layers[mul_name]['operation']
                input_shape = self.layers[mul_name]['input'][0]
                output_shape = self.layers[mul_name]['output']
                swish_output_shape = output_shape
                if operation == 'Mul' or operation == 'Add':
                    input_shape = output_shape
                kernel = self.layers[mul_name]['kernel']
                bias = self.layers[mul_name]['bias']
                groups = self.layers[mul_name]['groups']
                padding = self.layers[mul_name]['padding']
                mul = {"operation": operation,
                       "shape_in": input_shape,
                       "shape_out": output_shape,
                       "kernel": kernel,
                       "bias": bias,
                       "groups": groups,
                       "padding": padding}

                name = 'Swish_' + swish_module[0][1].split('_')[1]
                operation = 'Swish'
                self.modules[name] = {"operation": operation,
                                      "shape_in": swish_input_shape,
                                      "shape_out": swish_output_shape,
                                      sigmoid_name: sigmoid,
                                      mul_name: mul,
                                      "branching": False}

                del self.modules[sigmoid_name]
                del self.modules[mul_name]

                conv_name = swish_module[2][1]
                del self.modules[conv_name]

                operation = self.layers[conv_name]['operation']
                input_shape = self.layers[conv_name]['input'][0]
                output_shape = self.layers[conv_name]['output']
                if operation == 'Mul' or operation == 'Add':
                    input_shape = output_shape
                kernel = self.layers[conv_name]['kernel']
                bias = self.layers[conv_name]['bias']
                groups = self.layers[conv_name]['groups']
                padding = self.layers[conv_name]['padding']
                self.modules[conv_name] = {"operation": operation,
                                    "shape_in": input_shape,
                                    "shape_out": output_shape,
                                    "kernel": kernel,
                                    "bias": bias,
                                    "groups": groups,
                                    "padding": padding,
                                    "branching": False}

            se_module.append([operation, name])
            if se_module[0][0] == 'GlobalAveragePool' and se_module[1][0] == 'Conv' and se_module[2][0] == 'Relu' and se_module[3][0] == 'Conv' and se_module[4][0] == 'Sigmoid' and se_module[5][0] == 'Mul':
                logging.debug("Creating Squeeze and Excitation Module")

                gap_name = se_module[0][1]
                operation = self.layers[gap_name]['operation']
                input_shape = self.layers[gap_name]['input'][0]
                se_input_shape = input_shape
                output_shape = self.layers[gap_name]['output']
                if operation == 'Mul' or operation == 'Add':
                    input_shape = output_shape
                kernel = self.layers[gap_name]['kernel']
                bias = self.layers[gap_name]['bias']
                groups = self.layers[gap_name]['groups']
                padding = self.layers[gap_name]['padding']
                gap = {"operation": operation,
                       "shape_in": input_shape,
                       "shape_out": output_shape,
                       "kernel": kernel,
                       "bias": bias,
                       "groups": groups,
                       "padding": padding}

                conv1_name = se_module[1][1]
                operation = self.layers[conv1_name]['operation']
                input_shape = self.layers[conv1_name]['input'][0]
                output_shape = self.layers[conv1_name]['output']
                if operation == 'Mul' or operation == 'Add':
                    input_shape = output_shape
                kernel = self.layers[conv1_name]['kernel']
                bias = self.layers[conv1_name]['bias']
                groups = self.layers[conv1_name]['groups']
                padding = self.layers[conv1_name]['padding']
                conv1 = {"operation": operation,
                       "shape_in": input_shape,
                       "shape_out": output_shape,
                       "kernel": kernel,
                       "bias": bias,
                       "groups": groups,
                       "padding": padding}

                relu_name = se_module[2][1]
                operation = self.layers[relu_name]['operation']
                input_shape = self.layers[relu_name]['input'][0]
                output_shape = self.layers[relu_name]['output']
                if operation == 'Mul' or operation == 'Add':
                    input_shape = output_shape
                kernel = self.layers[relu_name]['kernel']
                bias = self.layers[relu_name]['bias']
                groups = self.layers[relu_name]['groups']
                padding = self.layers[relu_name]['padding']
                relu = {"operation": operation,
                       "shape_in": input_shape,
                       "shape_out": output_shape,
                       "kernel": kernel,
                       "bias": bias,
                       "groups": groups,
                       "padding": padding}

                conv2_name = se_module[3][1]
                operation = self.layers[conv2_name]['operation']
                input_shape = self.layers[conv2_name]['input'][0]
                output_shape = self.layers[conv2_name]['output']
                if operation == 'Mul' or operation == 'Add':
                    input_shape = output_shape
                kernel = self.layers[conv2_name]['kernel']
                bias = self.layers[conv2_name]['bias']
                groups = self.layers[conv2_name]['groups']
                padding = self.layers[conv2_name]['padding']
                conv2 = {"operation": operation,
                       "shape_in": input_shape,
                       "shape_out": output_shape,
                       "kernel": kernel,
                       "bias": bias,
                       "groups": groups,
                       "padding": padding}

                sigmoid_name = se_module[4][1]
                operation = self.layers[sigmoid_name]['operation']
                input_shape = self.layers[sigmoid_name]['input'][0]
                output_shape = self.layers[sigmoid_name]['output']
                se_branch_shape = output_shape
                if operation == 'Mul' or operation == 'Add':
                    input_shape = output_shape
                kernel = self.layers[sigmoid_name]['kernel']
                bias = self.layers[sigmoid_name]['bias']
                groups = self.layers[sigmoid_name]['groups']
                padding = self.layers[sigmoid_name]['padding']
                sigmoid = {"operation": operation,
                       "shape_in": input_shape,
                       "shape_out": output_shape,
                       "kernel": kernel,
                       "bias": bias,
                       "groups": groups,
                       "padding": padding}

                mul_name = se_module[5][1]
                operation = self.layers[mul_name]['operation']
                input_shape = self.layers[mul_name]['input'][0]
                output_shape = self.layers[mul_name]['output']
                se_output_shape = output_shape
                if operation == 'Mul' or operation == 'Add':
                    input_shape = output_shape
                kernel = self.layers[mul_name]['kernel']
                bias = self.layers[mul_name]['bias']
                groups = self.layers[mul_name]['groups']
                padding = self.layers[mul_name]['padding']
                mul = {"operation": operation,
                       "shape_in": input_shape,
                       "shape_out": output_shape,
                       "kernel": kernel,
                       "bias": bias,
                       "groups": groups,
                       "padding": padding}

                name = 'Se_' + se_module[0][1].split('_')[1]
                operation = 'SqueezeExcitation'
                self.modules[name] = {"operation": operation,
                                      "shape_in": se_input_shape,
                                      "shape_out": se_output_shape,
                                      "shape_branch": se_branch_shape,
                                      gap_name: gap,
                                      conv1_name: conv1,
                                      relu_name: relu,
                                      conv2_name: conv2,
                                      sigmoid_name: sigmoid,
                                      mul_name: mul,
                                      "branching": True}

                del self.modules[gap_name]
                del self.modules[conv1_name]
                del self.modules[relu_name]
                del self.modules[conv2_name]
                del self.modules[sigmoid_name]
                del self.modules[mul_name]

    def model_layer(self, layer, csv_writer, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem, depth, latency, mem_bounded_out=False, config=None, inter_size=None, buffering_enabled=False, module_mem_bw_in=-1):
        l_config = config if config is not None else "No configuration available"
        layer = layer
        folding = folding_name
        mem_kb = (mem*self.wb)/1e3
        mem_bram = math.ceil(mem_kb/(self.bram_mem/8))
        mem_util = (mem_bram/self.fpga_bram)*100

        bw_in_w = rate_in
        bw_out_w = rate_out

        thr_w_in = rate_in
        thr_w_out = rate_out

        muls = muls
        adds = adds
        dsps = muls

        #TODO: Revise these calculations. Seem to be wrong.
        bw_in_gb = (self.cycles_per_sec*bw_in_w*self.wb)/1e9	
        bw_out_gb = (self.cycles_per_sec*bw_out_w*self.wb)/1e9

        thr_in = (self.cycles_per_sec*thr_w_in)/in_size
        thr_out = (self.cycles_per_sec*thr_w_out)/out_size

        thr_go = ((muls + adds)*self.cycles_per_sec)/1e9
        dsps_util = (dsps/self.fpga_dsps)*100

        assert math.isclose(thr_out, thr_in) or mem_bounded_out, "Input and Output Throughput doesnt match. Aborting..."

        if dsps_util < 90.0 and mem_util < 90.0:
            logging.info("On Chip Mem (BRAM %) = {:<15.5f} DSPS % = {:<20.5f}\nConsumption(inputs/sec) = {:<20.5f} Throughtput(outputs/sec) = {:<20.5f}\nMemory Bandwidth In(words/cycle) = {:<20.5f} Memory Bandwidth Out(words/cycle) = {:<20.5f}\nOn-Chip Memory(KB) = {:<20.5f} On-Chip Memory(BRAM) = {:<20.5f} Adds = {:<20.5f} DSPS = {:<20.5f}\nMemory Bandwidth In(GBs/sec) = {:<20.5f} Memory Bandwidth Out(GBs/sec) = {:<20.5f}\nThroughtput(GOps/sec) = {:.3f}".format(mem_util, dsps_util, thr_in, thr_out, bw_in_w, bw_out_w, mem_kb, mem_bram, adds, dsps, bw_in_gb, bw_out_gb, thr_go))   
            # csv_writer.writerow([layer, folding, mem_util, dsps_util, thr_in, thr_out, bw_in_w, bw_out_w, mem_kb, mem_bram, bw_in_gb, bw_out_gb, muls, adds, dsps, depth, latency, thr_w_out, thr_go, l_config])
            return [layer, folding, mem_util, dsps_util, thr_in, thr_out, bw_in_w, bw_out_w, mem_kb, mem_bram, bw_in_gb, bw_out_gb, muls, adds, dsps, depth, latency, thr_w_out, thr_go, l_config]
        else:
            logging.info("Design point dropped because of too many recourses needed. DSPS = {} ({}%). BRAM = {} ({}%)".format(dsps, dsps_util, mem_bram, mem_util))
            return None
    
    @ray.remote
    @timebudget
    def design_points_per_layer(self, operation, name, s_in=1, s_out=1):
        # if not (operation == 'Conv'):
        #     continue
        layer_results = []
        if operation == 'Conv':
            out_shape = self.modules[name]['shape_out']
            dout = out_shape[2]
            hout = out_shape[3]
            wout = out_shape[4]

            in_shape = self.modules[name]['shape_in']
            din = in_shape[2]
            hin = in_shape[3]
            win = in_shape[4]
            
            kernel_shape = self.modules[name]['kernel']
            kd = kernel_shape[2]
            kh = kernel_shape[3]
            kw = kernel_shape[4]

            groups = self.modules[name]['groups']
            padding = self.modules[name]['padding']

            cin = in_shape[1]
            cout = out_shape[1]

            #TODO: Check the case with groups and input channels on depthwise conv
            in_size = cin * din * hin * win
            out_size = cout * dout * hout * wout

            pr_name = name

            depthwise = False
            if cout == groups:
                depthwise = True
            #     pr_name = pr_name + "_DepthWise"
            # if kd == 1 and kh == 1 and kw == 1:
            #     pr_name = pr_name + "_PointWise"

            coarse_in_config = [1, (cin)//4, (cin)//2, cin]
            coarse_in_config = np.unique(coarse_in_config)
            coarse_in_config = coarse_in_config[np.nonzero(coarse_in_config)].tolist()
            coarse_out_config = [1, cout//4, cout//2, cout]
            coarse_out_config = np.unique(coarse_out_config)
            coarse_out_config = coarse_out_config[np.nonzero(coarse_out_config)].tolist()
            if depthwise:
                coarse_out_config = [1]
            max_fine = kd * kh * kw
            fine_config = np.array([kd/max_fine, kh/max_fine, kw/max_fine, (kd * kh)/max_fine, (kh * kw)/max_fine, (kd * kw)/max_fine, 1])
            fine_config = np.unique(fine_config).tolist()
            if kd == 1 and kh == 1 and kw == 1:
                fine_config = [0.5, 1]
            
            mem_bw = s_in + s_out
            mem_bw_config = [(mem_bw*0.2, mem_bw*0.8), (mem_bw*0.4, mem_bw*0.6), (mem_bw*0.5, mem_bw*0.5), (mem_bw*0.6, mem_bw*0.4), (mem_bw*0.8, mem_bw*0.2), (mem_bw*0.9, mem_bw*0.1)]
            for coarse_in in coarse_in_config:
                for coarse_out in coarse_out_config:
                    for fine in fine_config:
                        for conv_bw_in, conv_bw_out in mem_bw_config:
                            coarse_in_name = str(coarse_in)
                            coarse_out_name = str(coarse_out)
                            folding_name = "N_Coarse({}/{}) - f_Fine({:.2f}) - Mem BW({:.2f}/{:.2f})".format(coarse_in_name, coarse_out_name, fine, conv_bw_in, conv_bw_out)

                            logging.info("Fold = {}. Channels In = {} - Filters = {}".format(folding_name, cin, cout))

                            rate_in, rate_out, muls, adds, mem, depth, latency, (mem_bounded_in, mem_bounded_out) = self.conv_layer_config(in_shape, out_shape, kernel_shape, padding, groups, fine, coarse_in, coarse_out, s_in=conv_bw_in, s_out=conv_bw_out)

                            current_config = [in_shape, out_shape, kernel_shape, padding, groups, fine, coarse_in, coarse_out, int(mem_bounded_in), int(mem_bounded_out)]
                            curr_result = self.model_layer(pr_name, None, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem, depth, latency, mem_bounded_out=mem_bounded_out, config=current_config)
                            if curr_result is not None:
                                layer_results.append(curr_result)

        elif operation == 'SqueezeExcitation':
            out_shape = self.modules[name]['shape_out']
            cout = out_shape[1]
            dout = out_shape[2]
            hout = out_shape[3]
            wout = out_shape[4]
            out_size = cout * dout * hout * wout

            in_shape = self.modules[name]['shape_in']
            cin = in_shape[1]
            din = in_shape[2]
            hin = in_shape[3]
            win = in_shape[4]
            in_size = cin * din * hin * win

            branch_shape = self.modules[name]['shape_branch']
            cbr = branch_shape[1]
            dbr = branch_shape[2]
            hbr = branch_shape[3]
            wbr = branch_shape[4]
            br_size = cbr * dbr * hbr * wbr

            se_keys = list(self.modules[name].keys())
            glavpool_key = se_keys[4]
            glavpool_in_shape = self.modules[name][glavpool_key]['shape_in']
            coarse_gap_config = [1, glavpool_in_shape[1]//8, glavpool_in_shape[1]//4, glavpool_in_shape[1]//2, glavpool_in_shape[1]]
            coarse_gap_config = np.unique(coarse_gap_config)
            coarse_gap_config = coarse_gap_config[np.nonzero(coarse_gap_config)].tolist()

            conv1_key = se_keys[5]
            conv1_in_shape = self.modules[name][conv1_key]['shape_in']
            conv1_out_shape = self.modules[name][conv1_key]['shape_out']
            conv1_kernel_shape = self.modules[name][conv1_key]['kernel']
            conv1_groups = self.modules[name][conv1_key]['groups']
            conv1_padding = self.modules[name][conv1_key]['padding']

            conv2_key = se_keys[7]
            conv2_in_shape = self.modules[name][conv2_key]['shape_in']
            conv2_out_shape = self.modules[name][conv2_key]['shape_out']
            conv2_kernel_shape = self.modules[name][conv2_key]['kernel']
            conv2_groups = self.modules[name][conv2_key]['groups']
            conv2_padding = self.modules[name][conv2_key]['padding']

            coarse_in_conv1 = conv1_kernel_shape[1]
            coarse_out_conv1 = conv1_kernel_shape[0]

            coarse_in_conv2 = conv2_kernel_shape[1]
            coarse_out_conv2 = conv2_kernel_shape[0]

            coarse_in_config_conv1 = [1, coarse_in_conv1//4, coarse_in_conv1//2, coarse_in_conv1]
            coarse_in_config_conv1 = np.unique(coarse_in_config_conv1)
            coarse_in_config_conv1 = coarse_in_config_conv1[np.nonzero(coarse_in_config_conv1)].tolist()

            coarse_in_config_conv2 = [1, coarse_in_conv2//4, coarse_in_conv2//2, coarse_in_conv2]
            coarse_in_config_conv2 = np.unique(coarse_in_config_conv2)
            coarse_in_config_conv2 = coarse_in_config_conv2[np.nonzero(coarse_in_config_conv2)].tolist()

            coarse_out_config_conv1 = [1, coarse_out_conv1//4, coarse_out_conv1//2, coarse_out_conv1]
            coarse_out_config_conv1 = np.unique(coarse_out_config_conv1)
            coarse_out_config_conv1 = coarse_out_config_conv1[np.nonzero(coarse_out_config_conv1)].tolist()

            coarse_out_config_conv2 = [1, coarse_out_conv2//4, coarse_out_conv2//2, coarse_out_conv2]
            coarse_out_config_conv2 = np.unique(coarse_out_config_conv2)
            coarse_out_config_conv2 = coarse_out_config_conv2[np.nonzero(coarse_out_config_conv2)].tolist()
            
            kd_1 = conv1_kernel_shape[2]
            kh_1 = conv1_kernel_shape[3]
            kw_1 = conv1_kernel_shape[4]
            max_fine = kd_1 * kh_1 * kw_1
            fine_config_1 = np.array([kd_1/max_fine, kh_1/max_fine, kw_1/max_fine, (kd_1 * kh_1)/max_fine, (kh_1 * kw_1)/max_fine, (kd_1 * kw_1)/max_fine, 1])
            fine_config_1 = np.unique(fine_config_1).tolist()
            if kd_1 == 1 and kh_1 == 1 and kw_1 == 1:
                fine_config_1 = [0.5, 1]

            kd_2 = conv2_kernel_shape[2]
            kh_2 = conv2_kernel_shape[3]
            kw_2 = conv2_kernel_shape[4]
            max_fine = kd_2 * kh_2 * kw_2
            fine_config_2 = np.array([kd_2/max_fine, kh_2/max_fine, kw_2/max_fine, (kd_2 * kh_2)/max_fine, (kh_2 * kw_2)/max_fine, (kd_2 * kw_2)/max_fine, 1])
            fine_config_2 = np.unique(fine_config_2).tolist()
            if kd_2 == 1 and kh_2 == 1 and kw_2 == 1:
                fine_config_2 = [0.5, 1]

            mem_bw = s_in + s_out
            mem_bw_config = [mem_bw*0.2, mem_bw*0.4, mem_bw*0.6, mem_bw*0.8, mem_bw*0.9]
            for coarse_gap in coarse_gap_config:
                for coarse_in_1 in coarse_in_config_conv1:
                    for coarse_in_2 in coarse_in_config_conv2:
                        for coarse_out_1 in coarse_out_config_conv1:
                            for coarse_out_2 in coarse_out_config_conv2:
                                for fine_1 in fine_config_1:
                                    for fine_2 in fine_config_2:
                                        for se_bw_in in mem_bw_config:
                                            for se_on_bram in [0, 1]:

                                                folding_name = "N_Coarse_1({}/{}) - N_Coarse_2({}/{}) - f_Fine_1({:.2f}) - f_Fine_2({:.2f}) - Mem BW IN {:.2f}".format(coarse_in_1, coarse_out_1, coarse_in_2, coarse_out_2, fine_1, fine_2, se_bw_in)
                                                if se_on_bram:
                                                    folding_name += " - BRAM"
                                                logging.info("Fold = {}".format(folding_name))
                                                
                                                rate_in, rate_out, muls, adds, mem, depth, latency, (mem_bounded_in, mem_bounded_out) = self.se_layer_config(glavpool_in_shape, coarse_gap, conv1_in_shape, conv1_out_shape, conv1_kernel_shape, conv1_padding, conv1_groups, fine_1, coarse_in_1, coarse_out_1, conv2_in_shape, conv2_out_shape, conv2_kernel_shape, conv2_padding, conv2_groups, fine_2, coarse_in_2, coarse_out_2, bw_in=se_bw_in, bw_total=mem_bw, se_on_bram=se_on_bram)

                                                #TODO: Added worst possible case for buffering on se module i.e., buffer the whole feature map and all of the channels. Should fix this by checking the depth/latency of the left branch in order to calculate the exact buffering that is gonna needed in each se module.
                                                #TODO: Another solution is to read again from off-chip memory which will prevent the buffering i.e., reduce the BRAM needs BUT will reduce the mem bw in total as well since we need to first write the results (in a bigger layer-wise partition) and the read them again i.e., will probably need to have mem_bw / 4 instead of mem_bw / 2 in each point that we access the off-chip memory.

                                                current_config = [glavpool_in_shape, coarse_gap, conv1_in_shape, conv1_out_shape, conv1_kernel_shape, conv1_padding, conv1_groups, fine_1, coarse_in_1, coarse_out_1, conv2_in_shape, conv2_out_shape, conv2_kernel_shape, conv2_padding, conv2_groups, fine_2, coarse_in_2, coarse_out_2, se_on_bram, int(mem_bounded_in), int(mem_bounded_out)]

                                                curr_result = self.model_layer(name, None, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem, depth, latency, mem_bounded_out=mem_bounded_out, config=current_config, inter_size=br_size, buffering_enabled=se_on_bram, module_mem_bw_in=se_bw_in)
                                                if curr_result is not None:
                                                    layer_results.append(curr_result)

        elif operation == 'BatchNormalization':
            out_shape = self.modules[name]['shape_out']
            cout = out_shape[1]
            out_size = int(np.prod(np.array(out_shape[1:])))

            in_shape = self.modules[name]['shape_in']
            cin = in_shape[1]
            in_size = int(np.prod(np.array(in_shape[1:])))

            assert out_shape == in_shape, 'Input and output shapes bust be identical in BatchNormalization Layer'

            folding_name = "Mem_Bw({}/{})".format(s_in, s_out)

            logging.info("Fold = {}. Channels In = {} - Filters = {}".format(folding_name, cin, cout))

            rate_in, rate_out, muls, adds, mem = self.batchnorm_layer_config(in_shape, s_in=s_in, s_out=s_out)

            curr_result = self.model_layer(name, None, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem, 1, 1, config=[in_shape])
            if curr_result is not None:
                layer_results.append(curr_result) 

        elif operation == 'Relu':
            out_shape = self.modules[name]['shape_out']
            cout = out_shape[1]
            out_size = int(np.prod(np.array(out_shape[1:])))

            in_shape = self.modules[name]['shape_in']
            cin = in_shape[1]
            in_size = int(np.prod(np.array(in_shape[1:])))

            assert out_shape == in_shape, 'Input and output shapes bust be identical in Relu Layer'

            folding_name = "Mem_Bw({}/{})".format(s_in, s_out)

            logging.info("Fold = {}. Channels In = {} - Filters = {}".format(folding_name, cin, cout))

            rate_in, rate_out, muls, adds, mem = self.relu_layer_config(s_in=s_in, s_out=s_out)

            curr_result = self.model_layer(name, None, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem, 1, 1)
            if curr_result is not None:
                layer_results.append(curr_result) 

        elif operation == 'GlobalAveragePool':
            out_shape = self.modules[name]['shape_out']
            cout = out_shape[1]
            out_size = int(np.prod(np.array(out_shape[1:])))

            in_shape = self.modules[name]['shape_in']
            cin = in_shape[1]
            in_size = int(np.prod(np.array(in_shape[1:])))

            assert cin == cout, 'Input and output channels bust be identical in GlobalAveragePool Layer'

            gap_config = [1, (cin)//16, (cin)//12, (cin)//8, (cin)//4]

            for gap_coarse in gap_config:
                rate_in, rate_out, muls, adds, mem, depth, latency, (mem_bounded_in, mem_bounded_out) = self.gap_layer_config(in_shape, coarse=gap_coarse, s_in=s_in, s_out=s_out)

                folding_name = "Coarse({:.2f}) - Mem_Bw({}/{})".format(gap_coarse, s_in, s_out)
                logging.info("Fold = {}. Channels In = {} - Filters = {}".format(folding_name, cin, cout))

                curr_result = self.model_layer(name, None, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem, depth, latency, mem_bounded_out=mem_bounded_out, config=[in_shape, gap_coarse, int(mem_bounded_in), int(mem_bounded_out)])
                if curr_result is not None:
                    layer_results.append(curr_result) 

        elif operation == 'Swish':
            out_shape = self.modules[name]['shape_out']
            cout = out_shape[1]
            out_size = int(np.prod(np.array(out_shape[1:])))

            in_shape = self.modules[name]['shape_in']
            cin = in_shape[1]
            in_size = int(np.prod(np.array(in_shape[1:])))

            assert out_shape == in_shape, 'Input and output shapes bust be identical in Swish Layer'

            folding_name = "Mem_Bw({}/{})".format(s_in, s_out)

            logging.info("Fold = {}. Channels In = {} - Filters = {}".format(folding_name, cin, cout))

            rate_in, rate_out, muls, adds, mem = self.swish_layer_config(s_in=s_in, s_out=s_out)

            curr_result = self.model_layer(name, None, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem, 1, 1)
            if curr_result is not None:
                layer_results.append(curr_result) 

        elif operation == 'Add':
            out_shape = self.modules[name]['shape_out']
            cout = out_shape[1]
            out_size = int(np.prod(np.array(out_shape[1:])))

            in_shape = self.modules[name]['shape_in']
            cin = in_shape[1]
            in_size = int(np.prod(np.array(in_shape[1:])))

            assert out_shape == in_shape, 'Input and output shapes bust be identical in Add Layer'

            folding_name = "Mem_Bw({}/{})".format(s_in, s_out)

            logging.info("Fold = {}. Channels In = {} - Filters = {}".format(folding_name, cin, cout))

            rate_in, rate_out, muls, adds, mem = self.add_layer_config(s_in=s_in, s_out=s_out)

            curr_result = self.model_layer(name, None, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem, 1 ,1)
            if curr_result is not None:
                layer_results.append(curr_result) 

        elif operation == 'Gemm' or operation == 'MatMul':
            '''
                The code below assumes the implementation of the fully connected layer as a 1x1 convolution with number of channels
                equal to the size of the input tensor and number of filters equal to the size of the output tensor.
            '''
            '''
            in_shape = self.modules[name]['shape_in'].copy()
            in_shape.extend([1, 1, 1])
            out_shape = self.modules[name]['shape_out'].copy()
            out_shape.extend([1, 1, 1])
            kernel_shape = self.modules[name]['kernel'].copy()
            kernel_shape.extend([1, 1, 1])
            padding = [0, 0, 0]
            groups = 1
            
            in_size = in_shape[0] * in_shape[1]
            out_size = out_shape[0] * out_shape[1]

            coarse_in_fc = kernel_shape[1]
            coarse_out_fc = kernel_shape[0]

            coarse_in_config = [1, coarse_in_fc//4, coarse_in_fc//2, coarse_in_fc]
            coarse_in_config = np.unique(coarse_in_config)
            coarse_in_config = coarse_in_config[np.nonzero(coarse_in_config)].tolist()

            coarse_out_config = [1, coarse_out_fc//4, coarse_out_fc//2, coarse_out_fc]
            coarse_out_config = np.unique(coarse_out_config)
            coarse_out_config = coarse_out_config[np.nonzero(coarse_out_config)].tolist()

            fine_config = [0.5, 1]

            mem_bw = s_in + s_out
            mem_bw_config = [(mem_bw*0.2, mem_bw*0.8), (mem_bw*0.4, mem_bw*0.6), (mem_bw*0.5, mem_bw*0.5), (mem_bw*0.6, mem_bw*0.4), (mem_bw*0.8, mem_bw*0.2), (mem_bw*0.9, mem_bw*0.1)]
            for coarse_in in coarse_in_config:
                for coarse_out in coarse_out_config:
                    for fine in fine_config:
                        for conv_bw_in, conv_bw_out in mem_bw_config:
                            coarse_in_name = str(coarse_in)
                            coarse_out_name = str(coarse_out)
                            folding_name = "N_Coarse({}/{}) - f_Fine({:.2f}) - Mem BW({:.2f}/{:.2f})".format(coarse_in_name, coarse_out_name, fine, conv_bw_in, conv_bw_out)

                            logging.info("Fold = {}. Channels In = {} - Filters = {}".format(folding_name, coarse_in_fc, coarse_out_fc))

                            rate_in, rate_out, muls, adds, mem, depth, (mem_bounded_in, mem_bounded_out) = self.conv_layer_config(in_shape, out_shape, kernel_shape, padding, groups, fine, coarse_in, coarse_out, s_in=conv_bw_in, s_out=conv_bw_out)

                            current_config = [in_shape, out_shape, kernel_shape, padding, groups, fine, coarse_in, coarse_out, int(mem_bounded_in), int(mem_bounded_out)]
                            curr_result = self.model_layer(name, None, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem, mem_bounded_out=mem_bounded_out, config=current_config)
                            if curr_result is not None:
                                layer_results.append(curr_result) 
            '''
            '''
                The code below models the fully connected layer as general matrix multiplication operation.
            '''
            in_shape = self.modules[name]['shape_in'].copy()
            out_shape = self.modules[name]['shape_out'].copy()
            kernel_shape = self.modules[name]['kernel'].copy()

            # Here we pass the kernel size as in_size since we are streaming the kernel weights instead of the input tensor of the fully connected layer.
            in_size = kernel_shape[0] * kernel_shape[1]
            out_size = out_shape[0] * out_shape[1]

            coarse_in_fc = kernel_shape[1]
            coarse_out_fc = kernel_shape[0]

            coarse_in_config = [1, coarse_in_fc//16, coarse_in_fc//12, coarse_in_fc//8, coarse_in_fc//4, coarse_in_fc]
            coarse_in_config = np.unique(coarse_in_config)
            coarse_in_config = coarse_in_config[np.nonzero(coarse_in_config)].tolist()

            coarse_out_config = [1, coarse_out_fc//16, coarse_out_fc//12, coarse_out_fc//8, coarse_out_fc//4, coarse_out_fc]
            coarse_out_config = np.unique(coarse_out_config)
            coarse_out_config = coarse_out_config[np.nonzero(coarse_out_config)].tolist()

            fine_config = [0.25, 0.5, 0.75, 1]

            mem_bw = s_in + s_out
            mem_bw_config = [(mem_bw*0.2, mem_bw*0.8), (mem_bw*0.4, mem_bw*0.6), (mem_bw*0.5, mem_bw*0.5), (mem_bw*0.6, mem_bw*0.4), (mem_bw*0.8, mem_bw*0.2), (mem_bw*0.9, mem_bw*0.1)]
            for coarse_in in coarse_in_config:
                for coarse_out in coarse_out_config:
                    for fine in fine_config:
                        for fc_bw_in, fc_bw_out in mem_bw_config:
                            folding_name = "N_Coarse({}/{}) - f_Fine({:.2f}) - Mem BW({:.2f}/{:.2f})".format(coarse_in, coarse_out, fine, fc_bw_in, fc_bw_out)
                            logging.info("Fold = {}. Tensor In = {} - Tensor Out = {}".format(folding_name, coarse_in_fc, coarse_out_fc))

                            rate_in, rate_out, muls, adds, mem, depth, latency, (mem_bounded_in, mem_bounded_out) = self.fc_layer_config(in_shape, out_shape, fine, coarse_in, coarse_out, s_in=s_in, s_out=s_out)

                            curr_result = self.model_layer(name, None, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem, depth, latency, mem_bounded_out=mem_bounded_out, config=[in_shape, out_shape, fine, coarse_in, coarse_out, int(mem_bounded_in), int(mem_bounded_out)])
                            if curr_result is not None:
                                layer_results.append(curr_result) 
        return layer_results

    @timebudget
    def create_design_points(self, file_name, s_in=1, s_out=1):
            if not os.path.exists(os.path.join(os.getcwd(), 'fpga_modeling_reports')):
                os.makedirs(os.path.join(os.getcwd(), 'fpga_modeling_reports'))
            csv_file = os.path.join(os.getcwd(), 'fpga_modeling_reports', file_name + '.csv')
            with open(csv_file, mode='w') as model_results:
                csv_writer = csv.writer(model_results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                csv_writer.writerow(["Layer", "Folding", "On-Chip Memory(BRAM %)", "DSPS %", "Consumption(inputs/sec)", "Throughtput(outputs/sec)", "Memory Bandwidth In(words/cycle)", "Memory Bandwidth Out(words/cycle)", "On-Chip Memory(KB)", "On-Chip Memory(BRAM)", "Memory Bandwidth In(GBs/sec)", "Memory Bandwidth Out(GBs/sec)", "Multipliers", "Adders", "DSPS", "Depth", "Latency", "Throughtput(words/cycle)", "Throughtput(GOps/sec)", "Configuration"])

                result_ids = []
                for k in self.modules.keys():
                    name = k
                    operation = self.modules[k]['operation']
                    logging.warning("Layer: {} -> Operation: {}.".format(name, operation))
                    result_ids.append(self.design_points_per_layer.remote(self, operation=operation, name=name, s_in=s_in, s_out=s_out))
                results = ray.get(result_ids)
                for layer in results:
                    for config in layer:
                        csv_writer.writerow(config)

    def product_dict(self, **kwargs):
        keys = kwargs.keys()
        vals = kwargs.values()
        for instance in itertools.product(*vals):
            yield dict(zip(keys, instance))

    def get_rates(self, layer, config, bw_in, bw_total, bw_out):
        operation = layer.split("_")[0]
        tmp_thr_in, tmp_thr_out = 0, 0
        if operation == 'Conv':
            rate_in, rate_out, muls, adds, mem, depth, latency, (mem_bounded_in, mem_bounded_out) = self.conv_layer_config(config[0], config[1], config[2], config[3], config[4], config[5], config[6], config[7], s_in=bw_in, s_out=bw_out)
            tmp_thr_in = (self.cycles_per_sec*rate_in)/int(np.prod(np.array(config[0][1:])))
            tmp_thr_out = (self.cycles_per_sec*rate_out)/int(np.prod(np.array(config[1][1:])))
            assert math.isclose(tmp_thr_in, tmp_thr_out) or mem_bounded_out, "Input and Output Throughputs doesnt match on CONV operation. Aborting..."
        elif operation == 'Se':
            rate_in, rate_out, muls, adds, mem, depth, latency, (mem_bounded_in, mem_bounded_out) = self.se_layer_config(config[0], config[1], config[2], config[3], config[4], config[5], config[6], config[7], config[8], config[9], config[10], config[11], config[12], config[13], config[14], config[15], config[16], config[17], bw_in=bw_in, bw_total=bw_total, se_on_bram=config[18])
            tmp_thr_in = (self.cycles_per_sec*rate_in)/int(np.prod(np.array(config[0][1:])))
            tmp_thr_out = (self.cycles_per_sec*rate_out)/int(np.prod(np.array(config[0][1:])))
            assert math.isclose(tmp_thr_in, tmp_thr_out) or mem_bounded_out, "Input and Output Throughputs doesnt match on SE operation. Aborting..."
        elif operation == 'GlobalAveragePool':
            rate_in, rate_out, muls, adds, mem, depth, latency, (mem_bounded_in, mem_bounded_out) = self.gap_layer_config(config[0], coarse=config[1], s_in=bw_in, s_out=bw_out)
            tmp_thr_in = (self.cycles_per_sec*rate_in)/int(np.prod(np.array(config[0][1:])))
            tmp_thr_out = (self.cycles_per_sec*rate_out)/int(np.prod(np.array(config[0][1])))
            assert math.isclose(tmp_thr_in, tmp_thr_out) or mem_bounded_out, "Input and Output Throughputs doesnt match on GlobalAveragePool operation. Aborting..."
        elif operation == 'MatMul':
            rate_in, rate_out, muls, adds, mem, depth, latency, (mem_bounded_in, mem_bounded_out) = self.fc_layer_config(config[0], config[1], config[2], config[3], config[4], s_in=bw_in, s_out=bw_out)
            tmp_thr_in = (self.cycles_per_sec*rate_in)/int(config[0][1]*config[1][1])
            tmp_thr_out = (self.cycles_per_sec*rate_out)/int(config[1][1])
            assert math.isclose(tmp_thr_in, tmp_thr_out) or mem_bounded_out, "Input and Output Throughputs doesnt match on MatMul operation. Aborting..."
        elif operation == 'Gemm':
            rate_in, rate_out, muls, adds, mem, depth, latency, (mem_bounded_in, mem_bounded_out) = self.fc_layer_config(config[0], config[1], config[2], config[3], config[4], s_in=bw_in, s_out=bw_out)
            tmp_thr_in = (self.cycles_per_sec*rate_in)/int(config[0][1]*config[1][1])
            tmp_thr_out = (self.cycles_per_sec*rate_out)/int(config[1][1])
            assert math.isclose(tmp_thr_in, tmp_thr_out) or mem_bounded_out, "Input and Output Throughputs doesnt match on Gemm operation. Aborting..."
        elif operation == 'Relu':
            rate_in, rate_out, muls, adds, mem = self.relu_layer_config(s_in=bw_in)
            assert math.isclose(rate_in, rate_out), "Input and Output Rates doesnt match on ReLu operation. Aborting..."
            depth = 1
            latency = 1
        elif operation == 'BatchNormalization':
            rate_in, rate_out, muls, adds, mem = self.batchnorm_layer_config(config[0], s_in=bw_in)
            assert math.isclose(rate_in, rate_out), "Input and Output Rates doesnt match on BatchNormalization operation. Aborting..."
            depth = 1
            latency = 1
        elif operation == 'Swish':
            rate_in, rate_out, muls, adds, mem = self.swish_layer_config(s_in=bw_in)
            assert math.isclose(rate_in, rate_out), "Input and Output Rates doesnt match on Swish operation. Aborting..."
            depth = 1
            latency = 1
        elif operation == 'Sigmoid':
            rate_in, rate_out, muls, adds, mem = self.sigmoid_layer_config(s_in=bw_in)
            assert math.isclose(rate_in, rate_out), "Input and Output Rates doesnt match on Sigmoid operation. Aborting..."
            depth = 1
            latency = 1
        elif operation == 'Add':
            rate_in, rate_out, muls, adds, mem = self.add_layer_config(s_in=bw_in)
            assert math.isclose(rate_in, rate_out), "Input and Output Rates doesnt match on Add operation. Aborting..."
            depth = 1
            latency = 1
        elif operation == 'Mul':
            rate_in, rate_out, muls, adds, mem = self.mul_layer_config(s_in=bw_in)
            assert math.isclose(rate_in, rate_out), "Input and Output Rates doesnt match on Mul operation. Aborting..."
            depth = 1
            latency = 1

        return rate_in, rate_out, muls, adds, mem, depth, latency, tmp_thr_in, tmp_thr_out

    @timebudget
    def non_branching_layer(self, layer_keys, r, membw, dsp_config, depth_config, latency_config, bram_config, bram_total_util, mem_bw_status, throughput_config, params_config):
        #TODO: This is a hardcoded version of supporting the x3d_m classifier. Should be revised in the future for a more generic implementation.
        classifier = False
        if self.model_name == 'x3d_m' and layer_keys == ['Relu_387', 'Conv_388', 'BatchNormalization_389', 'Relu_390', 'GlobalAveragePool_391', 'MatMul_401', 'Relu_402', 'Gemm_403']:
            classifier = True

        in_shape = self.modules[layer_keys[0]]['shape_in']
        in_size = int(np.prod(np.array(in_shape[1:])))
        out_shape = self.modules[layer_keys[-1]]['shape_out']
        out_size = int(np.prod(np.array(out_shape[1:])))

        membw_config = [0.2*membw, 0.3*membw, 0.4*membw, 0.5*membw, 0.6*membw, 0.7*membw, 0.8*membw]
        mem_on_chip_bw = 10000
        fc_layers_count = 0
        for l in layer_keys:
            if 'MatMul' in l.split('_') or 'Gemm' in l.split('_'):
                fc_layers_count += 1

        params_per_module = {}
        for mem_bw_in in membw_config:
            if classifier:
                fc_layer_bw = (membw - mem_bw_in)/2
                fc_layer_bw_split = fc_layer_bw/fc_layers_count

            total_latency = 0
            total_depth = 0
            total_mem = 0
            total_muls = 0
            total_adds = 0
            
            rate_graph = np.zeros( shape=(len(layer_keys),len(layer_keys)+1) , dtype=float )
            prev_mod_rout = 0
            early_exit = False

            for i, k in enumerate(layer_keys):
                if i == 0:
                    mod_rin, mod_rout, mod_muls, mod_adds, mod_mem, mod_depth, mod_latency, mod_thrin, mod_throut = self.get_rates(k, r[k][-1], mem_bw_in, membw, mem_on_chip_bw)
                else:
                    if classifier and ('MatMul' in k.split('_') or 'Gemm' in k.split('_')):
                        mod_rin, mod_rout, mod_muls, mod_adds, mod_mem, mod_depth, mod_latency, mod_thrin, mod_throut = self.get_rates(k, r[k][-1], fc_layer_bw_split, membw, mem_on_chip_bw)
                    else:
                        mod_rin, mod_rout, mod_muls, mod_adds, mod_mem, mod_depth, mod_latency, mod_thrin, mod_throut = self.get_rates(k, r[k][-1], prev_mod_rout, membw, mem_on_chip_bw)
                prev_mod_rout = mod_rout

                param_dict = {}
                if 'MatMul' in k.split('_') or 'Gemm' in k.split('_'):
                    param_dict["fine"] = r[k][-1][2]
                    param_dict["coarse in"] = r[k][-1][3]
                    param_dict["coarse out"] = r[k][-1][4]
                    params_per_module[k] = param_dict
                elif 'Conv' in k.split('_'):
                    param_dict["fine"] = r[k][-1][5]
                    param_dict["coarse in"] = r[k][-1][6]
                    param_dict["coarse out"] = r[k][-1][7]
                    params_per_module[k] = param_dict
                elif 'Se' in k.split('_'):
                    param_dict["coarse gap"] = r[k][-1][1]
                    param_dict["fine conv1"] = r[k][-1][7]
                    param_dict["coarse in conv1"] = r[k][-1][8]
                    param_dict["coarse out conv1"] = r[k][-1][9]
                    param_dict["fine conv2"]  = r[k][-1][15]
                    param_dict["coarse in conv2"] = r[k][-1][16]
                    param_dict["coarse out conv2"] = r[k][-1][17]
                    params_per_module[k] = param_dict
                elif 'GlobalAveragePool' in k.split('_'):
                    param_dict["coarse"] = r[k][-1][1]
                    params_per_module[k] = param_dict

                if classifier and k == 'MatMul_401':
                    fc1_thr_in = mod_thrin
                    fc1_thr_out = mod_throut
                if classifier and k == 'Gemm_403':
                    fc2_thr_in = mod_thrin
                    fc2_thr_out = mod_throut

                rate_graph[i,i] = mod_rin
                rate_graph[i,i+1] = mod_rout

                total_depth += mod_depth
                total_mem += mod_mem
                total_muls += mod_muls
                total_adds += mod_adds

                if "Se" in k:
                    se_layer_key = k
                    se_layer_bwin = mod_rin
                    se_layer_bwout = mod_rout
                if (total_muls/self.fpga_dsps)*100 > 100.0 or ((math.ceil(((total_mem*self.wb)/1e3)/(self.bram_mem/8)))/self.fpga_bram)*100 > 100.0:
                    early_exit = True
                    break
            if early_exit:
                continue
            
            if classifier:
                rate_graph = rate_graph[:5,:6]
                out_size = r['GlobalAveragePool_391'][-1][0][1]
                
            rate_graph = self.balance_module_rates(rate_graph)

            mem_bounded_in = False
            if mem_bw_in < rate_graph[0,0]:
                in_module_ratio = rate_graph[0,1] / rate_graph[0,0]
                rate_graph[0,0] = mem_bw_in
                rate_graph[0,1] = rate_graph[0,0] * in_module_ratio
                assert math.isclose(in_module_ratio, rate_graph[0,1] / rate_graph[0,0]), "wrong calculation of ratio" 
                rate_graph = self.balance_module_rates(rate_graph)
                mem_bounded_in = True

            rate_in = abs(rate_graph[0,0])
            rate_out = abs(rate_graph[-1,-1])
            
            mem_bw_left = membw - rate_in
            if classifier:
                mem_bw_left -= fc_layer_bw
            mem_bounded_out = False
            if mem_bw_left < rate_out:
                mem_bounded_out = True
                rate_out = mem_bw_left
            
            thr_in = (self.cycles_per_sec*rate_in)/in_size
            thr_out = (self.cycles_per_sec*rate_out)/out_size
            if classifier:
                thr_in = min(thr_in, fc1_thr_in, fc2_thr_in)
                thr_out = min(thr_out, fc1_thr_out, fc2_thr_out)

            assert math.isclose(thr_out, thr_in) or mem_bounded_out, "Input and Output Throughput doesnt match. Aborting..."

            mem_kb = (total_mem*self.wb)/1e3
            bram_util = math.ceil(mem_kb/(self.bram_mem/8))
            bram_util = (bram_util/self.fpga_bram)*100

            dsps_util = (total_muls/self.fpga_dsps)*100
            
            if bram_util < 90.0:
                bram_config.append("Below 90% BRAM")
            else:
                bram_config.append("Over 90% BRAM")
            
            if mem_bounded_in and mem_bounded_out:
                mem_bw_status.append("Memory Bounded (IN/OUT)")
            elif mem_bounded_in and not mem_bounded_out:
                mem_bw_status.append("Memory Bounded IN")
            elif not mem_bounded_in and mem_bounded_out:
                mem_bw_status.append("Memory Bounded OUT")
            else:
                mem_bw_status.append("Compute Bounded")
            
            workload_in = int(np.prod(np.array(in_shape)))
            workload_out = int(np.prod(np.array(out_shape)))
            total_latency = max(workload_in/rate_in, workload_out/rate_out)
            latency_config.append(total_latency)
            depth_config.append(total_depth)
            dsp_config.append(dsps_util)
            bram_total_util.append(bram_util)
            throughput_config.append(thr_out)
            params_config.append(json.dumps(params_per_module))

    @ray.remote
    @timebudget
    def compose_layers(self, file_name, layers_names, final_name, model_name, calculate_pareto, membw, branch_on_bram, plot_design_points=False):
        sns.set(rc={'figure.figsize':(15,8)})
        sns.set_style("whitegrid")

        l_configs = {}
        for l in layers_names:
            l_configs[l] = []

            csv_file = os.path.join(os.getcwd(), 'fpga_modeling_reports', file_name + '.csv')
            with open(csv_file, mode='r') as model_results:
                csv_reader = csv.reader(model_results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                cols = {}
                for i, c in enumerate(next(csv_reader)):
                    cols[c] = i

                for i, row in enumerate(csv_reader):
                    if "-" in row :
                        continue

                    if row[cols['Layer']] == l:
                        # l_configs[l].append([row[cols['Folding']], row[cols['On-Chip Memory(BRAM %)']], row[cols['DSPS %']], row[cols['Consumption(inputs/sec)']], row[cols['Throughtput(outputs/sec)']], row[cols['Memory Bandwidth In(words/cycle)']], row[cols['Memory Bandwidth Out(words/cycle)']]])
                        if l.split("_")[0] == 'Conv' or l.split("_")[0] == 'Se' or l.split("_")[0] == 'BatchNormalization' or l.split("_")[0] == 'GlobalAveragePool' or l.split("_")[0] == 'MatMul' or l.split("_")[0] == 'Gemm':
                            l_configs[l].append([row[cols['Folding']], json.loads(row[cols['Configuration']])])
                        else:
                            l_configs[l].append([row[cols['Folding']], row[cols['Configuration']]])

        sizes = []
        keys = []
        for k in l_configs.keys():
            sizes.append(len(l_configs[k]))
            keys.append(k)
        
        dsp_config = []
        latency_config = []
        depth_config = []
        bram_config = []
        bram_total_util = []
        mem_bw_status = []
        throughput_config = []
        params_config = []

        res = list(self.product_dict(**l_configs))
        
        for r in res:
            layer_keys = list(r.keys())
            se_layer_key = None
            se_layer_bwin = 0
            se_layer_bwout = 0

            branches_points = [i for i in range(len(layer_keys)) if self.modules[layer_keys[i]]['branching'] and not layer_keys[i].split("_")[0] == "Se"]
            if len(branches_points) == 0:
                self.non_branching_layer(layer_keys, r, membw, dsp_config, depth_config, latency_config, bram_config, bram_total_util, mem_bw_status, throughput_config, params_config)
                throughput_config, dsp_config, bram_total_util, mem_bw_status, depth_config, latency_config, params_config
                continue

            in_shape_wl = self.modules[layer_keys[0]]['shape_in']
            in_shape = self.modules[layer_keys[-1]]['shape_in']
            in_size = int(np.prod(np.array(in_shape[1:])))
            out_shape = self.modules[layer_keys[-1]]['shape_out']
            out_size = int(np.prod(np.array(out_shape[1:])))

            membw_config = [0.2*membw, 0.3*membw, 0.4*membw, 0.5*membw, 0.6*membw, 0.7*membw, 0.8*membw]
            mem_on_chip_bw = 10000
            
            params_per_module = {}
            for mem_bw_in in membw_config:
                total_latency = 0
                total_depth = 0
                total_mem = 0
                total_muls = 0
                total_adds = 0

                rates_graph_list = []
                bp_origin = branches_points[0]
                rates_graph = np.zeros( shape=(bp_origin,bp_origin+1) , dtype=float )
                rates_graph_list.append(rates_graph)
                for bp in branches_points[1:]:
                    rates_graph = np.zeros( shape=(bp-bp_origin,bp-bp_origin+1) , dtype=float )
                    rates_graph_list.append(rates_graph)
                prev_mod_rout = 0
                early_exit = False

                rg_idx = 0
                init_bp = 0
                for bp in branches_points:
                    for i, k in enumerate(layer_keys[init_bp:bp]):
                        if i == 0:
                            if rg_idx == 0:
                                mod_rin, mod_rout, mod_muls, mod_adds, mod_mem, mod_depth, mod_latency, mod_thrin, mod_throut = self.get_rates(k, r[k][-1], mem_bw_in, membw, mem_on_chip_bw)
                            else:
                                mod_rin, mod_rout, mod_muls, mod_adds, mod_mem, mod_depth, mod_latency, mod_thrin, mod_throut = self.get_rates(k, r[k][-1], mem_on_chip_bw, membw, mem_on_chip_bw)
                        else:
                            mod_rin, mod_rout, mod_muls, mod_adds, mod_mem, mod_depth, mod_latency, mod_thrin, mod_throut = self.get_rates(k, r[k][-1], prev_mod_rout, membw, mem_on_chip_bw)

                        param_dict = {}
                        if 'MatMul' in k.split('_') or 'Gemm' in k.split('_'):
                            param_dict["fine"] = r[k][-1][2]
                            param_dict["coarse in"] = r[k][-1][3]
                            param_dict["coarse out"] = r[k][-1][4]
                            params_per_module[k] = param_dict
                        elif 'Conv' in k.split('_'):
                            param_dict["fine"] = r[k][-1][5]
                            param_dict["coarse in"] = r[k][-1][6]
                            param_dict["coarse out"] = r[k][-1][7]
                            params_per_module[k] = param_dict
                        elif 'Se' in k.split('_'):
                            param_dict["coarse gap"] = r[k][-1][1]
                            param_dict["fine conv1"] = r[k][-1][7]
                            param_dict["coarse in conv1"] = r[k][-1][8]
                            param_dict["coarse out conv1"] = r[k][-1][9]
                            param_dict["fine conv2"]  = r[k][-1][15]
                            param_dict["coarse in conv2"] = r[k][-1][16]
                            param_dict["coarse out conv2"] = r[k][-1][17]
                            params_per_module[k] = param_dict
                        elif 'GlobalAveragePool' in k.split('_'):
                            param_dict["coarse"] = r[k][-1][1]
                            params_per_module[k] = param_dict

                        prev_mod_rout = mod_rout

                        rates_graph_list[rg_idx][i,i] = mod_rin
                        rates_graph_list[rg_idx][i,i+1] = mod_rout

                        total_depth += mod_depth
                        total_mem += mod_mem
                        total_muls += mod_muls
                        total_adds += mod_adds

                        if "Se" in k:
                            se_layer_key = k
                            se_layer_bwin = mod_rin
                            se_layer_bwout = mod_rout
                        if (total_muls/self.fpga_dsps)*100 > 100.0 or ((math.ceil(((total_mem*self.wb)/1e3)/(self.bram_mem/8)))/self.fpga_bram)*100 > 100.0:
                            early_exit = True
                            break
                    init_bp = bp
                    rg_idx += 1
                if early_exit:
                    continue
                
                
                mem_bounded_in = False
                for i_r, rate_graph in enumerate(rates_graph_list):
                    rate_graph = self.balance_module_rates(rate_graph)
                    if i_r == 0:
                        branch_mem_bw = mem_bw_in
                    elif i_r > 0 and branch_on_bram:
                        branch_mem_bw = mem_on_chip_bw
                    elif i_r > 0 and not branch_on_bram:
                        branch_mem_bw = membw - rates_graph_list[i_r-1][0,0] - rates_graph_list[i_r-1][-1,-1]

                    if branch_mem_bw < rate_graph[0,0]:
                        in_module_ratio = rate_graph[0,1] / rate_graph[0,0]
                        rate_graph[0,0] = branch_mem_bw
                        rate_graph[0,1] = rate_graph[0,0] * in_module_ratio
                        assert math.isclose(in_module_ratio, rate_graph[0,1] / rate_graph[0,0]), "wrong calculation of ratio" 
                        rate_graph = self.balance_module_rates(rate_graph)
                        mem_bounded_in = True

                mem_bounded_out = False
                se_on_bram = 0 if se_layer_key is None else r[se_layer_key][-1][18]
                if len(rates_graph_list) == 1:
                    rate_in = abs(rates_graph_list[0][0,0])
                    rate_out = abs(rates_graph_list[0][-1,-1])

                    final_layer = layer_keys[branches_points[-1]]
                    mod_rin, mod_rout, mod_muls, mod_adds, mod_mem, mod_depth, mod_latency, mod_thrin, mod_throut = self.get_rates(final_layer, r[final_layer][-1], rate_in, membw, mem_on_chip_bw)
                    
                    total_depth += mod_depth
                    total_mem += mod_mem
                    total_muls += mod_muls
                    total_adds += mod_adds

                    rate_in = mod_rin
                    rate_out = mod_rout

                    mem_bw_left = membw - abs(rates_graph_list[0][0,0])
                    if se_on_bram == 0:
                        mem_bw_left -= se_layer_bwin + se_layer_bwout
                    if branch_on_bram:
                        best_mem_bw_out = mem_bw_left
                    else:
                        mem_bw_left -= mod_rin
                        if mem_bw_left > 0:
                            best_mem_bw_out = mem_bw_left
                        else:
                            mem_bw_left += mod_rin
                            best_mem_bw_out = mem_bw_left/2
                            mem_bounded_in = True
                else:
                    rates_in = []
                    rates_out = []
                    for r_g in rates_graph_list:
                        rate_in = abs(r_g[0,0])
                        rate_out = abs(r_g[-1,-1])
                        rates_in.append(rate_in)
                        rates_out.append(rate_out)

                    rate_in_idx = rates_in.index(min(rates_in))
                    if not rate_in_idx == 0:
                        mem_bounded_in = True
                    rate_in = rates_in[rate_in_idx]
                    rate_out = rates_out[rate_in_idx]

                    final_layer = layer_keys[branches_points[-1]]
                    mod_rin, mod_rout, mod_muls, mod_adds, mod_mem, mod_depth, mod_latency, mod_thrin, mod_throut = self.get_rates(final_layer, r[final_layer][-1], rate_in, membw, mem_on_chip_bw)

                    total_depth += mod_depth
                    total_mem += mod_mem
                    total_muls += mod_muls
                    total_adds += mod_adds

                    rate_in = mod_rin
                    rate_out = mod_rout

                    mem_bw_left = membw - rates_in[0]
                    if se_on_bram == 0:
                        mem_bw_left -= se_layer_bwin + se_layer_bwout
                    if branch_on_bram:
                        best_mem_bw_out = mem_bw_left
                    else:
                        mem_bw_left -= mod_rin
                        if mem_bw_left > 0:
                            best_mem_bw_out = mem_bw_left
                        else:
                            mem_bw_left += mod_rin
                            best_mem_bw_out = mem_bw_left/2
                            mem_bounded_in = True

                mem_bounded_out = False
                if best_mem_bw_out < rate_out:
                    mem_bounded_out = True
                    rate_out = best_mem_bw_out

                thr_in = (self.cycles_per_sec*rate_in)/in_size
                thr_out = (self.cycles_per_sec*rate_out)/out_size
                assert math.isclose(thr_out, thr_in) or mem_bounded_out, "Input and Output Throughput doesnt match. Aborting..."

                if branch_on_bram:
                    total_mem += total_depth
                mem_kb = (total_mem*self.wb)/1e3
                bram_util = math.ceil(mem_kb/(self.bram_mem/8))
                bram_util = (bram_util/self.fpga_bram)*100

                dsps_util = (total_muls/self.fpga_dsps)*100
                
                if bram_util < 90.0:
                    bram_config.append("Below 90% BRAM")
                else:
                    bram_config.append("Over 90% BRAM")
                
                if mem_bounded_in and mem_bounded_out:
                    mem_bw_status.append("Memory Bounded (IN/OUT)")
                elif mem_bounded_in and not mem_bounded_out:
                    mem_bw_status.append("Memory Bounded IN")
                elif not mem_bounded_in and mem_bounded_out:
                    mem_bw_status.append("Memory Bounded OUT")
                else:
                    mem_bw_status.append("Compute Bounded")
                
                workload_in = int(np.prod(np.array(in_shape_wl)))
                workload_out = int(np.prod(np.array(out_shape)))
                total_latency = max(workload_in/rate_in, workload_out/rate_out)
                latency_config.append(total_latency)
                depth_config.append(total_depth)
                dsp_config.append(dsps_util)
                bram_total_util.append(bram_util)
                throughput_config.append(thr_out)
                params_config.append(json.dumps(params_per_module))

        if calculate_pareto:
            scores = np.zeros((len(throughput_config), 2))
            scores[:,0] = throughput_config
            scores[:,1] = dsp_config
            pareto = find_pareto(scores)
            pareto_front = scores[pareto]

            pareto_front_df = pd.DataFrame(pareto_front)
            pareto_front_df.sort_values(0, inplace=True)
            pareto_front = pareto_front_df.values

            sns.lineplot(x=pareto_front[:, 0], y=pareto_front[:, 1], color='red')

        # Search the points in pareto front (per layer) with the maximum throughput and save them in a csv file.
        model_name_ = file_name.split("onnx")[0]
        csv_file = os.path.join(os.getcwd(), 'fpga_modeling_reports', model_name_ + 'max_throughput_design_points.csv')
        max_throughput = 0
        best_dsp = 0
        best_bram = 0
        best_latency = 0
        best_depth = 0
        best_bw_stat = "Unknown"
        best_params = []

        for thr, dsp, bram, bw_stat, p_depth, p_latency, mod_params in zip(throughput_config, dsp_config, bram_total_util, mem_bw_status, depth_config, latency_config, params_config):
            if thr > max_throughput and dsp < 90.0 and bram < 90.0:
                max_throughput = thr
                best_dsp = dsp
                best_bram = bram
                best_depth = p_depth
                best_latency = p_latency
                best_bw_stat = bw_stat
                best_params = mod_params
            if thr == max_throughput:
                if (dsp < best_dsp and bram <= best_bram) or (dsp <= best_dsp and bram < best_bram):
                    best_dsp = dsp
                    best_bram = bram
                    best_depth = p_depth
                    best_latency = p_latency
                    best_bw_stat = bw_stat
                    best_params = mod_params
        with open(csv_file, mode='a') as model_results:
            csv_writer = csv.writer(model_results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if final_name == 1:
                csv_writer.writerow(["Layer Name", "Latency(cycles new)", "Latency(cycles/output)", "Throughput(outputs/sec)", "Latency(secs/output)", "DSPs(%)", "BRAM(%)", "Memory BW Status", "Pipeline Depth", "Modules Parameters"])
            interval = math.ceil(1/(max_throughput/self.cycles_per_sec))
            csv_writer.writerow([final_name, int(best_latency), interval, max_throughput, 1/max_throughput, best_dsp, best_bram, best_bw_stat, int(best_depth), best_params])
            logging.warning("Best config for layer {}. Latency(cycles new) = {}, Latency(cycles/output) = {}, Throughput(outputs/sec) = {:.5f}, Latency(secs/output) = {:.5f}, DSPs(%) = {:.5f}, BRAM(%) = {:.5f}, Mem BW = {}, Pipeline Depth = {}, Modules Parameters = {}".format(final_name, int(best_latency), interval, max_throughput, 1/max_throughput, best_dsp, best_bram, best_bw_stat, int(best_depth), best_params))

        if plot_design_points:
            sns.scatterplot(x=throughput_config, y=dsp_config, hue=bram_config, style=mem_bw_status, alpha=.5, size=bram_total_util)
            plt.axhline(y=100, color='r', linestyle='-')
            plt.axhline(y=90, color='r', linestyle='--')

            plt.title(str(final_name))
            plt.xlabel('Throughtput(outputs/sec)')
            plt.xscale("log")
            plt.ylabel('DSPS %')
            plt.legend(frameon=False, prop={"size":8}, loc='upper right', bbox_to_anchor=(1.11, 1.12), borderaxespad=0.)
            if not os.path.exists(os.path.join(os.getcwd(), 'fpga_modeling_reports', 'graphs', model_name, 'partition_layers')):
                os.makedirs(os.path.join(os.getcwd(), 'fpga_modeling_reports', 'graphs', model_name, 'partition_layers'))
            partitions_path = os.path.join(os.getcwd(), 'fpga_modeling_reports', 'graphs', model_name, 'partition_layers')
            plt.savefig(os.path.join(partitions_path, str(final_name) + ".png"))
            plt.clf()
        

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

def plot_graph(x, y, bram_util, bram, mem_compute_bounded, leg, name, type, model_name, calculate_pareto):
    se_layer = True if "Se" in name.split("_") else False
    
    dsps_dir = os.path.join(os.getcwd(), 'fpga_modeling_reports', 'graphs', model_name, 'throughput_dsps')
    if not os.path.exists(dsps_dir):
        os.makedirs(dsps_dir)
    mem_bw_dir = os.path.join(os.getcwd(), 'fpga_modeling_reports', 'graphs', model_name, 'throughput_mem_bw')
    if not os.path.exists(mem_bw_dir):
        os.makedirs(mem_bw_dir)
    if type == 'DSPS':

        if calculate_pareto:
            scores = np.zeros((len(x), 2))
            scores[:,0] = x
            scores[:,1] = y[0]

            pareto = find_pareto(scores)
            pareto_front = scores[pareto]

            pareto_front_df = pd.DataFrame(pareto_front)
            pareto_front_df.sort_values(0, inplace=True)
            pareto_front = pareto_front_df.values

            sns.lineplot(x=pareto_front[:, 0], y=pareto_front[:, 1], color='red')

        sns.scatterplot(x=np.array(x), y=np.array(y[0]), hue=bram, style=mem_compute_bounded, s=75) # , size=bram_util, alpha=0.5

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
            if calculate_pareto:
                legd = []
                for l in pareto:
                    legd.append(leg[l])

        plt.legend(frameon=False, prop={"size":8}, loc='upper right', bbox_to_anchor=(1.11, 1.12), borderaxespad=0.)

        file_name = name.replace('.', '_') + '.jpg'
        plt.savefig(os.path.join(dsps_dir, file_name))
        plt.clf()
    elif type == 'Memory Bandwidth':

        sns.scatterplot(x=np.array(x), y=np.array(y[0]), hue=bram, style=mem_compute_bounded, s=75) # , size=bram_util, alpha=0.5

        plt.title(name)
        plt.xlabel('Throughtput(outputs/sec)')
        plt.ylabel('Memory Bandwidth IN (GBs/sec)')
        if max(x) > 100:
            plt.xscale("log")
        plt.legend(frameon=False, prop={"size":8}, loc='upper right', bbox_to_anchor=(1.11, 1.12), borderaxespad=0.)

        file_name = name.replace('.', '_') + '_in.jpg'
        plt.savefig(os.path.join(mem_bw_dir, file_name))
        plt.clf()


        sns.scatterplot(x=np.array(x), y=np.array(y[1]), hue=bram, style=mem_compute_bounded, s=75) # , size=bram_util, alpha=0.5

        plt.title(name)
        plt.xlabel('Throughtput(outputs/sec)')
        plt.ylabel('Memory Bandwidth OUT (GBs/sec)')
        if max(x) > 100:
            plt.xscale("log")
        plt.legend(frameon=False, prop={"size":5}, loc='upper right', bbox_to_anchor=(1.05, 1.05), borderaxespad=0.)

        file_name = name.replace('.', '_') + '_out.jpg'
        plt.savefig(os.path.join(mem_bw_dir, file_name))
        plt.clf()

@timebudget
def performance_graphs(file_name="x3d_m", layers_to_plot=None, calculate_pareto=False):
    
    csv_file = os.path.join(os.getcwd(), 'fpga_modeling_reports', file_name + '.csv')
    with open(csv_file, mode='r') as model_results:
        csv_reader = csv.reader(model_results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        cols = {}
        for i, c in enumerate(next(csv_reader)):
            cols[c] = i
        print(cols)

        bram = []
        bram_util = []
        mem_compute_bounded = []
        folding = []
        dsp_util = []
        mem_bw_in = []
        mem_bw_out = []
        throughput = []

        first_layer = True
        for i, row in enumerate(csv_reader):
            operation = row[cols['Layer']].split("_")[0]
            if "-" in row or (layers_to_plot is not None and operation not in layers_to_plot):
                continue

            if first_layer and (i == 0 or (layers_to_plot is not None and operation in layers_to_plot)):
                prev_layer = row[cols['Layer']]
                first_layer = False

            if row[cols['Layer']] == prev_layer:
                bram_status = ""
                if float(row[cols['On-Chip Memory(BRAM %)']]) > 90:
                    bram_status = "Over 90 % BRAM"
                else:
                    bram_status = "Below 90 % BRAM"
                bram.append(bram_status)

                bounded = ""
                conf = json.loads(row[cols['Configuration']])
                mem_bounded_in, mem_bounded_out = conf[-2], conf[-1]
                if mem_bounded_in == 1 and mem_bounded_out == 1:
                    bounded = "Memory Bounded (IN/OUT)"
                elif mem_bounded_in == 1 and mem_bounded_out == 0:
                    bounded = "Memory Bounded IN"
                elif mem_bounded_in == 0 and mem_bounded_out == 1:
                    bounded = "Memory Bounded OUT"
                else:
                    bounded = "Compute Bounded"
                mem_compute_bounded.append(bounded)

                folding.append(row[cols['Folding']])
                dsp_util.append(float(row[cols['DSPS %']]))
                bram_util.append(float(row[cols['On-Chip Memory(BRAM %)']]))
                mem_bw_in.append(float(row[cols['Memory Bandwidth In(GBs/sec)']]))
                mem_bw_out.append(float(row[cols['Memory Bandwidth Out(GBs/sec)']]))
                throughput.append(float(row[cols['Throughtput(outputs/sec)']]))
            else:
                plot_graph(throughput, [dsp_util], bram_util, bram, mem_compute_bounded, folding, prev_layer, 'DSPS', file_name, calculate_pareto=calculate_pareto)
                plot_graph(throughput, [mem_bw_in, mem_bw_out], bram_util, bram, mem_compute_bounded, folding, prev_layer, 'Memory Bandwidth', file_name, calculate_pareto=calculate_pareto)

                bram.clear()
                mem_compute_bounded.clear()
                folding.clear()
                dsp_util.clear()
                bram_util.clear()
                mem_bw_in.clear()
                mem_bw_out.clear()
                throughput.clear()
                
                bram_status = ""
                if float(row[cols['On-Chip Memory(BRAM %)']]) > 90:
                    bram_status = "Over 90 % BRAM"
                else:
                    bram_status = "Below 90 % BRAM"
                bram.append(bram_status)

                bounded = ""
                conf = json.loads(row[cols['Configuration']])
                mem_bounded_in, mem_bounded_out = conf[-2], conf[-1]
                if mem_bounded_in == 1 and mem_bounded_out == 1:
                    bounded = "Memory Bounded (IN/OUT)"
                elif mem_bounded_in == 1 and mem_bounded_out == 0:
                    bounded = "Memory Bounded IN"
                elif mem_bounded_in == 0 and mem_bounded_out == 1:
                    bounded = "Memory Bounded OUT"
                else:
                    bounded = "Compute Bounded"
                mem_compute_bounded.append(bounded)

                folding.append(row[cols['Folding']])
                dsp_util.append(float(row[cols['DSPS %']]))
                bram_util.append(float(row[cols['On-Chip Memory(BRAM %)']]))
                mem_bw_in.append(float(row[cols['Memory Bandwidth In(GBs/sec)']]))
                mem_bw_out.append(float(row[cols['Memory Bandwidth Out(GBs/sec)']]))
                throughput.append(float(row[cols['Throughtput(outputs/sec)']]))

            prev_layer = row[cols['Layer']]

        plot_graph(throughput, [dsp_util], bram_util, bram, mem_compute_bounded, folding, prev_layer, 'DSPS', file_name, calculate_pareto=calculate_pareto)
        plot_graph(throughput, [mem_bw_in, mem_bw_out], bram_util, bram, mem_compute_bounded, folding, prev_layer, 'Memory Bandwidth', file_name, calculate_pareto=calculate_pareto)

def drop_duplicates(file_name="x3d_m", pareto=False):
    
    if pareto:
        csv_file_read = os.path.join(os.getcwd(), 'fpga_modeling_reports', file_name + '_pareto.csv')
    else:
        csv_file_read = os.path.join(os.getcwd(), 'fpga_modeling_reports', file_name + '.csv')

    data = pd.read_csv(csv_file_read)
    original_size = len(data.index)
    columns = data.columns.tolist()
    del(columns[1])
    del(columns[-1])
    del(columns[-1])

    data_droped = data.drop_duplicates(subset=columns)
    final_size = len(data_droped.index)
    logging.warning("Dropped {} rows due to duplicate".format(original_size - final_size))
    os.remove(csv_file_read)
    data_droped.to_csv(csv_file_read, index=False)

def plot_best_config_params(file_name="x3d_m"):
    csv_file = os.path.join(os.getcwd(), 'fpga_modeling_reports', file_name + '_max_throughput_design_points.csv')
    with open(csv_file, mode='r') as model_results:
        csv_reader = csv.reader(model_results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        cols = {}
        for i, c in enumerate(next(csv_reader)):
            cols[c] = i
        logging.info(cols)

        fine = []
        coarse_in = []
        coarse_out = []
        for i, row in enumerate(csv_reader):
            curr_dict = json.loads(row[cols["Modules Parameters"]])
            for layer in curr_dict.keys():
                operation_type = layer.split('_')[0]
                if operation_type == 'Conv':
                    fine.append(curr_dict[layer]["fine"])
                    coarse_in.append(curr_dict[layer]["coarse in"])
                    coarse_out.append(curr_dict[layer]["coarse out"])
                if operation_type == 'Se':
                    fine.append(curr_dict[layer]["fine conv1"])
                    fine.append(curr_dict[layer]["fine conv2"])
                    coarse_in.append(curr_dict[layer]["coarse in conv1"])
                    coarse_in.append(curr_dict[layer]["coarse in conv2"])
                    coarse_out.append(curr_dict[layer]["coarse out conv1"])
                    coarse_out.append(curr_dict[layer]["coarse out conv2"])

        x_axis = np.arange(0, len(fine))
        sns.lineplot(x=x_axis, y=fine, label='fine')
        sns.lineplot(x=x_axis, y=coarse_in, label='coarse in')
        sns.lineplot(x=x_axis, y=coarse_out, label='coarse out')

        plt.title('Parameters configuration over layers')
        plt.legend()
        plt.xlabel('Conv Layers')
        plt.yscale('log')
        plt.savefig('fpga_modeling_reports/param_analysis/' + file_name + '/best_configuration_param_comparison.png')

@timebudget
def get_paretto(file_name="x3d_m"):
    
    csv_file_par = os.path.join(os.getcwd(), 'fpga_modeling_reports', file_name + '_pareto.csv')
    with open(csv_file_par, mode='w') as pareto_results:
        csv_writer_par = csv.writer(pareto_results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csv_writer_par.writerow(["Layer", "Folding", "On-Chip Memory(BRAM %)", "DSPS %", "Consumption(inputs/sec)", "Throughtput(outputs/sec)", "Memory Bandwidth In(words/cycle)", "Memory Bandwidth Out(words/cycle)", "On-Chip Memory(KB)", "On-Chip Memory(BRAM)", "Memory Bandwidth In(GBs/sec)", "Memory Bandwidth Out(GBs/sec)", "Multipliers", "Adders", "DSPS", "Depth", "Latency", "Throughtput(words/cycle)", "Throughtput(GOps/sec)", "Configuration"])

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

def get_partition_layers(layers, model_name):
    final_layers = []
    if model_name == 'x3d_m':
        layer_type_1 = ['Relu', 'Conv', 'BatchNormalization', 'Relu', 'Conv', 'BatchNormalization', 'SqueezeExcitation', 'Swish', 'Conv', 'BatchNormalization', 'Conv', 'BatchNormalization', 'Add']
        layer_type_2 = ['Relu', 'Conv', 'BatchNormalization', 'Relu', 'Conv', 'BatchNormalization', 'SqueezeExcitation', 'Swish', 'Conv', 'BatchNormalization', 'Add']
        layer_type_3 = ['Relu', 'Conv', 'BatchNormalization', 'Relu', 'Conv', 'BatchNormalization', 'Swish', 'Conv', 'BatchNormalization', 'Add']
        layer_type_4 = ['Conv', 'Conv', 'BatchNormalization']
        layer_type_5 = ['Relu', 'Conv', 'BatchNormalization', 'Relu', 'GlobalAveragePool', 'MatMul', 'Relu', 'Gemm']
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
            if list(layer_queue_operations)[:-10] == layer_type_4:
                final_layers.append(list(layer_queue)[:-10])
            if list(layer_queue_operations)[5:] == layer_type_5:
                final_layers.append(list(layer_queue)[5:])
    elif model_name == 'i3d':
        layer_type_1 = ['Conv', 'BatchNormalization', 'Relu', 'Conv', 'BatchNormalization', 'Relu', 'Conv', 'BatchNormalization', 'Conv', 'BatchNormalization', 'Add']
        layer_type_2 = ['Relu', 'Conv', 'BatchNormalization', 'Relu', 'Conv', 'BatchNormalization', 'Relu', 'Conv', 'BatchNormalization', 'Add']
        layer_queue = deque(maxlen=11)
        layer_queue_operations = deque(maxlen=11)
        for k in layers.keys():
            layer_queue_operations.append(layers[k]['operation'])
            layer_queue.append(k)
            if list(layer_queue_operations) == layer_type_1:
                final_layers.append(list(layer_queue))
            if list(layer_queue_operations)[:-1] == layer_type_2:
                final_layers.append(list(layer_queue)[:-1])
    return final_layers


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 parse model')
    parser.add_argument('model_name', help='name of the har model')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--use_frames', action='store_true', help='whether to use video decoder or raw frame decoder')
    parser.add_argument('--calculate_pareto', action='store_true', help='whether to calculate and plot pareto front')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--label', default=None, help='label file')
    parser.add_argument('--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument('--imshape', type=int, nargs="+", default=[224, 224, 3], help='image size for inference')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    fname = args.model_name + '_onnx'
    fname_pareto = fname + "_pareto"

    '''
        The ZCU102 has BRAM size equal to 32.1 Mbits (4.0125 MBytes). This divided by the 18 Kbits size of each BRAM gives a total of 1825 BRAM units.
        The ZCU102 has 4Gb DDR4 memory clocked at 1200MHz with 16-bit bus width. This leads to a total of 18.75 Gb/s or 2.35 GB/s.
        The ZCU102 has a total of 2520 DSP slices
    '''
    '''
        The ZCU104 has BRAM size equal to 11 Mbits (1.375 MBytes). This divided by the 18 Kbits size of each BRAM gives a total of 624 BRAM units.
        The ZCU104 has also 27 Mbits (3.375 MBytes) of URAM. This divided by the 288 Kbits size of each URAM gives a total of 96 URAM units.
        The ZCU104 has 4Gb DDR4 memory clocked at 1200MHz with 16-bit bus width. This leads to a total of 18.75 Gb/s or 2.35 GB/s.
        The ZCU104 has a total of 1728 DSP slices
    '''
    # Target FPGA Zynq UltraScale+ MPSoC ZCU102. Assuming clock frequency of 100 MHz. The mem bandwidth used is 80% of its nominal value.
    onnx_modeling = ModelFeatureMapsOnnx(model=args.model_name, word_length=16, clock_freq=100, bram=1825, dsp=2520, mem_bw=18.8)


    onnx_modeling.from_onnx()
    onnx_modeling.get_info()
    exit()
    onnx_modeling.create_modules()

    ray.init(num_cpus=10)
    onnx_modeling.create_design_points(file_name=fname, s_in=onnx_modeling.max_words_per_cycle*0.5, s_out=onnx_modeling.max_words_per_cycle*0.5)
    drop_duplicates(file_name=fname, pareto=False)
    get_paretto(file_name=fname)
    drop_duplicates(file_name=fname, pareto=True)
    # performance_graphs(file_name=fname, layers_to_plot=['Conv', 'Se', 'GlobalAveragePool', 'MatMul', 'Gemm'], calculate_pareto=args.calculate_pareto)
    ray.shutdown()


    partition_layers = get_partition_layers(onnx_modeling.modules, args.model_name)


    ray.init(num_cpus=10)
    # processes_pool = Pool(10)
    # input_vars = []
    result_ids = []
    for n, l in enumerate(partition_layers):
        print("Evaluating Layer {}/{}".format(n+1, len(partition_layers)))
        # input_vars.append([fname_pareto, l, n+1, fname, args.calculate_pareto, onnx_modeling.max_words_per_cycle, False, False])
        result_ids.append(onnx_modeling.compose_layers.remote(onnx_modeling, file_name=fname_pareto, layers_names=l, final_name=n+1, model_name=fname, calculate_pareto=args.calculate_pareto, membw=onnx_modeling.max_words_per_cycle, branch_on_bram=False, plot_design_points=False))
        # onnx_modeling.compose_layers(fname_pareto, l, n+1, fname, args.calculate_pareto, onnx_modeling.max_words_per_cycle, branch_on_bram=False, plot_design_points=False)
    # multithreaded_modeling(onnx_modeling.compose_layers, input_vars, processes_pool)
    results = ray.get(result_ids)
    plot_best_config_params(file_name=args.model_name)
    ray.shutdown()

if __name__ == '__main__':
    main()