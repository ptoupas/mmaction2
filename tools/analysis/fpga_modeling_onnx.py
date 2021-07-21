import argparse
import csv
import os
import sys
import math
import coloredlogs
import logging
import onnx
import json
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

coloredlogs.install(level='WARNING')
logging.basicConfig(level=logging.WARNING)
np.set_printoptions(precision=5, suppress=True, linewidth=150)


class ModelFeatureMapsOnnx():

    def __init__(self, model, word_length, clock_freq, bram, dsp, mem_bw):
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
        onnx.checker.check_model(self.onnx_model)

        # print(onnx.helper.printable_graph(self.onnx_model.graph))

    def thread_helper(self, arguments):
        self.compose_layers(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6])

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

    def get_shape_onnx(self, input):
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

    def from_onnx(self):

        layers_outputs = {}
        first_layer = True
        self.input_shape = self.get_shape_onnx(self.onnx_model.graph.input[0])[1:]
        logging.info("Model input shape = {}".format(self.input_shape))
                
        for n in self.onnx_model.graph.node:
            if n.op_type in self.op_list:

                logging.info("Node ({}):\n{}".format(n.name, n.input))
                
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
                    for layer_in in n.input:
                        exists, shape = self.is_in_inputs(layer_in)
                        if exists:
                            logging.info("VARIABLE INPUT {} - {}".format(layer_in, shape))
                            if 'weight' in layer_in:
                                kernel = shape.copy()
                            if 'bias' in layer_in:
                                bias = shape.copy()
                            if 'running_mean' in layer_in:
                                running_mean = shape.copy()
                            if 'running_var' in layer_in:
                                running_var = shape.copy()
                        else:
                            layer_input_shape.append(self.input_shape.copy())
                            layer_input_id.append(layer_in)
                            logging.info("INTERMEDIADE INPUT {} - {}".format(layer_in, self.input_shape))
                    first_layer = False
                else:
                    for layer_in in n.input:
                        exists, shape = self.is_in_inputs(layer_in)
                        if exists:
                            logging.info("VARIABLE INPUT {} - {}".format(layer_in, shape))
                            if 'weight' in layer_in:
                                kernel = shape.copy()
                            if 'bias' in layer_in:
                                bias = shape.copy()
                            if 'running_mean' in layer_in:
                                running_mean = shape.copy()
                            if 'running_var' in layer_in:
                                running_var = shape.copy()
                        else:
                            if layer_in not in layers_outputs.keys():
                                skip_layer = True
                                break
                            layer_input_shape.append(layers_outputs[layer_in])
                            layer_input_id.append(layer_in)
                            logging.info("INTERMEDIADE INPUT {} - {}".format(layer_in, layers_outputs[layer_in]))
                if skip_layer:
                    logging.warning("Could not find the input of layer {}. This layer will be skipped in the analysis".format(n.name))
                    continue
                out_shape = []
                if n.op_type == 'Conv':
                    dilation = n.attribute[0].ints
                    groups = n.attribute[1].i
                    padding = n.attribute[3].ints[:3]
                    stride = n.attribute[4].ints

                    out_shape = self.calculate_conv_out_shape(layer_input_shape[0].copy(), groups, dilation, kernel, padding, stride)
                elif 'Pool' in n.op_type:
                    if n.op_type == 'GlobalAveragePool':
                        out_shape = layer_input_shape[0].copy()
                        out_shape[2] = 1
                        out_shape[3] = 1
                        out_shape[4] = 1
                    else:
                        kernel_shape = n.attribute[1].ints
                        padding = n.attribute[2].ints[:3]
                        stride = n.attribute[3].ints

                        in_shape = layer_input_shape[0].copy()
                        dout = math.floor((in_shape[2] + 2 * padding[0] - (kernel_shape[0] - 1) - 1)/stride[0] + 1)
                        hout = math.floor((in_shape[3] + 2 * padding[1] - (kernel_shape[1] - 1) - 1)/stride[1] + 1)
                        wout = math.floor((in_shape[4] + 2 * padding[2] - (kernel_shape[2] - 1) - 1)/stride[2] + 1)
                            
                        out_shape.append(in_shape[0])
                        out_shape.append(in_shape[1])
                        out_shape.append(dout)
                        out_shape.append(hout)
                        out_shape.append(wout)
                elif n.op_type == 'Mul' or n.op_type == 'Add' or n.op_type == 'Div':
                    inputs_curr = np.array(layer_input_shape)
                    shape_idx = np.argmax(np.prod(inputs_curr, axis=1))
                    out_shape = layer_input_shape[shape_idx].copy()
                elif n.op_type == 'MatMul' or n.op_type == 'Gemm':
                    logging.warning('Cannot connect with previous layers due to lack of support of some operations like squeeze, reshape etc.')
                    continue
                else:
                    out_shape = layer_input_shape[0].copy()
                
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
    
    def get_info(self):
        for k in self.layers.keys():
            logging.info("Node ({}):\n{}".format(k, self.layers[k]))

    def batchnorm_layer_config(self, in_shape, s_in=1, s_out=1):
        cin = in_shape[1]

        rate_in = 1 * s_in
        rate_out = 1 * s_in
        mem = cin * 4
        muls = 1 * s_in
        adds = 3 * s_in

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
        adds = 0
        mem = 0

        return rate_in, rate_out, muls, adds, mem

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
        rate_out = 1/(din * hin * win) * gap_folding
        mem = cin
        muls = 2 * gap_folding
        adds = 1 * gap_folding
        depth = cin * din * hin * win

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

        return rate_in, rate_out, muls, adds, mem, depth, (mem_bounded_in, mem_bounded_out)

    def swish_layer_config(self, s_in=1, s_out=1):
        rate_in = 1 * s_in
        rate_out = 1 * s_in
        mem = 0
        muls = 4 * s_in
        adds = 1 * s_in

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

        cin = kernel_shape[1] * groups
        cout = kernel_shape[0]

        muls_unrl = kd * kh * kw * fine
        adds_unrl_1 = (kd * kh * kw - 1 ) * fine
        adds_unrl_2 = 1

        depthwise = False
        if cout == groups:
            depthwise = True

        rates_graph = np.zeros( shape=(4,5) , dtype=float )

        # The convolution operation is a Layer and is composed of the following modules: Sliding window, Conv, Accumulator 
        # Rates for the SW module
        if kd == 1 and kh == 1 and kw == 1:
            rin_sw = 1
            rout_sw = (dout*hout*wout)/(din*hin*win)
        else:
            rin_sw = 1
            rout_sw = (dout*hout*wout)/(din*hin*win)
        rates_graph[0,0] = rin_sw * coarse_in
        rates_graph[0,1] = rout_sw * (kd * kh * kw) * coarse_in

        rates_graph[1,1] = 1 * rates_graph[0,1]
        rates_graph[1,2] = 1 * rates_graph[0,1] #* coarse_out

        # Rates for the Conv module
        rin_conv = (fine * groups * rates_graph[1,2] * coarse_out)/cout
        rout_conv = fine * rates_graph[1,2] * coarse_out
        rates_graph[2,2] = rin_conv
        rates_graph[2,3] = rout_conv / (kd * kh * kw)
        
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
        rate_out = abs(rates_graph[3,4])

        mem_bounded_out = False
        if mem_bw_out < rate_out:
            mem_bounded_out = True
            rate_out = mem_bw_out

        if kd == 1 and kh == 1 and kw == 1:
            pb = 1
            sw_depth = min(((din*win+padding[0]+padding[1]+padding[2])*cin*kh)+cin*kh*kh, ((win*hin+padding[0]+padding[1]+padding[2])*cin*kd)+cin*kd*kd)
        else:
            # Plane buffer + Line buffer (needed in conjuction with plane buffer)
            pb = min((din*win*kh), (win*hin*kd)) + min((din*kw), (win*kh))
            sw_depth = min(((din*win+padding[0]+padding[1]+padding[2])*cin*(kh-1))+cin*kh*(kh-1), ((win*hin+padding[0]+padding[1]+padding[2])*cin*(kd-1))+cin*kd*(kd-1))
        kernel_size = int(np.prod(np.array(kernel_shape)))

        conv_depth = math.ceil(1/fine)
        acc_depth = cin*cout
        if not depthwise:
            depth = sw_depth + conv_depth + acc_depth
        else:
            depth = sw_depth + conv_depth
        mem = pb + kernel_size + cin

        muls = math.ceil(muls_unrl * coarse_in * coarse_out)
        #TODO: This calculations are not correct. Need revision.
        adds = math.ceil(adds_unrl_1 * coarse_in * coarse_out) + math.ceil(adds_unrl_2 * coarse_in * coarse_out)
        return rate_in, rate_out, muls, adds, mem, depth, (mem_bounded_in, mem_bounded_out)

    def se_layer_config(self, glavpool_in_shape, glavpool_coarse, conv1_in_shape, conv1_out_shape, conv1_kernel_shape, conv1_padding, conv1_groups, fine1, coarse_in_1, coarse_out_1, conv2_in_shape, conv2_out_shape, conv2_kernel_shape, conv2_padding, conv2_groups, fine2, coarse_in_2, coarse_out_2, bw_in=1, bw_total=1, se_on_bram=1):
        
        mem_bw_in = bw_in
        mem_bw_total = bw_total

        glavpool_rate_in, glavpool_rate_out, glavpool_muls, glavpool_adds, glavpool_mem, glavpool_depth, (glavpool_mem_bounded_in, glavpool_mem_bounded_out) = self.gap_layer_config(glavpool_in_shape, coarse=glavpool_coarse, s_in=mem_bw_in, s_out=10000)
        glavpool_thr_in = (self.cycles_per_sec*glavpool_rate_in)/int(np.prod(np.array(glavpool_in_shape[1:])))
        glavpool_thr_out = (self.cycles_per_sec*glavpool_rate_out)/int(glavpool_in_shape[1])
        assert math.isclose(glavpool_thr_in, glavpool_thr_out), "Input and Output Throughputs doesnt match on glavpool. Aborting..."

        conv1_rate_in, conv1_rate_out, conv1_muls, conv1_adds, conv1_mem, conv1_depth, (conv1_mem_bounded_in, conv1_mem_bounded_out) = self.conv_layer_config(conv1_in_shape, conv1_out_shape, conv1_kernel_shape, conv1_padding, conv1_groups, fine1, coarse_in_1, coarse_out_1, s_in=glavpool_rate_out, s_out=10000)
        conv1_thr_in = (self.cycles_per_sec*conv1_rate_in)/int(np.prod(np.array(conv1_in_shape[1:])))
        conv1_thr_out = (self.cycles_per_sec*conv1_rate_out)/int(np.prod(np.array(conv1_out_shape[1:])))
        assert math.isclose(conv1_thr_in, conv1_thr_out), "Input and Output Throughputs doesnt match on conv1. Aborting..."

        relu_rate_in, relu_rate_out, _, _, _ = self.relu_layer_config(s_in=conv1_rate_out)

        conv2_rate_in, conv2_rate_out, conv2_muls, conv2_adds, conv2_mem, conv2_depth, (conv2_mem_bounded_in, conv2_mem_bounded_out) = self.conv_layer_config(conv2_in_shape, conv2_out_shape, conv2_kernel_shape, conv2_padding, conv2_groups, fine2, coarse_in_2, coarse_out_2, s_in=relu_rate_out, s_out=10000)
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
        return rate_in, rate_out, muls, adds, mem, depth, (mem_bounded_in, mem_bounded_out)

    def get_layer_from_id(self, layer_id):
        for k in self.layers.keys():
            if layer_id == int(self.layers[k]['output_id']):
                return self.layers[k], k
        return None, None

    def create_modules(self):
        se_module = deque(maxlen=6)
        swish_module = deque(maxlen=3)
        prev_output_id = -1
        for k in self.layers.keys():
            curr_output_id = int(self.layers[k]['output_id'])
            if not prev_output_id == -1:
                assert curr_output_id == prev_output_id + 1, "Modules are not in the correct order. Revise the graph creation"
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
                in1_layer, in1_layer_name = self.get_layer_from_id(int(self.layers[k]['input_id'][0]))
                oldest_input = int(in1_layer['output_id']) if in1_layer is not None else int(self.layers[k]['output_id'])
                logging.info("Layer name = {} ({}). Input = {} ({})".format(name, self.layers[k]['output_id'], in1_layer_name, self.layers[k]['input_id'][0]))
            #TODO: Should find a more generic way to detect and filter branching behaviours on networks.
            if int(self.layers[k]['output_id']) - oldest_input > 2:
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

    def model_layer(self, layer, csv_writer, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem, mem_bounded_out=False, config=None, inter_size=None, buffering_enabled=False, module_mem_bw_in=-1):
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
            csv_writer.writerow([layer, folding, mem_util, dsps_util, thr_in, thr_out, bw_in_w, bw_out_w, mem_kb, mem_bram, bw_in_gb, bw_out_gb, muls, adds, dsps, thr_w_out, thr_go, l_config])

            logging.info("On Chip Mem (BRAM %) = {:<15.5f} DSPS % = {:<20.5f}\nConsumption(inputs/sec) = {:<20.5f} Throughtput(outputs/sec) = {:<20.5f}\nMemory Bandwidth In(words/cycle) = {:<20.5f} Memory Bandwidth Out(words/cycle) = {:<20.5f}\nOn-Chip Memory(KB) = {:<20.5f} On-Chip Memory(BRAM) = {:<20.5f} Adds = {:<20.5f} DSPS = {:<20.5f}\nMemory Bandwidth In(GBs/sec) = {:<20.5f} Memory Bandwidth Out(GBs/sec) = {:<20.5f}\nThroughtput(GOps/sec) = {:.3f}".format(mem_util, dsps_util, thr_in, thr_out, bw_in_w, bw_out_w, mem_kb, mem_bram, adds, dsps, bw_in_gb, bw_out_gb, thr_go))
        else:
            logging.warning("Design point dropped because of too many recourses needed. DSPS = {} ({}%). BRAM = {} ({}%)".format(dsps, dsps_util, mem_bram, mem_util))

    def create_design_points(self, file_name, s_in=1, s_out=1):
            if not os.path.exists(os.path.join(os.getcwd(), 'fpga_modeling_reports')):
                os.makedirs(os.path.join(os.getcwd(), 'fpga_modeling_reports'))
            csv_file = os.path.join(os.getcwd(), 'fpga_modeling_reports', file_name + '.csv')
            with open(csv_file, mode='w') as model_results:
                csv_writer = csv.writer(model_results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                csv_writer.writerow(["Layer", "Folding", "On-Chip Memory(BRAM %)", "DSPS %", "Consumption(inputs/sec)", "Throughtput(outputs/sec)", "Memory Bandwidth In(words/cycle)", "Memory Bandwidth Out(words/cycle)", "On-Chip Memory(KB)", "On-Chip Memory(BRAM)", "Memory Bandwidth In(GBs/sec)", "Memory Bandwidth Out(GBs/sec)", "Multipliers", "Adders", "DSPS", "Throughtput(words/cycle)", "Throughtput(GOps/sec)", "Configuration"])

                for k in self.modules.keys():
                    name = k
                    operation = self.modules[k]['operation']
                    logging.warning("Layer: {} -> Operation: {}.".format(name, operation))

                    # if not operation == 'Conv' and not operation == 'SqueezeExcitation':
                    #     continue
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

                        cin = kernel_shape[1] * groups
                        cout = kernel_shape[0]

                        in_size = cin * din * hin * win
                        out_size = cout * dout * hout * wout

                        pr_name = name

                        # if cout == groups:
                        #     pr_name = pr_name + "_DepthWise"
                        # if kd == 1 and kh == 1 and kw == 1:
                        #     pr_name = pr_name + "_PointWise"

                        coarse_in_config = [1, (cin)//4, (cin)//2, cin]
                        coarse_in_config = np.unique(coarse_in_config)
                        coarse_in_config = coarse_in_config[np.nonzero(coarse_in_config)].tolist()
                        coarse_out_config = [1, cout//4, cout//2, cout]
                        coarse_out_config = np.unique(coarse_out_config)
                        coarse_out_config = coarse_out_config[np.nonzero(coarse_out_config)].tolist()
                        max_fine = kd * kh * kw
                        fine_config = np.array([kd/max_fine, kh/max_fine, kw/max_fine, (kd * kh)/max_fine, (kh * kw)/max_fine, (kd * kw)/max_fine, 1])
                        fine_config = np.unique(fine_config).tolist()
                        if kd == 1 and kh == 1 and kw == 1:
                            fine_config = [0.5, 1]
                        
                        mem_bw_config = [((s_in+s_out)*0.25, (s_in+s_out)*0.75), ((s_in+s_out)*0.50, (s_in+s_out)*0.50), ((s_in+s_out)*0.75, (s_in+s_out)*0.25)]
                        for coarse_in in coarse_in_config:
                            for coarse_out in coarse_out_config:
                                for fine in fine_config:
                                    for conv_bw_in, conv_bw_out in mem_bw_config:
                                        coarse_in_name = str(coarse_in)
                                        coarse_out_name = str(coarse_out)
                                        folding_name = "N_Coarse({}/{}) - f_Fine({:.2f}) - Mem BW({:.2f}/{:.2f})".format(coarse_in_name, coarse_out_name, fine, conv_bw_in, conv_bw_out)

                                        logging.info("Fold = {}. Channels In = {} - Filters = {}".format(folding_name, cin, cout))

                                        rate_in, rate_out, muls, adds, mem, depth, (mem_bounded_in, mem_bounded_out) = self.conv_layer_config(in_shape, out_shape, kernel_shape, padding, groups, fine, coarse_in, coarse_out, s_in=conv_bw_in, s_out=conv_bw_out)

                                        current_config = [in_shape, out_shape, kernel_shape, padding, groups, fine, coarse_in, coarse_out, int(mem_bounded_in), int(mem_bounded_out)]
                                        self.model_layer(pr_name, csv_writer, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem, mem_bounded_out=mem_bounded_out, config=current_config)

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
                                                            
                                                            rate_in, rate_out, muls, adds, mem, depth, (mem_bounded_in, mem_bounded_out) = self.se_layer_config(glavpool_in_shape, coarse_gap, conv1_in_shape, conv1_out_shape, conv1_kernel_shape, conv1_padding, conv1_groups, fine_1, coarse_in_1, coarse_out_1, conv2_in_shape, conv2_out_shape, conv2_kernel_shape, conv2_padding, conv2_groups, fine_2, coarse_in_2, coarse_out_2, bw_in=se_bw_in, bw_total=mem_bw, se_on_bram=se_on_bram)
            
                                                            #TODO: Added worst possible case for buffering on se module i.e., buffer the whole feature map and all of the channels. Should fix this by checking the depth/latency of the left branch in order to calculate the exact buffering that is gonna needed in each se module.
                                                            #TODO: Another solution is to read again from off-chip memory which will prevent the buffering i.e., reduce the BRAM needs BUT will reduce the mem bw in total as well since we need to first write the results (in a bigger layer-wise partition) and the read them again i.e., will probably need to have mem_bw / 4 instead of mem_bw / 2 in each point that we access the off-chip memory.

                                                            current_config = [glavpool_in_shape, coarse_gap, conv1_in_shape, conv1_out_shape, conv1_kernel_shape, conv1_padding, conv1_groups, fine_1, coarse_in_1, coarse_out_1, conv2_in_shape, conv2_out_shape, conv2_kernel_shape, conv2_padding, conv2_groups, fine_2, coarse_in_2, coarse_out_2, se_on_bram, int(mem_bounded_in), int(mem_bounded_out)]

                                                            self.model_layer(name, csv_writer, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem, mem_bounded_out=mem_bounded_out, config=current_config, inter_size=br_size, buffering_enabled=se_on_bram, module_mem_bw_in=se_bw_in)

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

                        self.model_layer(name, csv_writer, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem, config=[in_shape])

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

                        self.model_layer(name, csv_writer, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem)

                    elif operation == 'GlobalAveragePool':
                        out_shape = self.modules[name]['shape_out']
                        cout = out_shape[1]
                        out_size = int(np.prod(np.array(out_shape[1:])))

                        in_shape = self.modules[name]['shape_in']
                        cin = in_shape[1]
                        in_size = int(np.prod(np.array(in_shape[1:])))

                        assert cin == cout, 'Input and output channels bust be identical in GlobalAveragePool Layer'

                        logging.info("Fold = {}. Channels In = {} - Filters = {}".format(folding_name, cin, cout))

                        gap_config = [1, (cin)//16, (cin)//12, (cin)//8, (cin)//4]

                        for gap_coarse in gap_config:
                            rate_in, rate_out, muls, adds, mem, depth, (mem_bounded_in, mem_bounded_out) = self.gap_layer_config(in_shape, coarse=gap_coarse, s_in=s_in, s_out=s_out)

                            folding_name = "Coarse({:.2f}) - Mem_Bw({}/{})".format(gap_coarse, s_in, s_out)

                            self.model_layer(name, csv_writer, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem, mem_bounded_out=mem_bounded_out, config=[in_shape, gap_coarse, int(mem_bounded_in), int(mem_bounded_out)])

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

                        self.model_layer(name, csv_writer, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem)

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

                        self.model_layer(name, csv_writer, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem)
                    

    def product_dict(self, **kwargs):
        keys = kwargs.keys()
        vals = kwargs.values()
        for instance in itertools.product(*vals):
            yield dict(zip(keys, instance))

    def get_rates(self, layer, config, bw_in, bw_total, bw_out):
        operation = layer.split("_")[0]
        tmp_thr_in, tmp_thr_out = 0, 0
        if operation == 'Conv':
            rate_in, rate_out, muls, adds, mem, depth, (mem_bounded_in, mem_bounded_out) = self.conv_layer_config(config[0], config[1], config[2], config[3], config[4], config[5], config[6], config[7], s_in=bw_in, s_out=bw_out)
            tmp_thr_in = (self.cycles_per_sec*rate_in)/int(np.prod(np.array(config[0][1:])))
            tmp_thr_out = (self.cycles_per_sec*rate_out)/int(np.prod(np.array(config[1][1:])))
            assert math.isclose(tmp_thr_in, tmp_thr_out), "Input and Output Throughputs doesnt match on CONV operation. Aborting..."
        elif operation == 'Se':
            rate_in, rate_out, muls, adds, mem, depth, (mem_bounded_in, mem_bounded_out) = self.se_layer_config(config[0], config[1], config[2], config[3], config[4], config[5], config[6], config[7], config[8], config[9], config[10], config[11], config[12], config[13], config[14], config[15], config[16], config[17], bw_in=bw_in, bw_total=bw_total, se_on_bram=config[18])
            tmp_thr_in = (self.cycles_per_sec*rate_in)/int(np.prod(np.array(config[0][1:])))
            tmp_thr_out = (self.cycles_per_sec*rate_out)/int(np.prod(np.array(config[0][1:])))
            assert math.isclose(tmp_thr_in, tmp_thr_out), "Input and Output Throughputs doesnt match on SE operation. Aborting..."
        elif operation == 'GlobalAveragePool':
            rate_in, rate_out, muls, adds, mem, depth, (mem_bounded_in, mem_bounded_out) = self.gap_layer_config(config[0], coarse=config[1], s_in=bw_in)
            tmp_thr_in = (self.cycles_per_sec*rate_in)/int(np.prod(np.array(config[0][1:])))
            tmp_thr_out = (self.cycles_per_sec*rate_out)/int(np.prod(np.array(config[0][1])))
            assert math.isclose(tmp_thr_in, tmp_thr_out), "Input and Output Throughputs doesnt match on GlobalAveragePool operation. Aborting..."
        elif operation == 'Relu':
            rate_in, rate_out, muls, adds, mem = self.relu_layer_config(s_in=bw_in)
            assert math.isclose(rate_in, rate_out), "Input and Output Rates doesnt match on ReLu operation. Aborting..."
            depth = 1
        elif operation == 'BatchNormalization':
            rate_in, rate_out, muls, adds, mem = self.batchnorm_layer_config(config[0], s_in=bw_in)
            assert math.isclose(rate_in, rate_out), "Input and Output Rates doesnt match on BatchNormalization operation. Aborting..."
            depth = 1
        elif operation == 'Swish':
            rate_in, rate_out, muls, adds, mem = self.swish_layer_config(s_in=bw_in)
            assert math.isclose(rate_in, rate_out), "Input and Output Rates doesnt match on Swish operation. Aborting..."
            depth = 1
        elif operation == 'Sigmoid':
            rate_in, rate_out, muls, adds, mem = self.sigmoid_layer_config(s_in=bw_in)
            assert math.isclose(rate_in, rate_out), "Input and Output Rates doesnt match on Sigmoid operation. Aborting..."
            depth = 1
        elif operation == 'Add':
            rate_in, rate_out, muls, adds, mem = self.add_layer_config(s_in=bw_in)
            assert math.isclose(rate_in, rate_out), "Input and Output Rates doesnt match on Add operation. Aborting..."
            depth = 1
        elif operation == 'Mul':
            rate_in, rate_out, muls, adds, mem = self.mul_layer_config(s_in=bw_in)
            assert math.isclose(rate_in, rate_out), "Input and Output Rates doesnt match on Mul operation. Aborting..."
            depth = 1

        return rate_in, rate_out, muls, adds, mem, depth, tmp_thr_in, tmp_thr_out

    def compose_layers(self, file_name, layers_names, final_name, model_name, calculate_pareto, membw, branch_on_bram):
        sns.set(rc={'figure.figsize':(15,8)})
        sns.set_style("darkgrid", {"axes.facecolor": ".85"})

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
                        if l.split("_")[0] == 'Conv' or l.split("_")[0] == 'Se' or l.split("_")[0] == 'BatchNormalization':
                            l_configs[l].append([row[cols['Folding']], json.loads(row[cols['Configuration']])])
                        else:
                            l_configs[l].append([row[cols['Folding']], row[cols['Configuration']]])
        
        sizes = []
        keys = []
        for k in l_configs.keys():
            sizes.append(len(l_configs[k]))
            keys.append(k)
        
        dsp_config = []
        bram_config = []
        bram_total_util = []
        mem_bw_status = []
        throughput_config = []
        
        res = list(self.product_dict(**l_configs))
        
        for r in tqdm(res, leave=False):
            layer_keys = list(r.keys())
            se_layer_key = None
            se_layer_bwin = 0
            se_layer_bwout = 0

            branches_points = [i for i in range(len(layer_keys)) if self.modules[layer_keys[i]]['branching'] and not layer_keys[i].split("_")[0] == "Se"]

            # in_shape = self.modules[layer_keys[0]]['shape_in']
            in_shape = self.modules[layer_keys[-1]]['shape_in']
            in_size = int(np.prod(np.array(in_shape[1:])))
            out_shape = self.modules[layer_keys[-1]]['shape_out']
            out_size = int(np.prod(np.array(out_shape[1:])))

            total_depth = 0
            total_mem = 0
            total_muls = 0
            total_adds = 0

            membw_config = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            mem_on_chip_bw = 10000
            
            for mem_bw_in in membw_config:
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
                                mod_rin, mod_rout, mod_muls, mod_adds, mod_mem, mod_depth, mod_thrin, mod_throut = self.get_rates(k, r[k][-1], mem_bw_in, membw, mem_on_chip_bw)
                            else:
                                mod_rin, mod_rout, mod_muls, mod_adds, mod_mem, mod_depth, mod_thrin, mod_throut = self.get_rates(k, r[k][-1], mem_on_chip_bw, membw, mem_on_chip_bw)
                        else:
                            mod_rin, mod_rout, mod_muls, mod_adds, mod_mem, mod_depth, mod_thrin, mod_throut = self.get_rates(k, r[k][-1], prev_mod_rout, membw, mem_on_chip_bw)
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
                        branch_mem_bw = mem_bw_in

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
                    mod_rin, mod_rout, mod_muls, mod_adds, mod_mem, mod_depth, mod_thrin, mod_throut = self.get_rates(final_layer, r[final_layer][-1], rate_in, membw, mem_on_chip_bw)
                    
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
                    mod_rin, mod_rout, mod_muls, mod_adds, mod_mem, mod_depth, mod_thrin, mod_throut = self.get_rates(final_layer, r[final_layer][-1], rate_in, membw, mem_on_chip_bw)
                    
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
                dsp_config.append(dsps_util)
                bram_total_util.append(bram_util)
                throughput_config.append(thr_out)

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
        model_name = file_name.split("onnx")[0]
        csv_file = os.path.join(os.getcwd(), 'fpga_modeling_reports', model_name + 'max_throughput_design_points.csv')
        max_throughput = 0
        best_dsp = 0
        best_bram = 0
        best_bw_stat = "Unknown"
        for thr, dsp, bram, bw_stat in zip(throughput_config, dsp_config, bram_total_util, mem_bw_status):
            if thr > max_throughput and dsp < 90.0 and bram < 90.0:
                max_throughput = thr
                best_dsp = dsp
                best_bram = bram
                best_bw_stat = bw_stat
            if thr == max_throughput:
                if (dsp < best_dsp and bram <= best_bram) or (dsp <= best_dsp and bram < best_bram):
                    best_dsp = dsp
                    best_bram = bram
                    best_bw_stat = bw_stat
        with open(csv_file, mode='a') as model_results:
            csv_writer = csv.writer(model_results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if final_name == 1:
                csv_writer.writerow(['Layer Name', 'Throughput', 'DSPs(%)', 'BRAM(%)', 'Memory BW Status'])
            csv_writer.writerow([final_name, max_throughput, best_dsp, best_bram, best_bw_stat])
        logging.warning("Best config for layer {}. Throughput = {:.5f}, DSPs(%) = {:.5f}, BRAM(%) = {:.5f}, Mem BW = {}".format(final_name, max_throughput, best_dsp, best_bram, best_bw_stat))

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
    sns.set(rc={'figure.figsize':(15,8)})
    sns.set_style("darkgrid", {"axes.facecolor": ".85"})
    
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
        #         plt.legend(legd, frameon=False, prop={"size":8}, loc='upper right', bbox_to_anchor=(1.11, 1.12), borderaxespad=0.)
        #     else:
        #         plt.legend([],[], frameon=False)
        # else:
        plt.legend(frameon=False, prop={"size":8}, loc='upper right', bbox_to_anchor=(1.11, 1.12), borderaxespad=0.)

        if se_layer and calculate_pareto:
            logging.info(name)
            for l_i, l in enumerate(legd):
                logging.info("Config: {} -> Throughput: {}. DSPs: {}".format(l, pareto_front[l_i, 0], pareto_front[l_i, 1]))
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
    columns = data.columns.tolist()
    del(columns[1])
    del(columns[-1])
    del(columns[-1])

    data_droped = data.drop_duplicates(subset=columns)
    os.remove(csv_file_read)
    data_droped.to_csv(csv_file_read, index=False)

def get_paretto(file_name="x3d_m"):
    
    csv_file_par = os.path.join(os.getcwd(), 'fpga_modeling_reports', file_name + '_pareto.csv')
    with open(csv_file_par, mode='w') as pareto_results:
        csv_writer_par = csv.writer(pareto_results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csv_writer_par.writerow(["Layer", "Folding", "On-Chip Memory(BRAM %)", "DSPS %", "Consumption(inputs/sec)", "Throughtput(outputs/sec)", "Memory Bandwidth In(words/cycle)", "Memory Bandwidth Out(words/cycle)", "On-Chip Memory(KB)", "On-Chip Memory(BRAM)", "Memory Bandwidth In(GBs/sec)", "Memory Bandwidth Out(GBs/sec)", "Multipliers", "Adders", "DSPS", "Throughtput(words/cycle)", "Throughtput(GOps/sec)", "Configuration"])

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
    parser.add_argument('config', help='test config file path')
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

    # Target FPGA Zynq UltraScale+ MPSoC ZCU104. Assuming clock frequency of 100 MHz.
    # The actual BRAM size is 11 Mbits (1.375 MBytes). This divided by the 18 Kbits size of each BRAM gives a total of 624 BRAM units.
    # The ZCU104 has also 27 Mbits (3.375 MBytes) of URAM. This divided by the 288 Kbits size of each URAM gives a total of 96 URAM units.
    # The ZCU104 has 20 GTH gigabit transceivers (16.3 Gb/s or 2.03 GB/s) on the PL-size
    onnx_modeling = ModelFeatureMapsOnnx(model=args.model_name, word_length=16, clock_freq=100, bram=624, dsp=1728, mem_bw=16.3)

    onnx_modeling.from_onnx()

    # onnx_modeling.get_info()

    onnx_modeling.create_modules()

    onnx_modeling.create_design_points(file_name=fname, s_in=onnx_modeling.max_words_per_cycle*0.5, s_out=onnx_modeling.max_words_per_cycle*0.5)
    drop_duplicates(file_name=fname, pareto=False)
    get_paretto(file_name=fname)
    drop_duplicates(file_name=fname, pareto=True)

    # partition_layers = get_partition_layers(onnx_modeling.modules, args.model_name)
    # for n, l in enumerate(partition_layers):
    #     print("Evaluating Layer {}/{}".format(n+1, len(partition_layers)))
    #     onnx_modeling.compose_layers(fname_pareto, l, n+1, fname, args.calculate_pareto, onnx_modeling.max_words_per_cycle, branch_on_bram=False)

    # performance_graphs(file_name=fname, layers_to_plot=['Conv', 'Se', 'GlobalAveragePool'], calculate_pareto=args.calculate_pareto)

if __name__ == '__main__':
    main()