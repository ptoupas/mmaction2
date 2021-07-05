import argparse
import csv
import os
import math
import coloredlogs
import logging
import onnx
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

coloredlogs.install(level='INFO')
np.set_printoptions(precision=5, suppress=True, linewidth=120)


class ModelFeatureMapsOnnx():

    def __init__(self, model, word_length, clock_freq, bram, dsp, mem_bw):
        self.model_path = model + ".onnx"
        self.layers = {}
        self.modules = {}
        self.wl = word_length
        self.wb = word_length//8
        self.clock_freq = clock_freq # in MHz
        self.cycles_per_sec = clock_freq*1e6
        self.clock_period = (1/(clock_freq*1e6))*1e9 # in nanosec
        self.bram_mem = 18 # Bram size is 18 Kbits or 2.25 KBytes
        self.fpga_bram = bram # each can hold a total of 18 Kbits or 2.25 KBytes
        self.fpga_dsps = dsp
        self.mem_bandwidth = mem_bw * 1e9 # in b/s (bits per second)
        self.max_words_per_cycle = (self.mem_bandwidth / self.wl) // self.cycles_per_sec

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
                    logging.error("Couldn't read the dimensions of tensor")
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

                logging.warning("Node ({}):\n{}".format(n.name, n.input))
                
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
                    logging.error("Could not find the input of layer {}. This layer will be skipped in the analysis".format(n.name))
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
                    logging.error('Cannot connect with previous layers due to lack of support of some operations like squeeze, reshape etc.')
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

    def conv_layer_config(self, module, fine, coarse_in, coarse_out, s_in=1, s_out=1):

        mem_bw_in = s_in
        mem_bw_out = s_out

        in_shape = module['shape_in']
        din = in_shape[2]
        hin = in_shape[3]
        win = in_shape[4]

        out_shape = module['shape_out']
        dout = out_shape[2]
        hout = out_shape[3]
        wout = out_shape[4]

        kernel_shape = module['kernel']
        kd = kernel_shape[2]
        kh = kernel_shape[3]
        kw = kernel_shape[4]

        cin = kernel_shape[1] * module['groups']
        cout = kernel_shape[0]

        muls_unrl = kd * kh * kw * fine
        adds_unrl_1 = kd * kh * kw * fine - 1
        adds_unrl_2 = 1

        depthwise = False
        if cout == module['groups']:
            depthwise = True

        if not depthwise:
            rates_graph = np.zeros( shape=(3,4) , dtype=float )
        else:
            rates_graph = np.zeros( shape=(2,3) , dtype=float )

        # The convolution operation is a Layer and is composed of the following modules: Sliding window, Conv, Accumulator 

        # Rates for the SW module
        if kd == 1 and kh == 1 and kw == 1:
            rin_sw = 1
            rout_sw = 1
        else:
            rin_sw = 1
            rout_sw = (dout*hout*wout)/(din*hin*win)
        rates_graph[0,0] = rin_sw * mem_bw_in
        rates_graph[0,1] = rout_sw * mem_bw_in

        # Rates for the Conv module
        # TODO: Check if we can add another layer of parallelization here above the coarse in/out. When the rate out from previous layer is greater than the rate in in conv can we parallelize more to increase the throughput further since we have the data to do it?
        rin_conv = fine/cout
        rout_conv = fine
        rates_graph[1,1] = rin_conv * coarse_out
        rates_graph[1,2] = rout_conv
        
        if not depthwise:
            # Rates for the Accumulator module
            rin_accum = 1
            rout_accum = 1/cin
            rates_graph[2,2] = rin_accum * mem_bw_out
            rates_graph[2,3] = rout_accum * coarse_in * mem_bw_out

            # print("CONV RATE GRAPH")
            # print(rates_graph)
            # print("-"*50)
            rates_graph = self.balance_module_rates(rates_graph)
            # print(rates_graph)
            # print("=="*50)
            rate_in = abs(rates_graph[0,0])
            rate_out = abs(rates_graph[2,3])
        else:
            # print("CONV RATE GRAPH (DW)")
            # print(rates_graph)
            # print("-"*50)
            rates_graph = self.balance_module_rates(rates_graph)
            # print(rates_graph)
            # print("=="*50)
            rate_in = abs(rates_graph[0,0])
            rate_out = abs(rates_graph[1,2])

        if not depthwise:
            out_in_ratio = (dout*hout*wout)/(din*hin*win)
            rate_in_tst = 1/cout * out_in_ratio * coarse_in * coarse_out * fine 
            rate_out_tst = 1/(cout*cin) * out_in_ratio * coarse_in * coarse_out * fine
        else:
            out_in_ratio = (dout*hout*wout)/(din*hin*win)
            rate_in_tst = 1/cout * out_in_ratio * coarse_in * coarse_out * fine 
            rate_out_tst = 1/cout * out_in_ratio * coarse_in * coarse_out * fine

        if mem_bw_in < rate_in_tst:
            logging.error("CONV OP: Memory bounded on reading input. Expected: {} - Max available: {}".format(rate_in_tst, mem_bw_in))
        if mem_bw_out < rate_out_tst:
            logging.error("CONV OP: Memory bounded on writing output. Expected: {} - Max available: {}".format(rate_out_tst, mem_bw_out))
        rate_in_tst = min(rate_in_tst, mem_bw_in)
        rate_out_tst = min(rate_out_tst, mem_bw_out)

        # print("Shape In (Din, Hin, Win) = ({}, {}, {})".format(din, hin, win))
        # print("Shape Out (Dout, Hout, Wout) = ({}, {}, {})".format(dout, hout, wout))
        # print("Rate in old = {:.5f}. Rate out old = {:.5f}.".format(rate_in, rate_out))
        # print("Rate in new = {:.5f}. Rate out new = {:.5f}.".format(rate_in_tst, rate_out_tst))

        rate_in = rate_in_tst
        rate_out = rate_out_tst

        if kd == 1 and kh == 1 and kw == 1:
            pb = 1
        else:
            # Plane buffer + Line buffer (needed in conjuction with plane buffer)
            pb = min((din*win*kh), (win*hin*kd)) + min((din*kw), (win*kh))
        kernel_size = int(np.prod(np.array(kernel_shape)))
        mem = pb + kernel_size + cin

        muls = math.ceil(muls_unrl * coarse_in * coarse_out)
        #TODO: This calculations are not correct. Need revision.
        adds = math.ceil(adds_unrl_1 * coarse_in * coarse_out) + math.ceil(adds_unrl_2 * coarse_in * coarse_out)
        return rate_in, rate_out, muls, adds, mem

    def se_layer_config(self, module, coarse_in_1, coarse_out_1, coarse_in_2, coarse_out_2, fine1, fine2, s_in=1, s_out=1):
        
        mem_bw_in = s_in
        mem_bw_out = s_out

        se_keys = list(module.keys())
        glavpool_key = se_keys[3]
        conv1_key = se_keys[4]
        conv2_key = se_keys[6]

        in_shape = module[glavpool_key]['shape_in']
        glavpool_rate_in = 1 * mem_bw_in
        glavpool_rate_out = 1/(in_shape[2]*in_shape[3]*in_shape[4]) * mem_bw_in
        glavpool_mem = in_shape[1]
        # TODO: Pass this into layer config
        glavpool_muls = 2 * mem_bw_in
        # glavpool_depth = 

        conv1_rate_in, conv1_rate_out, conv1_muls, conv1_adds, conv1_mem = self.conv_layer_config(module[conv1_key], coarse_in_1, coarse_out_1, fine1, s_in=glavpool_rate_out, s_out=10000)

        relu_rate_in = conv1_rate_out
        relu_rate_out = conv1_rate_out

        conv2_rate_in, conv2_rate_out, conv2_muls, conv2_adds, conv2_mem = self.conv_layer_config(module[conv2_key], coarse_in_2, coarse_out_2, fine2, s_in=relu_rate_out, s_out=10000)

        sigmoid_rate_in = conv2_rate_out
        sigmoid_rate_out = conv2_rate_out
        sigmoid_dsps = max(3, math.ceil(3 * conv2_rate_out))
        
        elemwise_mul_rate_in = sigmoid_rate_out
        elemwise_mul_rate_out = sigmoid_rate_out
        elemwise_mul_rate_dsps = max(1, math.ceil(sigmoid_rate_out))

        rates_graph = np.zeros( shape=(6,7) , dtype=float )
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

        rates_graph[5,5] = elemwise_mul_rate_in
        rates_graph[5,6] = elemwise_mul_rate_out 
        
        # print("SE RATE GRAPH")
        # print(rates_graph)
        # print("-"*50)
        rates_graph = self.balance_module_rates(rates_graph)
        # print(rates_graph)
        # print("=="*50)
        rate_in = abs(rates_graph[0,0])
        rate_out = abs(rates_graph[5,6])

        if mem_bw_in < rate_in:
            logging.error("SE OP: Memory bounded on reading input. Expected: {} - Max available: {}".format(rate_in, mem_bw_in))
        if mem_bw_out < rate_out:
            logging.error("SE OP: Memory bounded on writing output. Expected: {} - Max available: {}".format(rate_out, mem_bw_out))
        rate_in = min(rate_in, mem_bw_in)
        rate_out = min(rate_out, mem_bw_out)

        return rate_in, rate_out, glavpool_muls + conv1_muls + conv2_muls + sigmoid_dsps + elemwise_mul_rate_dsps, conv1_adds + conv2_adds, glavpool_mem + conv1_mem + conv2_mem 

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
            
            if len(self.layers[k]['input']) > 1:
                in1_layer, in1_layer_name = self.get_layer_from_id(int(self.layers[k]['input_id'][0]))
                in2_layer, in2_layer_name = self.get_layer_from_id(int(self.layers[k]['input_id'][1]))
                oldest_input = min(int(in1_layer['output_id']), int(in2_layer['output_id'])) if (in1_layer is not None and in2_layer is not None) else int(self.layers[k]['output_id'])
                # print("Layer name = {} ({}). Input_0 = {} ({}). Input_1 = {} ({})".format(name, self.layers[k]['output_id'], in1_layer_name, self.layers[k]['input_id'][0], in2_layer_name, self.layers[k]['input_id'][1]))
            else:
                in1_layer, in1_layer_name = self.get_layer_from_id(int(self.layers[k]['input_id'][0]))
                oldest_input = int(in1_layer['output_id']) if in1_layer is not None else int(self.layers[k]['output_id'])
                # print("Layer name = {} ({}). Input = {} ({})".format(name, self.layers[k]['output_id'], in1_layer_name, self.layers[k]['input_id'][0]))
            if int(self.layers[k]['output_id']) - oldest_input > 2:
                print("Identified a branching behaviour")

            if operation == 'Mul' or operation == 'Add':
                input_shape = output_shape
            kernel = self.layers[k]['kernel']
            bias = self.layers[k]['bias']
            groups = self.layers[k]['groups']

            self.modules[name] = {"operation": operation,
                                  "shape_in": input_shape,
                                  "shape_out": output_shape,
                                  "kernel": kernel,
                                  "bias": bias,
                                  "groups": groups}

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
                sigmoid = {"operation": operation,
                       "shape_in": input_shape,
                       "shape_out": output_shape,
                       "kernel": kernel,
                       "bias": bias,
                       "groups": groups}

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
                mul = {"operation": operation,
                       "shape_in": input_shape,
                       "shape_out": output_shape,
                       "kernel": kernel,
                       "bias": bias,
                       "groups": groups}

                name = 'Swish_' + swish_module[0][1].split('_')[1]
                operation = 'Swish'
                self.modules[name] = {"operation": operation,
                                      "shape_in": swish_input_shape,
                                      "shape_out": swish_output_shape,
                                      sigmoid_name: sigmoid,
                                      mul_name: mul}

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
                self.modules[conv_name] = {"operation": operation,
                                    "shape_in": input_shape,
                                    "shape_out": output_shape,
                                    "kernel": kernel,
                                    "bias": bias,
                                    "groups": groups}

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
                gap = {"operation": operation,
                       "shape_in": input_shape,
                       "shape_out": output_shape,
                       "kernel": kernel,
                       "bias": bias,
                       "groups": groups}

                conv1_name = se_module[1][1]
                operation = self.layers[conv1_name]['operation']
                input_shape = self.layers[conv1_name]['input'][0]
                output_shape = self.layers[conv1_name]['output']
                if operation == 'Mul' or operation == 'Add':
                    input_shape = output_shape
                kernel = self.layers[conv1_name]['kernel']
                bias = self.layers[conv1_name]['bias']
                groups = self.layers[conv1_name]['groups']
                conv1 = {"operation": operation,
                       "shape_in": input_shape,
                       "shape_out": output_shape,
                       "kernel": kernel,
                       "bias": bias,
                       "groups": groups}

                relu_name = se_module[2][1]
                operation = self.layers[relu_name]['operation']
                input_shape = self.layers[relu_name]['input'][0]
                output_shape = self.layers[relu_name]['output']
                if operation == 'Mul' or operation == 'Add':
                    input_shape = output_shape
                kernel = self.layers[relu_name]['kernel']
                bias = self.layers[relu_name]['bias']
                groups = self.layers[relu_name]['groups']
                relu = {"operation": operation,
                       "shape_in": input_shape,
                       "shape_out": output_shape,
                       "kernel": kernel,
                       "bias": bias,
                       "groups": groups}

                conv2_name = se_module[3][1]
                operation = self.layers[conv2_name]['operation']
                input_shape = self.layers[conv2_name]['input'][0]
                output_shape = self.layers[conv2_name]['output']
                if operation == 'Mul' or operation == 'Add':
                    input_shape = output_shape
                kernel = self.layers[conv2_name]['kernel']
                bias = self.layers[conv2_name]['bias']
                groups = self.layers[conv2_name]['groups']
                conv2 = {"operation": operation,
                       "shape_in": input_shape,
                       "shape_out": output_shape,
                       "kernel": kernel,
                       "bias": bias,
                       "groups": groups}

                sigmoid_name = se_module[4][1]
                operation = self.layers[sigmoid_name]['operation']
                input_shape = self.layers[sigmoid_name]['input'][0]
                output_shape = self.layers[sigmoid_name]['output']
                if operation == 'Mul' or operation == 'Add':
                    input_shape = output_shape
                kernel = self.layers[sigmoid_name]['kernel']
                bias = self.layers[sigmoid_name]['bias']
                groups = self.layers[sigmoid_name]['groups']
                sigmoid = {"operation": operation,
                       "shape_in": input_shape,
                       "shape_out": output_shape,
                       "kernel": kernel,
                       "bias": bias,
                       "groups": groups}

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
                mul = {"operation": operation,
                       "shape_in": input_shape,
                       "shape_out": output_shape,
                       "kernel": kernel,
                       "bias": bias,
                       "groups": groups}

                name = 'Se_' + se_module[0][1].split('_')[1]
                operation = 'SqueezeExcitation'
                self.modules[name] = {"operation": operation,
                                      "shape_in": se_input_shape,
                                      "shape_out": se_output_shape,
                                      gap_name: gap,
                                      conv1_name: conv1,
                                      relu_name: relu,
                                      conv2_name: conv2,
                                      sigmoid_name: sigmoid,
                                      mul_name: mul}

                del self.modules[gap_name]
                del self.modules[conv1_name]
                del self.modules[relu_name]
                del self.modules[conv2_name]
                del self.modules[sigmoid_name]
                del self.modules[mul_name]

    def model_layer(self, layer, csv_writer, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem):
        
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

        bw_in_gb = (self.cycles_per_sec*bw_in_w*self.wb)/1e9	
        bw_out_gb = (self.cycles_per_sec*bw_out_w*self.wb)/1e9

        thr_in = (self.cycles_per_sec*thr_w_in)/in_size
        thr_out = (self.cycles_per_sec*thr_w_out)/out_size

        thr_go = ((muls + adds)*self.cycles_per_sec)/1e9
        dsps_util = (dsps/self.fpga_dsps)*100

        if dsps_util < 90.0 and mem_util < 90.0:
            csv_writer.writerow([layer, folding, mem_util, dsps_util, thr_in, thr_out, bw_in_w, bw_out_w, mem_kb, mem_bram, bw_in_gb, bw_out_gb, muls, adds, dsps, thr_w_out, thr_go])

            print("On Chip Mem(KB) = {:<15.3f} On Chip Mem(BRAM) = {:<20.3f} On Chip Mem (BRAM %) = {:<20.3f}\nMem BW In(words/cycle) = {:<20.3f} Mem BW In(GBs/sec) = {:<20.3f} Mem BW Out(words/cycle) = {:<20.3f} Mem BW Out(GBs/sec) = {:<20.3f}\nMuls = {:<20.3f} Adds = {:<20.3f} DSPS = {:<20.3f} DSPS % = {:<20.3f}\nThroughtput(words/cycle) = {:<20.3f} Consumption(inputs/sec) = {:<20.3f} Throughtput(outputs/sec) = {:<20.3f} Throughtput(GOps/sec) = {:.3f}".format(mem_kb, mem_bram, mem_util, bw_in_w, bw_in_gb, bw_out_w, bw_out_gb, muls, adds, dsps, dsps_util, thr_w_out, thr_in, thr_out, thr_go))
        else:
            logging.warning("Design point dropped because of too many recourses needed. DSPS = {} ({}%). BRAM = {} ({}%)".format(dsps, dsps_util, mem_bram, mem_util))

    def create_design_points(self, file_name, s_in=1, s_out=1):
            if not os.path.exists(os.path.join(os.getcwd(), 'fpga_modeling_reports')):
                os.makedirs(os.path.join(os.getcwd(), 'fpga_modeling_reports'))
            csv_file = os.path.join(os.getcwd(), 'fpga_modeling_reports', file_name + '.csv')
            with open(csv_file, mode='w') as model_results:
                csv_writer = csv.writer(model_results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                csv_writer.writerow(["Layer", "Folding", "On-Chip Memory(BRAM %)", "DSPS %", "Consumption(inputs/sec)", "Throughtput(outputs/sec)", "Memory Bandwidth In(words/cycle)", "Memory Bandwidth Out(words/cycle)", "On-Chip Memory(KB)", "On-Chip Memory(BRAM)", "Memory Bandwidth In(GBs/sec)", "Memory Bandwidth Out(GBs/sec)", "Multipliers", "Adders", "DSPS", "Throughtput(words/cycle)", "Throughtput(GOps/sec)"])

                for k in self.modules.keys():
                    name = k
                    operation = self.modules[k]['operation']
                    logging.error("Layer: {} -> Operation: {}.".format(name, operation))

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
                        cin = kernel_shape[1]
                        cout = kernel_shape[0]

                        in_size = cin * din * hin * win
                        out_size = cout * dout * hout * wout

                        pr_name = name
                        # if cout == self.modules[name]['groups']:
                        #     pr_name = pr_name + "_DepthWise"
                        # if kd == 1 and kh == 1 and kw == 1:
                        #     pr_name = pr_name + "_PointWise"

                        coarse_in_config = [1, (cin * self.modules[name]['groups'])//4, (cin * self.modules[name]['groups'])//2, cin * self.modules[name]['groups']]
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
                        
                        for coarse_in in coarse_in_config:
                            for coarse_out in coarse_out_config:
                                for fine in fine_config:

                                    coarse_in_name = str(coarse_in)
                                    coarse_out_name = str(coarse_out)
                                    folding_name = "N_Coarse({}/{}) - f_Fine({:.2f})".format(coarse_in_name, coarse_out_name, fine)

                                    logging.warning("Fold = {}. Channels In = {} - Filters = {}".format(folding_name, cin, cout))

                                    rate_in, rate_out, muls, adds, mem = self.conv_layer_config(self.modules[name], fine, coarse_in, coarse_out, s_in=s_in, s_out=s_out)

                                    self.model_layer(pr_name, csv_writer, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem)

                    elif operation == 'BatchNormalization':
                        out_shape = self.modules[name]['shape_out']
                        cout = out_shape[1]
                        dout = out_shape[2]
                        hout = out_shape[3]
                        wout = out_shape[4]
                        out_size = int(np.prod(np.array(out_shape[1:])))

                        in_shape = self.modules[name]['shape_in']
                        cin = in_shape[1]
                        din = in_shape[2]
                        hin = in_shape[3]
                        win = in_shape[4]
                        in_size = int(np.prod(np.array(in_shape[1:])))

                        assert out_shape == in_shape, 'Input and output shapes bust be identical in BatchNormalization Layer'

                        # coarse_config = list(reduce(list.__add__, ([i, cin//i] for i in range(1, int(cin**0.5) + 1) if cin % i == 0)))
                        # coarse_config = [1, cin//4, cin//2, cin]
                        coarse_config = [cin//2]
                        coarse_config = np.unique(coarse_config)
                        coarse_config = coarse_config[np.nonzero(coarse_config)].tolist()

                        for coarse in coarse_config:
                            # TODO: Keep this as rate in = 1 = rate out and same with the resources. Change these values on bigger layer config during "runtime"
                            rate_in = 1 * coarse
                            rate_out = 1 * coarse
                            mem = cin * 4
                            muls = 1 * coarse
                            adds = 3 * coarse
                            folding_name = "N_Coarse({}/{})".format(coarse, coarse)

                            logging.warning("Fold = {}. Channels In = {} - Filters = {}".format(folding_name, cin, cout))

                            self.model_layer(name, csv_writer, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem)

                    elif operation == 'Relu':
                        out_shape = self.modules[name]['shape_out']
                        cout = out_shape[1]
                        dout = out_shape[2]
                        hout = out_shape[3]
                        wout = out_shape[4]
                        out_size = int(np.prod(np.array(out_shape[1:])))

                        in_shape = self.modules[name]['shape_in']
                        cin = in_shape[1]
                        din = in_shape[2]
                        hin = in_shape[3]
                        win = in_shape[4]
                        in_size = int(np.prod(np.array(in_shape[1:])))

                        assert out_shape == in_shape, 'Input and output shapes bust be identical in BatchNormalization Layer'

                        # coarse_config = list(reduce(list.__add__, ([i, cin//i] for i in range(1, int(cin**0.5) + 1) if cin % i == 0)))
                        # coarse_config = [1, cin//4, cin//2, cin]
                        coarse_config = [cin//2]
                        coarse_config = np.unique(coarse_config)
                        coarse_config = coarse_config[np.nonzero(coarse_config)].tolist()
                        

                        for coarse in coarse_config:
                            # TODO: Keep this as rate in = 1 = rate out and same with the resources. Change these values on bigger layer config during "runtime"
                            rate_in = 1 * coarse
                            rate_out = 1 * coarse
                            mem = 0
                            muls = 0
                            adds = 0
                            folding_name = "N_Coarse({}/{})".format(coarse, coarse)

                            logging.warning("Fold = {}. Channels In = {} - Filters = {}".format(folding_name, cin, cout))

                            self.model_layer(name, csv_writer, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem)
                    elif operation == 'GlobalAveragePool':
                        out_shape = self.modules[name]['shape_out']
                        cout = out_shape[1]
                        dout = out_shape[2]
                        hout = out_shape[3]
                        wout = out_shape[4]
                        out_size = int(np.prod(np.array(out_shape[1:])))

                        in_shape = self.modules[name]['shape_in']
                        cin = in_shape[1]
                        din = in_shape[2]
                        hin = in_shape[3]
                        win = in_shape[4]
                        in_size = int(np.prod(np.array(in_shape[1:])))

                        assert cin == cout, 'Input and output shapes bust be identical in BatchNormalization Layer'

                        # coarse_config = list(reduce(list.__add__, ([i, cin//i] for i in range(1, int(cin**0.5) + 1) if cin % i == 0)))
                        coarse_config = [1, cin//4, cin//2, cin]
                        coarse_config = np.unique(coarse_config)
                        coarse_config = coarse_config[np.nonzero(coarse_config)].tolist()

                        for coarse in coarse_config:

                            rate_in = 1 * coarse
                            rate_out = 1/(din * hin * win) * coarse
                            mem = cin
                            muls = 2 * coarse
                            adds = 1 * coarse
                            folding_name = "N_Coarse({}/{})".format(coarse, coarse)

                            logging.warning("Fold = {}. Channels In = {} - Filters = {}".format(folding_name, cin, cout))

                            self.model_layer(name, csv_writer, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem)
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

                        se_keys = list(self.modules[name].keys())
                        conv1_key = se_keys[4]
                        conv2_key = se_keys[6]
                        coarse_in_conv1 = self.modules[name][conv1_key]['kernel'][1]
                        coarse_in_conv2 = self.modules[name][conv2_key]['kernel'][1]
                        coarse_out_conv1 = self.modules[name][conv1_key]['kernel'][0]
                        coarse_out_conv2 = self.modules[name][conv2_key]['kernel'][0]

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
                      
                        kd_1 = self.modules[name][conv1_key]['kernel'][2]
                        kh_1 = self.modules[name][conv1_key]['kernel'][3]
                        kw_1 = self.modules[name][conv1_key]['kernel'][4]
                        max_fine = kd_1 * kh_1 * kw_1
                        fine_config_1 = np.array([kd_1/max_fine, kh_1/max_fine, kw_1/max_fine, (kd_1 * kh_1)/max_fine, (kh_1 * kw_1)/max_fine, (kd_1 * kw_1)/max_fine, 1])
                        fine_config_1 = np.unique(fine_config_1).tolist()
                        if kd_1 == 1 and kh_1 == 1 and kw_1 == 1:
                            fine_config_1 = [0.5, 1]

                        kd_2 = self.modules[name][conv2_key]['kernel'][2]
                        kh_2 = self.modules[name][conv2_key]['kernel'][3]
                        kw_2 = self.modules[name][conv2_key]['kernel'][4]
                        max_fine = kd_2 * kh_2 * kw_2
                        fine_config_2 = np.array([kd_2/max_fine, kh_2/max_fine, kw_2/max_fine, (kd_2 * kh_2)/max_fine, (kh_2 * kw_2)/max_fine, (kd_2 * kw_2)/max_fine, 1])
                        fine_config_2 = np.unique(fine_config_2).tolist()
                        if kd_2 == 1 and kh_2 == 1 and kw_2 == 1:
                            fine_config_2 = [0.5, 1]

                        for coarse_in_1 in coarse_in_config_conv1:
                            for coarse_in_2 in coarse_in_config_conv2:
                                for coarse_out_1 in coarse_out_config_conv1:
                                    for coarse_out_2 in coarse_out_config_conv2:
                                        for fine_1 in fine_config_1:
                                            for fine_2 in fine_config_2:
                                                # if not coarse_out_1 == coarse_in_2:
                                                #     logging.error("Skipping configuration N_Coarse_1({}/{}) - N_Coarse_2({}/{}) - f_Fine_1({}) - f_Fine_2({}) since N_Coarse_1 out ({}) does not match with N_Coarse_2 in ({}).".format(coarse_in_1, coarse_out_1, coarse_in_2, coarse_out_2, fine_1, fine_2, coarse_out_1, coarse_in_2))
                                                #     continue

                                                folding_name = "N_Coarse_1({}/{}) - N_Coarse_2({}/{}) - f_Fine_1({:.2f}) - f_Fine_2({:.2f})".format(coarse_in_1, coarse_out_1, coarse_in_2, coarse_out_2, fine_1, fine_2)
                                                
                                                logging.warning("Fold = {}".format(folding_name))
                                                
                                                #TODO: The input mem bw on this layer is very important so we add the 9/10 of the total bw as the input bw and only the 1/10 as the output bw. When this layer is combined with others in a bigger partition the input rate of this layer will be driven by the output rate of the previous on the graph.
                                                rate_in, rate_out, muls, adds, mem = self.se_layer_config(self.modules[name], coarse_in_1, coarse_out_1, coarse_in_2, coarse_out_2, fine_1, fine_2, s_in=s_in+s_out - 1, s_out=1)

                                                #TODO: Added worst possible case for buffering on se module i.e., buffer the whole feature map and all of the channels. Should fix this by checking the depth/latency of the left branch in order to calculate the exact buffering that is gonna needed in each se module.
                                                #TODO: Another solution is to read again from off-chip memory which will prevent the buffering i.e., reduce the BRAM needs BUT will reduce the mem bw in total as well since we need to first write the results (in a bigger layer-wise partition) and the read them again i.e., will probably need to have mem_bw / 4 instead of mem_bw / 2 in each point that we access the off-chip memory.
                                                branch_buffering = din * hin * win
                                                self.model_layer(name, csv_writer, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem + branch_buffering)

                    elif operation == 'Swish':
                        out_shape = self.modules[name]['shape_out']
                        cout = out_shape[1]
                        dout = out_shape[2]
                        hout = out_shape[3]
                        wout = out_shape[4]
                        out_size = int(np.prod(np.array(out_shape[1:])))

                        in_shape = self.modules[name]['shape_in']
                        cin = in_shape[1]
                        din = in_shape[2]
                        hin = in_shape[3]
                        win = in_shape[4]
                        in_size = int(np.prod(np.array(in_shape[1:])))

                        assert out_shape == in_shape, 'Input and output shapes bust be identical in BatchNormalization Layer'

                        # coarse_config = list(reduce(list.__add__, ([i, cin//i] for i in range(1, int(cin**0.5) + 1) if cin % i == 0)))
                        # coarse_config = [1, cin//4, cin//2, cin]
                        coarse_config = [cin//2]
                        coarse_config = np.unique(coarse_config)
                        coarse_config = coarse_config[np.nonzero(coarse_config)].tolist()


                        for coarse in coarse_config:
                            # TODO: Keep this as rate in = 1 = rate out and same with the resources. Change these values on bigger layer config during "runtime"
                            rate_in = 1 * coarse
                            rate_out = 1 * coarse
                            mem = 0
                            muls = 4 * coarse
                            adds = 1 * coarse
                            folding_name = "N_Coarse({}/{})".format(coarse, coarse)

                            logging.warning("Fold = {}. Channels In = {} - Filters = {}".format(folding_name, cin, cout))

                            self.model_layer(name, csv_writer, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem)
                    elif operation == 'Add':
                        out_shape = self.modules[name]['shape_out']
                        cout = out_shape[1]
                        dout = out_shape[2]
                        hout = out_shape[3]
                        wout = out_shape[4]
                        out_size = int(np.prod(np.array(out_shape[1:])))

                        in_shape = self.modules[name]['shape_in']
                        cin = in_shape[1]
                        din = in_shape[2]
                        hin = in_shape[3]
                        win = in_shape[4]
                        in_size = int(np.prod(np.array(in_shape[1:])))

                        assert out_shape == in_shape, 'Input and output shapes bust be identical in BatchNormalization Layer'

                        # coarse_config = list(reduce(list.__add__, ([i, cin//i] for i in range(1, int(cin**0.5) + 1) if cin % i == 0)))
                        # coarse_config = [1, cin//4, cin//2, cin]
                        coarse_config = [cin//2]
                        coarse_config = np.unique(coarse_config)
                        coarse_config = coarse_config[np.nonzero(coarse_config)].tolist()


                        for coarse in coarse_config:
                            # TODO: Keep this as rate in = 1 = rate out and same with the resources. Change these values on bigger layer config during "runtime"
                            rate_in = 1 * coarse
                            rate_out = 1 * coarse
                            mem = 0
                            muls = 0
                            adds = 1 * coarse
                            folding_name = "N_Coarse({}/{})".format(coarse, coarse)

                            logging.warning("Fold = {}. Channels In = {} - Filters = {}".format(folding_name, cin, cout))

                            self.model_layer(name, csv_writer, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem)
                    
                    # csv_writer.writerow(["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"])

    def compose_layers(self, file_name, layers_names, final_name, model_name, calculate_pareto, membw_in, membw_out):
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
                        l_configs[l].append([row[cols['Folding']], row[cols['On-Chip Memory(BRAM %)']], row[cols['DSPS %']], row[cols['Consumption(inputs/sec)']], row[cols['Throughtput(outputs/sec)']], row[cols['Memory Bandwidth In(words/cycle)']], row[cols['Memory Bandwidth Out(words/cycle)']]])
        
        sizes = []
        keys = []
        for k in l_configs.keys():
            sizes.append(len(l_configs[k]))
            keys.append(k)
        
        dsp_config = []
        bram_config = []
        throughput_config = []
        if len(sizes) == 13:
            with tqdm(total=int(np.prod(np.array(sizes)))) as pbar:
                for r1 in tqdm(range(sizes[0]), leave=False):
                    for c1 in tqdm(range(sizes[1]), leave=False):
                        for b1 in tqdm(range(sizes[2]), leave=False):
                            for r2 in tqdm(range(sizes[3]), leave=False):
                                for c2 in tqdm(range(sizes[4]), leave=False):
                                    for b2 in tqdm(range(sizes[5]), leave=False):
                                        for se1 in tqdm(range(sizes[6]), leave=False):
                                            for sw1 in tqdm(range(sizes[7]), leave=False):
                                                for c3 in tqdm(range(sizes[8]), leave=False):
                                                    for b3 in tqdm(range(sizes[9]), leave=False):
                                                        for c4 in tqdm(range(sizes[10]), leave=False):
                                                            for b4 in tqdm(range(sizes[11]), leave=False):
                                                                for a1 in tqdm(range(sizes[12]), leave=False):
                                                                    pbar.update(1)
                                                                    rates_graph = np.zeros( shape=(13,14) , dtype=float )

                                                                    rates_graph[0,0] = float(l_configs[keys[0]][r1][5])
                                                                    rates_graph[0,1] = float(l_configs[keys[0]][r1][6])
                                                                    in_module_ratio = rates_graph[0,1] / rates_graph[0,0]
                                                                    rates_graph[0,0] = min(rates_graph[0,0], membw_in)
                                                                    rates_graph[0,1] = min(rates_graph[0,1], membw_in * in_module_ratio)

                                                                    rates_graph[1,1] = float(l_configs[keys[1]][c1][5])
                                                                    rates_graph[1,2] = float(l_configs[keys[1]][c1][6])

                                                                    rates_graph[2,2] = float(l_configs[keys[2]][b1][5])
                                                                    rates_graph[2,3] = float(l_configs[keys[2]][b1][6])

                                                                    rates_graph[3,3] = float(l_configs[keys[3]][r2][5])
                                                                    rates_graph[3,4] = float(l_configs[keys[3]][r2][6])

                                                                    rates_graph[4,4] = float(l_configs[keys[4]][c2][5])
                                                                    rates_graph[4,5] = float(l_configs[keys[4]][c2][6])

                                                                    rates_graph[5,5] = float(l_configs[keys[5]][b2][5])
                                                                    rates_graph[5,6] = float(l_configs[keys[5]][b2][6])

                                                                    rates_graph[6,6] = float(l_configs[keys[6]][se1][5])
                                                                    rates_graph[6,7] = float(l_configs[keys[6]][se1][6])

                                                                    rates_graph[7,7] = float(l_configs[keys[7]][sw1][5])
                                                                    rates_graph[7,8] = float(l_configs[keys[7]][sw1][6])

                                                                    rates_graph[8,8] = float(l_configs[keys[8]][c3][5])
                                                                    rates_graph[8,9] = float(l_configs[keys[8]][c3][6])

                                                                    rates_graph[9,9] = float(l_configs[keys[9]][b3][5])
                                                                    rates_graph[9,10] = float(l_configs[keys[9]][b3][6])

                                                                    rates_graph[10,10] = float(l_configs[keys[10]][c4][5])
                                                                    rates_graph[10,11] = float(l_configs[keys[10]][c4][6])

                                                                    rates_graph[11,11] = float(l_configs[keys[11]][b4][5])
                                                                    rates_graph[11,12] = float(l_configs[keys[11]][b4][6])

                                                                    rates_graph[12,12] = float(l_configs[keys[12]][a1][5])
                                                                    rates_graph[12,13] = float(l_configs[keys[12]][a1][6])
                                                                    out_module_ratio = rates_graph[12,13] / rates_graph[12,12]
                                                                    rates_graph[12,12] = min(rates_graph[12,12], membw_out)
                                                                    rates_graph[12,13] = min(rates_graph[12,13], membw_out * out_module_ratio)

                                                                    bram_total = float(l_configs[keys[0]][r1][1]) + float(l_configs[keys[1]][c1][1]) + float(l_configs[keys[2]][b1][1]) + float(l_configs[keys[3]][r2][1]) + float(l_configs[keys[4]][c2][1]) + float(l_configs[keys[5]][b2][1]) + float(l_configs[keys[6]][se1][1]) + float(l_configs[keys[7]][sw1][1]) + float(l_configs[keys[8]][c3][1]) + float(l_configs[keys[9]][b3][1]) + float(l_configs[keys[10]][c4][1]) + float(l_configs[keys[11]][b4][1]) + float(l_configs[keys[12]][a1][1])

                                                                    dsps_total = float(l_configs[keys[0]][r1][2]) + float(l_configs[keys[1]][c1][2]) + float(l_configs[keys[2]][b1][2]) + float(l_configs[keys[3]][r2][2]) + float(l_configs[keys[4]][c2][2]) + float(l_configs[keys[5]][b2][2]) + float(l_configs[keys[6]][se1][2]) + float(l_configs[keys[7]][sw1][2]) + float(l_configs[keys[8]][c3][2]) + float(l_configs[keys[9]][b3][2]) + float(l_configs[keys[10]][c4][2]) + float(l_configs[keys[11]][b4][2]) + float(l_configs[keys[12]][a1][2])

                                                                    rates_graph_balanced = np.copy(rates_graph)
                                                                    rates_graph_balanced = self.balance_module_rates(rates_graph_balanced)

                                                                    rate_in = abs(rates_graph_balanced[0,0])
                                                                    rate_out = abs(rates_graph_balanced[12,13])
                                                                    
                                                                    in_shape = self.modules[keys[0]]['shape_in']
                                                                    in_size = int(np.prod(np.array(in_shape[1:])))
                                                                    out_shape = self.modules[keys[12]]['shape_out']
                                                                    out_size = int(np.prod(np.array(out_shape[1:])))

                                                                    thr_in = (self.cycles_per_sec*rate_in)/in_size
                                                                    thr_out = (self.cycles_per_sec*rate_out)/out_size

                                                                    dsp_config.append(dsps_total)
                                                                    bram_config.append(bram_total)
                                                                    throughput_config.append(thr_out)
        elif len(sizes) == 11:
            with tqdm(total=int(np.prod(np.array(sizes)))) as pbar:
                for r1 in tqdm(range(sizes[0]), leave=False):
                    for c1 in tqdm(range(sizes[1]), leave=False):
                        for b1 in tqdm(range(sizes[2]), leave=False):
                            for r2 in tqdm(range(sizes[3]), leave=False):
                                for c2 in tqdm(range(sizes[4]), leave=False):
                                    for b2 in tqdm(range(sizes[5]), leave=False):
                                        for se1 in tqdm(range(sizes[6]), leave=False):
                                            for sw1 in tqdm(range(sizes[7]), leave=False):
                                                for c3 in tqdm(range(sizes[8]), leave=False):
                                                    for b3 in tqdm(range(sizes[9]), leave=False):
                                                        for c4 in tqdm(range(sizes[10]), leave=False):
                                                            pbar.update(1)
                                                            rates_graph = np.zeros( shape=(11,12) , dtype=float )

                                                            rates_graph[0,0] = float(l_configs[keys[0]][r1][5])
                                                            rates_graph[0,1] = float(l_configs[keys[0]][r1][6])
                                                            in_module_ratio = rates_graph[0,1] / rates_graph[0,0]
                                                            rates_graph[0,0] = min(rates_graph[0,0], membw_in)
                                                            rates_graph[0,1] = min(rates_graph[0,1], membw_in * in_module_ratio)

                                                            rates_graph[1,1] = float(l_configs[keys[1]][c1][5])
                                                            rates_graph[1,2] = float(l_configs[keys[1]][c1][6])

                                                            rates_graph[2,2] = float(l_configs[keys[2]][b1][5])
                                                            rates_graph[2,3] = float(l_configs[keys[2]][b1][6])

                                                            rates_graph[3,3] = float(l_configs[keys[3]][r2][5])
                                                            rates_graph[3,4] = float(l_configs[keys[3]][r2][6])

                                                            rates_graph[4,4] = float(l_configs[keys[4]][c2][5])
                                                            rates_graph[4,5] = float(l_configs[keys[4]][c2][6])

                                                            rates_graph[5,5] = float(l_configs[keys[5]][b2][5])
                                                            rates_graph[5,6] = float(l_configs[keys[5]][b2][6])

                                                            rates_graph[6,6] = float(l_configs[keys[6]][se1][5])
                                                            rates_graph[6,7] = float(l_configs[keys[6]][se1][6])

                                                            rates_graph[7,7] = float(l_configs[keys[7]][sw1][5])
                                                            rates_graph[7,8] = float(l_configs[keys[7]][sw1][6])

                                                            rates_graph[8,8] = float(l_configs[keys[8]][c3][5])
                                                            rates_graph[8,9] = float(l_configs[keys[8]][c3][6])

                                                            rates_graph[9,9] = float(l_configs[keys[9]][b3][5])
                                                            rates_graph[9,10] = float(l_configs[keys[9]][b3][6])

                                                            rates_graph[10,10] = float(l_configs[keys[10]][c4][5])
                                                            rates_graph[10,11] = float(l_configs[keys[10]][c4][6])
                                                            out_module_ratio = rates_graph[10,11] / rates_graph[10,10]
                                                            rates_graph[10,10] = min(rates_graph[10,10], membw_out)
                                                            rates_graph[10,11] = min(rates_graph[10,11], membw_out * out_module_ratio)

                                                            bram_total = float(l_configs[keys[0]][r1][1]) + float(l_configs[keys[1]][c1][1]) + float(l_configs[keys[2]][b1][1]) + float(l_configs[keys[3]][r2][1]) + float(l_configs[keys[4]][c2][1]) + float(l_configs[keys[5]][b2][1]) + float(l_configs[keys[6]][se1][1]) + float(l_configs[keys[7]][sw1][1]) + float(l_configs[keys[8]][c3][1]) + float(l_configs[keys[9]][b3][1]) + float(l_configs[keys[10]][c4][1])

                                                            dsps_total = float(l_configs[keys[0]][r1][2]) + float(l_configs[keys[1]][c1][2]) + float(l_configs[keys[2]][b1][2]) + float(l_configs[keys[3]][r2][2]) + float(l_configs[keys[4]][c2][2]) + float(l_configs[keys[5]][b2][2]) + float(l_configs[keys[6]][se1][2]) + float(l_configs[keys[7]][sw1][2]) + float(l_configs[keys[8]][c3][2]) + float(l_configs[keys[9]][b3][2]) + float(l_configs[keys[10]][c4][2])

                                                            rates_graph_balanced = np.copy(rates_graph)
                                                            rates_graph_balanced = self.balance_module_rates(rates_graph_balanced)

                                                            rate_in = abs(rates_graph_balanced[0,0])
                                                            rate_out = abs(rates_graph_balanced[10,11])
                                                            
                                                            in_shape = self.modules[keys[0]]['shape_in']
                                                            in_size = int(np.prod(np.array(in_shape[1:])))
                                                            out_shape = self.modules[keys[10]]['shape_out']
                                                            out_size = int(np.prod(np.array(out_shape[1:])))

                                                            thr_in = (self.cycles_per_sec*rate_in)/in_size
                                                            thr_out = (self.cycles_per_sec*rate_out)/out_size

                                                            dsp_config.append(dsps_total)
                                                            bram_config.append(bram_total)
                                                            throughput_config.append(thr_out)
        elif len(sizes) == 10:
            with tqdm(total=int(np.prod(np.array(sizes)))) as pbar:
                for r1 in tqdm(range(sizes[0]), leave=False):
                    for c1 in tqdm(range(sizes[1]), leave=False):
                        for b1 in tqdm(range(sizes[2]), leave=False):
                            for r2 in tqdm(range(sizes[3]), leave=False):
                                for c2 in tqdm(range(sizes[4]), leave=False):
                                    for b2 in tqdm(range(sizes[5]), leave=False):
                                        for se1 in tqdm(range(sizes[6]), leave=False):
                                            for sw1 in tqdm(range(sizes[7]), leave=False):
                                                for c3 in tqdm(range(sizes[8]), leave=False):
                                                    for b3 in tqdm(range(sizes[9]), leave=False):
                                                        pbar.update(1)
                                                        rates_graph = np.zeros( shape=(10,11) , dtype=float )

                                                        rates_graph[0,0] = float(l_configs[keys[0]][r1][5])
                                                        rates_graph[0,1] = float(l_configs[keys[0]][r1][6])
                                                        in_module_ratio = rates_graph[0,1] / rates_graph[0,0]
                                                        rates_graph[0,0] = min(rates_graph[0,0], membw_in)
                                                        rates_graph[0,1] = min(rates_graph[0,1], membw_in * in_module_ratio)

                                                        rates_graph[1,1] = float(l_configs[keys[1]][c1][5])
                                                        rates_graph[1,2] = float(l_configs[keys[1]][c1][6])

                                                        rates_graph[2,2] = float(l_configs[keys[2]][b1][5])
                                                        rates_graph[2,3] = float(l_configs[keys[2]][b1][6])

                                                        rates_graph[3,3] = float(l_configs[keys[3]][r2][5])
                                                        rates_graph[3,4] = float(l_configs[keys[3]][r2][6])

                                                        rates_graph[4,4] = float(l_configs[keys[4]][c2][5])
                                                        rates_graph[4,5] = float(l_configs[keys[4]][c2][6])

                                                        rates_graph[5,5] = float(l_configs[keys[5]][b2][5])
                                                        rates_graph[5,6] = float(l_configs[keys[5]][b2][6])

                                                        rates_graph[6,6] = float(l_configs[keys[6]][se1][5])
                                                        rates_graph[6,7] = float(l_configs[keys[6]][se1][6])

                                                        rates_graph[7,7] = float(l_configs[keys[7]][sw1][5])
                                                        rates_graph[7,8] = float(l_configs[keys[7]][sw1][6])

                                                        rates_graph[8,8] = float(l_configs[keys[8]][c3][5])
                                                        rates_graph[8,9] = float(l_configs[keys[8]][c3][6])

                                                        rates_graph[9,9] = float(l_configs[keys[9]][b3][5])
                                                        rates_graph[9,10] = float(l_configs[keys[9]][b3][6])
                                                        out_module_ratio = rates_graph[9,10] / rates_graph[9,9]
                                                        rates_graph[9,9] = min(rates_graph[9,9], membw_out)
                                                        rates_graph[9,10] = min(rates_graph[9,10], membw_out * out_module_ratio)

                                                        bram_total = float(l_configs[keys[0]][r1][1]) + float(l_configs[keys[1]][c1][1]) + float(l_configs[keys[2]][b1][1]) + float(l_configs[keys[3]][r2][1]) + float(l_configs[keys[4]][c2][1]) + float(l_configs[keys[5]][b2][1]) + float(l_configs[keys[6]][se1][1]) + float(l_configs[keys[7]][sw1][1]) + float(l_configs[keys[8]][c3][1]) + float(l_configs[keys[9]][b3][1])

                                                        dsps_total = float(l_configs[keys[0]][r1][2]) + float(l_configs[keys[1]][c1][2]) + float(l_configs[keys[2]][b1][2]) + float(l_configs[keys[3]][r2][2]) + float(l_configs[keys[4]][c2][2]) + float(l_configs[keys[5]][b2][2]) + float(l_configs[keys[6]][se1][2]) + float(l_configs[keys[7]][sw1][2]) + float(l_configs[keys[8]][c3][2]) + float(l_configs[keys[9]][b3][2])

                                                        rates_graph_balanced = np.copy(rates_graph)
                                                        rates_graph_balanced = self.balance_module_rates(rates_graph_balanced)

                                                        rate_in = abs(rates_graph_balanced[0,0])
                                                        rate_out = abs(rates_graph_balanced[9,10])
                                                        
                                                        in_shape = self.modules[keys[0]]['shape_in']
                                                        in_size = int(np.prod(np.array(in_shape[1:])))
                                                        out_shape = self.modules[keys[9]]['shape_out']
                                                        out_size = int(np.prod(np.array(out_shape[1:])))

                                                        thr_in = (self.cycles_per_sec*rate_in)/in_size
                                                        thr_out = (self.cycles_per_sec*rate_out)/out_size

                                                        dsp_config.append(dsps_total)
                                                        bram_config.append(bram_total)
                                                        throughput_config.append(thr_out)
        elif len(sizes) == 9:
            with tqdm(total=int(np.prod(np.array(sizes)))) as pbar:
                for r1 in tqdm(range(sizes[0]), leave=False):
                    for c1 in tqdm(range(sizes[1]), leave=False):
                        for b1 in tqdm(range(sizes[2]), leave=False):
                            for r2 in tqdm(range(sizes[3]), leave=False):
                                for c2 in tqdm(range(sizes[4]), leave=False):
                                    for b2 in tqdm(range(sizes[5]), leave=False):
                                        for se1 in tqdm(range(sizes[6]), leave=False):
                                            for sw1 in tqdm(range(sizes[7]), leave=False):
                                                for c3 in tqdm(range(sizes[8]), leave=False):
                                                    pbar.update(1)
                                                    rates_graph = np.zeros( shape=(9,10) , dtype=float )

                                                    rates_graph[0,0] = float(l_configs[keys[0]][r1][5])
                                                    rates_graph[0,1] = float(l_configs[keys[0]][r1][6])
                                                    in_module_ratio = rates_graph[0,1] / rates_graph[0,0]
                                                    rates_graph[0,0] = min(rates_graph[0,0], membw_in)
                                                    rates_graph[0,1] = min(rates_graph[0,1], membw_in * in_module_ratio)

                                                    rates_graph[1,1] = float(l_configs[keys[1]][c1][5])
                                                    rates_graph[1,2] = float(l_configs[keys[1]][c1][6])

                                                    rates_graph[2,2] = float(l_configs[keys[2]][b1][5])
                                                    rates_graph[2,3] = float(l_configs[keys[2]][b1][6])

                                                    rates_graph[3,3] = float(l_configs[keys[3]][r2][5])
                                                    rates_graph[3,4] = float(l_configs[keys[3]][r2][6])

                                                    rates_graph[4,4] = float(l_configs[keys[4]][c2][5])
                                                    rates_graph[4,5] = float(l_configs[keys[4]][c2][6])

                                                    rates_graph[5,5] = float(l_configs[keys[5]][b2][5])
                                                    rates_graph[5,6] = float(l_configs[keys[5]][b2][6])

                                                    rates_graph[6,6] = float(l_configs[keys[6]][se1][5])
                                                    rates_graph[6,7] = float(l_configs[keys[6]][se1][6])

                                                    rates_graph[7,7] = float(l_configs[keys[7]][sw1][5])
                                                    rates_graph[7,8] = float(l_configs[keys[7]][sw1][6])

                                                    rates_graph[8,8] = float(l_configs[keys[8]][c3][5])
                                                    rates_graph[8,9] = float(l_configs[keys[8]][c3][6])
                                                    out_module_ratio = rates_graph[8,9] / rates_graph[8,8]
                                                    rates_graph[8,8] = min(rates_graph[8,8], membw_out)
                                                    rates_graph[8,9] = min(rates_graph[8,9], membw_out * out_module_ratio)

                                                    bram_total = float(l_configs[keys[0]][r1][1]) + float(l_configs[keys[1]][c1][1]) + float(l_configs[keys[2]][b1][1]) + float(l_configs[keys[3]][r2][1]) + float(l_configs[keys[4]][c2][1]) + float(l_configs[keys[5]][b2][1]) + float(l_configs[keys[6]][se1][1]) + float(l_configs[keys[7]][sw1][1]) + float(l_configs[keys[8]][c3][1])

                                                    dsps_total = float(l_configs[keys[0]][r1][2]) + float(l_configs[keys[1]][c1][2]) + float(l_configs[keys[2]][b1][2]) + float(l_configs[keys[3]][r2][2]) + float(l_configs[keys[4]][c2][2]) + float(l_configs[keys[5]][b2][2]) + float(l_configs[keys[6]][se1][2]) + float(l_configs[keys[7]][sw1][2]) + float(l_configs[keys[8]][c3][2])

                                                    rates_graph_balanced = np.copy(rates_graph)
                                                    rates_graph_balanced = self.balance_module_rates(rates_graph_balanced)

                                                    rate_in = abs(rates_graph_balanced[0,0])
                                                    rate_out = abs(rates_graph_balanced[8,9])
                                                    
                                                    in_shape = self.modules[keys[0]]['shape_in']
                                                    in_size = int(np.prod(np.array(in_shape[1:])))
                                                    out_shape = self.modules[keys[8]]['shape_out']
                                                    out_size = int(np.prod(np.array(out_shape[1:])))

                                                    thr_in = (self.cycles_per_sec*rate_in)/in_size
                                                    thr_out = (self.cycles_per_sec*rate_out)/out_size

                                                    dsp_config.append(dsps_total)
                                                    bram_config.append(bram_total)
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

        sns.scatterplot(x=throughput_config, y=dsp_config, s=50)
        plt.axhline(y=100, color='r', linestyle='-')
        plt.axhline(y=90, color='r', linestyle='--')

        print(np.unique(np.array(bram_config)))
        bram_tot = "{:.3f}".format(max(bram_config))

        plt.title(str(final_name) + " (" + bram_tot + " % BRAM Usage)")
        plt.xlabel('Throughtput(outputs/sec)')
        plt.xscale("log")
        plt.ylabel('DSPS %')
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

def plot_graph(x, y, leg, name, type, model_name, calculate_pareto):
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

        sns.scatterplot(x=np.array(x), y=np.array(y[0]), hue=leg, style=leg, s=75)

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
                plt.legend(legd, frameon=False, prop={"size":8}, loc='upper right', bbox_to_anchor=(1.11, 1.12), borderaxespad=0.)
            else:
                plt.legend([],[], frameon=False)
        else:
            plt.legend(frameon=False, prop={"size":8}, loc='upper right', bbox_to_anchor=(1.11, 1.12), borderaxespad=0.)

        file_name = name.replace('.', '_') + '.jpg'
        plt.savefig(os.path.join(dsps_dir, file_name))
        plt.clf()
    elif type == 'Memory Bandwidth':

        sns.scatterplot(x=np.array(x), y=np.array(y[0]), hue=leg, style=leg, s=75)

        plt.title(name)
        plt.xlabel('Throughtput(outputs/sec)')
        plt.ylabel('Memory Bandwidth IN (GBs/sec)')
        if max(x) > 100:
            plt.xscale("log")
        plt.legend(frameon=False, prop={"size":8}, loc='upper right', bbox_to_anchor=(1.11, 1.12), borderaxespad=0.)

        file_name = name.replace('.', '_') + '_in.jpg'
        plt.savefig(os.path.join(mem_bw_dir, file_name))
        plt.clf()


        sns.scatterplot(x=np.array(x), y=np.array(y[1]), hue=leg, style=leg, s=75)

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
                folding.append(row[cols['Folding']])
                dsp_util.append(float(row[cols['DSPS %']]))
                mem_bw_in.append(float(row[cols['Memory Bandwidth In(GBs/sec)']]))
                mem_bw_out.append(float(row[cols['Memory Bandwidth Out(GBs/sec)']]))
                throughput.append(float(row[cols['Throughtput(outputs/sec)']]))
            else:
                plot_graph(throughput, [dsp_util], folding, prev_layer, 'DSPS', file_name, calculate_pareto=calculate_pareto)
                plot_graph(throughput, [mem_bw_in, mem_bw_out], folding, prev_layer, 'Memory Bandwidth', file_name, calculate_pareto=calculate_pareto)

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

        plot_graph(throughput, [dsp_util], folding, prev_layer, 'DSPS', file_name, calculate_pareto=calculate_pareto)
        plot_graph(throughput, [mem_bw_in, mem_bw_out], folding, prev_layer, 'Memory Bandwidth', file_name, calculate_pareto=calculate_pareto)

def drop_duplicates(file_name="x3d_m", pareto=False):
    
    if pareto:
        csv_file_read = os.path.join(os.getcwd(), 'fpga_modeling_reports', file_name + '_pareto.csv')
    else:
        csv_file_read = os.path.join(os.getcwd(), 'fpga_modeling_reports', file_name + '.csv')

    data = pd.read_csv(csv_file_read)
    columns = data.columns.tolist()
    del(columns[1])

    data_droped = data.drop_duplicates(subset=columns)
    os.remove(csv_file_read)
    data_droped.to_csv(csv_file_read, index=False)

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
                    # csv_writer_par.writerow(["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"])
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
            # csv_writer_par.writerow(["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"])

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
    parser.add_argument('--calculate_pareto', action='store_true', help='whether to calculate and plot pareto front')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--label', default=None, help='label file')
    parser.add_argument('--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument('--imshape', type=int, nargs="+", default=[224, 224, 3], help='image size for inference')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Target FPGA Zynq UltraScale+ MPSoC ZCU104. Assuming clock frequency of 100 MHz.
    # The actual BRAM size is 11 Mbits (1.375 MBytes). This divided by the 18 Kbits size of each BRAM gives a total of 624 BRAM units.
    # The ZCU104 has also 27 Mbits (3.375 MBytes) of URAM. This divided by the 288 Kbits size of each URAM gives a total of 96 URAM units.
    # The ZCU104 has 20 GTH gigabit transceivers (16.3 Gb/s or 2.03 GB/s) on the PL-size
    onnx_modeling = ModelFeatureMapsOnnx(model=args.model_name, word_length=16, clock_freq=100, bram=624, dsp=1728, mem_bw=16.3)

    onnx_modeling.from_onnx()

    # onnx_modeling.get_info()

    onnx_modeling.create_modules()

    fname = args.model_name + '_onnx'
    onnx_modeling.create_design_points(file_name=fname, s_in=onnx_modeling.max_words_per_cycle//2, s_out=onnx_modeling.max_words_per_cycle//2)
    #TODO: Additionaly to saving the per module results as regards the different configurations save the configurations themselves to use them and create on the fly the results during the layer creation on creating bigger layers.
    # onnx_modeling.create_design_points(file_name=fname, s_in=10000, s_out=10000)

    drop_duplicates(file_name=fname, pareto=False)

    get_paretto(file_name=fname)

    drop_duplicates(file_name=fname, pareto=True)

    #TODO: (URGENT) Take into consideration the buffering needed in branching or read again from the off-chip memory and reduce the bw in the individual layers.
    partition_layers = get_partition_layers(onnx_modeling.modules, args.model_name)

    fname_pareto = fname + "_pareto"
    
    # args = list()
    # for n, l in enumerate(partition_layers):
    #     args.append((fname_pareto, l, n+1, fname,))
    # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    #     for _ in executor.map(onnx_modeling.thread_helper, args):
    #         pass

    for n, l in enumerate(partition_layers):
        if len(l)<13:
            print("Evaluating Layer {}/{}".format(n+1, len(partition_layers)))
            onnx_modeling.compose_layers(fname_pareto, l, n+1, fname, args.calculate_pareto, onnx_modeling.max_words_per_cycle//2, onnx_modeling.max_words_per_cycle//2)   

    # performance_graphs(file_name=fname, layers_to_plot=['Conv', 'Se', 'GlobalAveragePool'], calculate_pareto=args.calculate_pareto)

if __name__ == '__main__':
    main()