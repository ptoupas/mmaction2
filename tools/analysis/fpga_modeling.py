import argparse
from collections import deque
import csv
from logging.handlers import DatagramHandler
import os
import math
import numpy as np
from matplotlib import pyplot as plt
from numpy.testing._private.utils import assert_equal
import seaborn as sns
import coloredlogs, logging
coloredlogs.install(level='INFO')
from functools import reduce

import pandas as pd
import torch
import onnx
import onnxruntime as rt

import mmcv
from mmcv.parallel import collate, scatter

from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose

np.set_printoptions(precision=5, suppress=True)

EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode', 'FrameSelector'
]

class ModelFeatureMapsOnnx():

    def __init__(self, model, word_length, clock_freq, bram, dsp):
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

        self.op_list = ['Conv', 'BatchNormalization', 'Relu', 'GlobalAveragePool', 'AveragePool', 'MaxPool', 'Sigmoid', 'Mul', 'Add', 'Div', 'MatMul', 'Gemm', 'Elu', 'Flatten', 'GRU', 'HardSigmoid', 'LSTM', 'LeakyRelu', 'PRelu', 'RNN', 'Selu', 'Tanh', 'Celu', 'HardSwish', 'Softmax']
        self.onnx_model = onnx.load(self.model_path)
        onnx.checker.check_model(self.onnx_model)

        # print(onnx.helper.printable_graph(self.onnx_model.graph))

    def balance_module_rates(self, rate_graph):
        
        rate_ratio = [ abs(rate_graph[i,i+1]/rate_graph[i,i]) for i in range(rate_graph.shape[0]) ]

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

    def conv_layer_config(self, module, fine, coarse_in, coarse_out):
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

        cin = kernel_shape[1]
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
        rates_graph[0,0] = rin_sw
        rates_graph[0,1] = rout_sw

        # Rates for the Conv module
        rin_conv = fine/cout
        rout_conv = fine
        rates_graph[1,1] = rin_conv * coarse_out
        rates_graph[1,2] = rout_conv * coarse_in
        
        if not depthwise:
            # Rates for the Accumulator module
            rin_accum = 1
            rout_accum = 1/cin
            rates_graph[2,2] = rin_accum
            rates_graph[2,3] = rout_accum * coarse_in

            # print("CONV RATE GRAPH")
            # print(rates_graph)
            # print("-"*50)
            rates_graph = self.balance_module_rates(rates_graph)
            # print(rates_graph)
            # print("=="*50)
            rate_in = abs(rates_graph[0,0])
            rate_out = abs(rates_graph[2,3])
        else:
            # print("CONV RATE GRAPH")
            # print(rates_graph)
            # print("-"*50)
            rates_graph = self.balance_module_rates(rates_graph)
            # print(rates_graph)
            # print("=="*50)
            rate_in = abs(rates_graph[0,0])
            rate_out = abs(rates_graph[1,2])

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

    def se_layer_config(self, module, coarse_in_1, coarse_out_1, coarse_in_2, coarse_out_2, fine1, fine2):
        se_keys = list(module.keys())
        glavpool_key = se_keys[3]
        conv1_key = se_keys[4]
        conv2_key = se_keys[6]

        in_shape = module[glavpool_key]['shape_in']
        glavpool_rate_in = 1 # * module[conv1_key]['kernel'][1]
        glavpool_rate_out = 1/(in_shape[2]*in_shape[3]*in_shape[4]) # * module[conv1_key]['kernel'][1]
        glavpool_mem = in_shape[1]
        glavpool_muls = 2

        conv1_rate_in, conv1_rate_out, conv1_muls, conv1_adds, conv1_mem = self.conv_layer_config(module[conv1_key], coarse_in_1, coarse_out_1, fine1)

        relu_rate_in = 1
        relu_rate_out = 1

        conv2_rate_in, conv2_rate_out, conv2_muls, conv2_adds, conv2_mem = self.conv_layer_config(module[conv2_key], coarse_in_2, coarse_out_2, fine2)

        sigmoid_rate_in = 1
        sigmoid_rate_out = 1
        sigmoid_dsps = 3

        elemwise_mul_rate_in = 1
        elemwise_mul_rate_out = 1
        elemwise_mul_rate_dsps = 1

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

        return rate_in, rate_out, glavpool_muls + conv1_muls + conv2_muls + sigmoid_dsps + elemwise_mul_rate_dsps, conv1_adds + conv2_adds, glavpool_mem + conv1_mem + conv2_mem 

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
            logging.error("Design point dropped because of too many recourses needed. DSPS = {} ({}%). BRAM = {} ({}%)".format(dsps, dsps_util, mem_bram, mem_util))

    def create_design_points(self, file_name):
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
                        if cout == self.modules[name]['groups']:
                            pr_name = pr_name + "_DepthWise"
                        if kd == 1 and kh == 1 and kw == 1:
                            pr_name = pr_name + "_PointWise"

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

                                    rate_in, rate_out, muls, adds, mem = self.conv_layer_config(self.modules[name], fine, coarse_in, coarse_out)

                                    # logging.error("Fold = {}. rate IN = {}. rate OUT = {}".format(folding_name, rate_in, rate_out))

                                    self.model_layer(pr_name, csv_writer, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem)

                    elif operation == 'BatchNormalization':
                        out_shape = self.modules[name]['shape_out']
                        cout = out_shape[1]
                        dout = out_shape[2]
                        hout = out_shape[3]
                        wout = out_shape[4]
                        out_size = int(np.prod(np.array(out_shape)))

                        in_shape = self.modules[name]['shape_in']
                        cin = in_shape[1]
                        din = in_shape[2]
                        hin = in_shape[3]
                        win = in_shape[4]
                        in_size = int(np.prod(np.array(in_shape)))

                        assert out_shape == in_shape, 'Input and output shapes bust be identical in BatchNormalization Layer'

                        # coarse_config = list(reduce(list.__add__, ([i, cin//i] for i in range(1, int(cin**0.5) + 1) if cin % i == 0)))
                        coarse_config = [1, cin//4, cin//2, cin]
                        coarse_config = np.unique(coarse_config)
                        coarse_config = coarse_config[np.nonzero(coarse_config)].tolist()

                        for coarse in coarse_config:

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
                        out_size = int(np.prod(np.array(out_shape)))

                        in_shape = self.modules[name]['shape_in']
                        cin = in_shape[1]
                        din = in_shape[2]
                        hin = in_shape[3]
                        win = in_shape[4]
                        in_size = int(np.prod(np.array(in_shape)))

                        assert out_shape == in_shape, 'Input and output shapes bust be identical in BatchNormalization Layer'

                        # coarse_config = list(reduce(list.__add__, ([i, cin//i] for i in range(1, int(cin**0.5) + 1) if cin % i == 0)))
                        coarse_config = np.unique(coarse_config)
                        coarse_config = coarse_config[np.nonzero(coarse_config)].tolist()
                        

                        for coarse in coarse_config:

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
                        out_size = int(np.prod(np.array(out_shape)))

                        in_shape = self.modules[name]['shape_in']
                        cin = in_shape[1]
                        din = in_shape[2]
                        hin = in_shape[3]
                        win = in_shape[4]
                        in_size = int(np.prod(np.array(in_shape)))

                        assert cin == cout, 'Input and output shapes bust be identical in BatchNormalization Layer'

                        # coarse_config = list(reduce(list.__add__, ([i, cin//i] for i in range(1, int(cin**0.5) + 1) if cin % i == 0)))
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

                        # coarse_in_config_conv1 = [1, coarse_in_conv1]
                        # coarse_in_config_conv2 = [1, coarse_in_conv2]
                        # coarse_out_config_conv1 = [1, coarse_out_conv1]
                        # coarse_out_config_conv2 = [1, coarse_out_conv2]
                        
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
                                                if not coarse_out_1 == coarse_in_2:
                                                    logging.error("Skipping configuration N_Coarse_1({}/{}) - N_Coarse_2({}/{}) - f_Fine_1({}) - f_Fine_2({}) since N_Coarse_1 out ({}) does not match with N_Coarse_2 in ({}).".format(coarse_in_1, coarse_out_1, coarse_in_2, coarse_out_2, fine_1, fine_2, coarse_out_1, coarse_in_2))
                                                    continue

                                                folding_name = "N_Coarse_1({}/{}) - N_Coarse_2({}/{}) - f_Fine_1({:.2f}) - f_Fine_2({:.2f})".format(coarse_in_1, coarse_out_1, coarse_in_2, coarse_out_2, fine_1, fine_2)
                                                
                                                logging.warning("Fold = {}".format(folding_name))

                                                rate_in, rate_out, muls, adds, mem = self.se_layer_config(self.modules[name], coarse_in_1, coarse_out_1, coarse_in_2, coarse_out_2, fine_1, fine_2)
                                                # logging.error("Fold = {}. rate IN = {}. rate OUT = {}".format(folding_name, rate_in, rate_out))

                                                self.model_layer(name, csv_writer, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem)

                    elif operation == 'Swish':
                        out_shape = self.modules[name]['shape_out']
                        cout = out_shape[1]
                        dout = out_shape[2]
                        hout = out_shape[3]
                        wout = out_shape[4]
                        out_size = int(np.prod(np.array(out_shape)))

                        in_shape = self.modules[name]['shape_in']
                        cin = in_shape[1]
                        din = in_shape[2]
                        hin = in_shape[3]
                        win = in_shape[4]
                        in_size = int(np.prod(np.array(in_shape)))

                        assert out_shape == in_shape, 'Input and output shapes bust be identical in BatchNormalization Layer'

                        # coarse_config = list(reduce(list.__add__, ([i, cin//i] for i in range(1, int(cin**0.5) + 1) if cin % i == 0)))
                        coarse_config = np.unique(coarse_config)
                        coarse_config = coarse_config[np.nonzero(coarse_config)].tolist()


                        for coarse in coarse_config:

                            rate_in = 1 * coarse
                            rate_out = 1 * coarse
                            mem = 0
                            muls = 4 * coarse
                            adds = 1 * coarse
                            folding_name = "N_Coarse({}/{})".format(coarse, coarse)

                            logging.warning("Fold = {}. Channels In = {} - Filters = {}".format(folding_name, cin, cout))

                            self.model_layer(name, csv_writer, folding_name, in_size, out_size, rate_in, rate_out, muls, adds, mem)

                    csv_writer.writerow(["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"])

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
    sns.set_style("darkgrid", {"axes.facecolor": ".85"})
    
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

def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 parse model')
    parser.add_argument('model_name', help='name of the har model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--from_onnx', action='store_true', help='whether to do the modeling from an onnx file or not')
    parser.add_argument('--use_frames', help='whether to use video decoder or raw frame decoder')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--label', default=None, help='label file')
    parser.add_argument('--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument('--imshape', type=int, nargs="+", default=[224, 224, 3], help='image size for inference')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if not args.from_onnx:
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

    else:
        # Target FPGA Zynq UltraScale+ MPSoC ZCU104. Assuming clock frequency of 100 MHz.
        # The actual BRAM size is 11 Mbits (1.375 MBytes). This divided by the 18 Kbits size of each BRAM gives a total of 624 BRAM units.
        # The ZCU104 has also 27 Mbits (3.375 MBytes) of URAM. This divided by the 288 Kbits size of each URAM gives a total of 96 URAM units.
        onnx_modeling = ModelFeatureMapsOnnx(model=args.model_name, word_length=16, clock_freq=100, bram=624, dsp=1728)

        onnx_modeling.from_onnx()

        # onnx_modeling.get_info()

        onnx_modeling.create_modules()

        fname = args.model_name + '_onnx'
        onnx_modeling.create_design_points(file_name=fname)

        performance_graphs(file_name=fname, layer_to_plot=None)

if __name__ == '__main__':
    main()