import onnx
import onnxruntime
import os
import onnx.numpy_helper as onh
from onnx import helper
import numpy as np
import argparse
import coloredlogs
import logging
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

coloredlogs.install(level='INFO')
logging.basicConfig(level=logging.INFO)
sns.set(rc={'figure.figsize':(15,8)})
sns.set_style("whitegrid")

supported_ops_onnx = ['Conv', 'Add', 'AveragePool', 'BatchNormalization', 'Div', 'Einsum', 'Elu', 'Exp', 'GRU', 'Gemm', 'GlobalAveragePool', 'GlobalLpPool', 'GlobalMaxPool', 'HardSigmoid', 'LSTM', 'LeakyRelu', 'MatMul', 'MaxPool', 'PRelu', 'Pow', 'RNN', 'Reciprocal', 'Relu', 'Selu', 'Sigmoid', 'Sqrt', 'Sub', 'Tan', 'Tanh', 'Transpose', 'Celu', 'HardSwish', 'LogSoftmax', 'Softmax']

class ModelAnalyser():
    def __init__(self, model):
        self.model_name = model
        self.model_path = model + ".onnx"
        self.model_path_extended = self.model_name + "_extended.onnx"
        
        self.onnx_model = onnx.load(self.model_path)
        onnx.checker.check_model(self.onnx_model)

    def get_params(self, input):
        node_params = []
        if len(input) == 0:
            logging.info("Layer has no inputs")
            return node_params
        
        for node_input in input:
            for inp in self.onnx_model.graph.initializer:
                if node_input == inp.name:
                    curr_params = onh.to_array(inp)
                    node_params.append(curr_params)
                    break
        
        if len(node_params) == 0:
            logging.info("Layer has no parameters")

        return node_params

    def add_fmaps_to_outputs(self):
        model_nodes = self.onnx_model.graph.node
        model_outputs = [n.name for n in self.onnx_model.graph.output]
        for node in model_nodes:
            if node.output[0] in model_outputs:
                continue
            logging.info("{} -> {} ({})".format(node.name, node.output, len(node.output)))
            intermediate_layer_value_info = helper.ValueInfoProto()
            intermediate_layer_value_info.name = node.output[0]
            self.onnx_model.graph.output.append(intermediate_layer_value_info)
        onnx.save(self.onnx_model, self.model_path_extended)

    def get_fmaps(self):
        self.add_fmaps_to_outputs()

        session = onnxruntime.InferenceSession(self.model_path_extended, None)

        input_name = session.get_inputs()[0].name  
        ucf_data = np.load('/home/ptoupas/Development/ACTIVE/mmaction2/ucf101_examples.npy')
        final_outputs = []

        for i in range(1): #ucf_data.shape[0]
            input = ucf_data[i]
            outputs = session.run([], {input_name: input})
            if len(final_outputs) == 0:
                final_outputs = outputs
                continue
            for j in range(len(outputs)):
                final_outputs[j] = (final_outputs[j] + outputs[j])/2

        self.plot_inter_fmaps(final_outputs)

    def is_supported(self, out_id):
        for node in self.onnx_model.graph.node:
            if node.output[0] == out_id.name:
                if node.op_type in supported_ops_onnx:
                    return True
        return False

    def plot_inter_fmaps(self, outputs):
        if not os.path.exists(os.path.join(os.getcwd(), 'fpga_modeling_reports', 'param_analysis', self.model_name, 'fmaps')):
            os.makedirs(os.path.join(os.getcwd(), 'fpga_modeling_reports', 'param_analysis', self.model_name, 'fmaps'))

        model_outputs = [n.name for n in self.onnx_model.graph.output if self.is_supported(n)]
        
        tota_max = -1000000
        total_min = 1000000
        for out, out_name in tqdm(zip(outputs, model_outputs), leave=False):
            if out_name == '961' or out_name == '962' or out_name == '963' or out_name == '964' or out_name == '965' or out_name == '966' or out_name == '967' or out_name == '968':
                continue
            q0 = np.percentile(out.flatten(),50)
            q1 = np.percentile(out.flatten(),25)
            q3 = np.percentile(out.flatten(),75)
            iqr = q3-q1
            whis = 3.5
            upper_wisk = q0 + whis*iqr
            lower_wisk = q0 - whis*iqr
            tota_max = max(tota_max,upper_wisk)
            total_min = min(total_min,lower_wisk)

            # sns.histplot(x=out.flatten(), bins=250, kde=True)
            # sns.boxplot(x=out.flatten(), whis=[1, 99])
            sns.boxplot(x=out.flatten(), whis=3.5)
            plt.axvline(x=min(out.flatten()), color='r')
            plt.axvline(x=max(out.flatten()), color='r')
            plt.title(out_name + " ({})".format(out.flatten().shape[0]))
            img_name = out_name + ".png"
            plt.savefig(os.path.join('fpga_modeling_reports', 'param_analysis', self.model_name, 'fmaps', img_name))
            plt.clf()
        logging.info("FMAPS: total max = {:.4f}, total min = {:.4f}".format(tota_max, total_min))

    def get_filters(self):
        if not os.path.exists(os.path.join(os.getcwd(), 'fpga_modeling_reports', 'param_analysis', self.model_name, 'filters')):
            os.makedirs(os.path.join(os.getcwd(), 'fpga_modeling_reports', 'param_analysis', self.model_name, 'filters'))

        model_nodes = self.onnx_model.graph.node

        tota_max = -1000000
        total_min = 1000000
        for node in model_nodes:
            logging.info("{}".format(node.name))
            params = self.get_params(node.input)
            
            concatenated_params = []
            for p in params:
                concatenated_params.extend(list(p.flatten()))
            node_name = node.name

            num_params = len(concatenated_params)
            if num_params > 0:
                concatenated_params = np.array(concatenated_params).flatten()
                q0 = np.percentile(concatenated_params,50)
                q1 = np.percentile(concatenated_params,25)
                q3 = np.percentile(concatenated_params,75)
                iqr = q3-q1
                whis = 3.5
                upper_wisk = q0 + whis*iqr
                lower_wisk = q0 - whis*iqr
                tota_max = max(tota_max,upper_wisk)
                total_min = min(total_min,lower_wisk)

                # sns.histplot(x=concatenated_params, bins=250, kde=True)
                # sns.boxplot(x=concatenated_params, whis=[1, 99])
                sns.boxplot(x=concatenated_params, whis=3.5)
                plt.axvline(x=min(concatenated_params), color='r')
                plt.axvline(x=max(concatenated_params), color='r')
                plt.title(node_name + " ({})".format(num_params))
                img_name = node_name + ".png"
                plt.savefig(os.path.join('fpga_modeling_reports', 'param_analysis', self.model_name, 'filters', img_name))
                plt.clf()
        logging.info("FILTERS: total max = {:.4f}, total min = {:.4f}".format(tota_max, total_min))

def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 parse model')
    parser.add_argument('model_name', help='name of the har model')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    analyser = ModelAnalyser(args.model_name)
    # analyser.get_filters()
    analyser.get_fmaps()