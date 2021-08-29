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
sns.set_style("darkgrid", {"axes.facecolor": ".85"})

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

        #TODO: Should load some inputs from the kinetics 400 dataset and get the final results using this input.
        dummy_input = np.random.randn(1, 1, 3, 16, 224, 224).astype(np.float32)
        
        outputs = session.run([], {input_name: dummy_input})
        self.plot_inter_fmaps(outputs)

    def plot_inter_fmaps(self, outputs):
        if not os.path.exists(os.path.join(os.getcwd(), 'fpga_modeling_reports', 'param_analysis', self.model_name, 'fmaps')):
            os.makedirs(os.path.join(os.getcwd(), 'fpga_modeling_reports', 'param_analysis', self.model_name, 'fmaps'))

        model_outputs = [n.name for n in self.onnx_model.graph.output]
        for out, out_name in tqdm(zip(outputs, model_outputs), leave=False):
            sns.histplot(x=out.flatten(), bins=250, kde=True)
            plt.title(out_name + " ({})".format(out.flatten().shape[0]))
            img_name = out_name + ".png"
            plt.savefig(os.path.join('fpga_modeling_reports', 'param_analysis', self.model_name, 'fmaps', img_name))
            plt.clf()

    def get_filters(self):
        if not os.path.exists(os.path.join(os.getcwd(), 'fpga_modeling_reports', 'param_analysis', self.model_name, 'filters')):
            os.makedirs(os.path.join(os.getcwd(), 'fpga_modeling_reports', 'param_analysis', self.model_name, 'filters'))

        model_nodes = self.onnx_model.graph.node

        for node in tqdm(model_nodes, leave=False):
            logging.info("{}".format(node.name))
            params = self.get_params(node.input)
            
            concatenated_params = []
            for p in params:
                concatenated_params.extend(list(p.flatten()))
            node_name = node.name

            num_params = len(concatenated_params)
            if num_params > 0:
                sns.histplot(x=concatenated_params, bins=250, kde=True)
                plt.title(node_name + " ({})".format(num_params))
                img_name = node_name + ".png"
                plt.savefig(os.path.join('fpga_modeling_reports', 'param_analysis', self.model_name, 'filters', img_name))
                plt.clf()

def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 parse model')
    parser.add_argument('model_name', help='name of the har model')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    analyser = ModelAnalyser(args.model_name)
    analyser.get_filters()
    analyser.get_fmaps()