import onnx
import os
import onnx.numpy_helper as onh
import argparse
import coloredlogs
import logging
import seaborn as sns
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO)
sns.set(rc={'figure.figsize':(15,8)})
sns.set_style("darkgrid", {"axes.facecolor": ".85"})

class ModelAnalyser():
    def __init__(self, model):
        self.model_name = model
        self.model_path = model + ".onnx"
        
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

    def plot_params(self):
        if not os.path.exists(os.path.join(os.getcwd(), 'fpga_modeling_reports', 'param_analysis', self.model_name)):
            os.makedirs(os.path.join(os.getcwd(), 'fpga_modeling_reports', 'param_analysis', self.model_name))

        model_nodes = self.onnx_model.graph.node

        for node in model_nodes:
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
                plt.savefig(os.path.join('fpga_modeling_reports', 'param_analysis', self.model_name, img_name))
                plt.clf()

def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 parse model')
    parser.add_argument('model_name', help='name of the har model')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    analyser = ModelAnalyser(args.model_name)
    analyser.plot_params()