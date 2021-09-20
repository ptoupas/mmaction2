import onnx
import onnxruntime
import numpy as np
import argparse
import coloredlogs
import logging
import onnx.numpy_helper as onh
from onnx import helper

coloredlogs.install(level='INFO')
logging.basicConfig(level=logging.INFO)

class Quantizer():
    def __init__(self, model):
      self.fractional_part = 8
      self.model_name = model
      self.model_path = model + ".onnx"
      self.model_path_extended = self.model_name + "_extended.onnx"
      self.model_path_quantized = self.model_name + "_quantized.onnx"
      
      self.onnx_model = onnx.load(self.model_path)
      self.onnx_model = onnx.shape_inference.infer_shapes(self.onnx_model)
      onnx.checker.check_model(self.onnx_model)

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

    def get_params(self, input):
      node_params = []
      node_params_indicies = []
      if len(input) == 0:
          logging.info("Layer has no inputs")
          return node_params, node_params_indicies
      
      for node_input in input:
          for idx, inp in enumerate(self.onnx_model.graph.initializer):
              if node_input == inp.name:
                  # logging.info("input {}".format(node_input))
                  curr_params = onh.to_array(inp)
                  node_params.append(curr_params)
                  node_params_indicies.append(idx)
                  break
      
      if len(node_params) == 0:
          logging.info("Layer has no parameters")

      return node_params, node_params_indicies

    def get_weights(self):
      model_nodes = self.onnx_model.graph.node

      max_val = -1000000
      min_val = 1000000
      for node in model_nodes:
          logging.info("Node {}".format(node.name))
          params, indicies = self.get_params(node.input)
          
          for p, idx in zip(params, indicies):
            max_val = max(max_val, np.max(p))
            min_val = min(min_val, np.min(p))
            p_quantized = self.quantize(p)
            # convert numpy to TensorProto
            tensor = onh.from_array(p_quantized)
            tensor.name = self.onnx_model.graph.initializer[idx].name
            # copy this TensorProto to target tensor
            self.onnx_model.graph.initializer[idx].CopyFrom(tensor)
            self.quantization_error(p, p_quantized)

      logging.info("Min val = {}. Max val = {}".format(min_val, max_val))
      onnx.save(self.onnx_model, self.model_path_quantized)

    def quantize(self, data):
      shift_left = np.ones((data.shape))*(2**self.fractional_part)
      shift_left = shift_left.astype(np.float32)

      shift_right = np.ones((data.shape))*(2**(-self.fractional_part))
      shift_right = shift_right.astype(np.float32)

      fp_data = data * shift_left
      if fp_data.min() < -32768 or fp_data.max() > 32767:
        logging.error("Overflow on conversion to int16")
        exit()
        of_high = np.where(fp_data>32767)
        fp_data[of_high] = 32767
        of_low = np.where(fp_data<-32768)
        fp_data[of_low] = -32767
        
      fp_data = fp_data.astype(np.short)
      fq = fp_data * shift_right

      return fq

    def quantization_error(self, data, quantized_data):
      mse = (np.square(data - quantized_data)).mean(axis=None)

      logging.info("Quantization error as MSE = {}".format(mse))
      
def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 parse model')
    parser.add_argument('model_name', help='name of the har model')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    quantize = Quantizer(args.model_name)
    quantize.get_weights()