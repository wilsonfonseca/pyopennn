import opennn
import numpy as np

data_set = opennn.DataSet()

data_set.set_data_file_name("../../examples/simple_function_regression/data/simplefunctionregression.dat");

data_set.load_data()

data_set.print_data()

inputs_statistics = data_set.scale_inputs_minimum_maximum()
targets_statistics = data_set.scale_targets_minimum_maximum()

#variables = data_set.get_variables()

#variables.set_use(0, "Input")
#variables.set_use(1, "Target")

#variables.set_name(0, "x")
#variables.set_name(1, "y")

# Pasar OpenNN::Matrix a Eigen::Matrix

#inputs_information = variables.get_inputs_information()
#targets_information = variables.get_targets_information()

#instances = data_set.get_instances()
#instances.set_training()


neural_network = opennn.NeuralNetwork(1, 2, 1)

#inputs_pointer = neural_network.get_inputs()
#inputs_pointer.set_information(inputs_information)

#outputs_pointer = neural_network.get_outputs()
#outputs_pointer.set_information(targets_information)

neural_network.construct_scaling_layer()
scaling_layer_pointer = neural_network.get_scaling_layer()
scaling_layer_pointer.set_statistics(inputs_statistics)
#scaling_layer_pointer.set_scaling_methods("NoScaling")

neural_network.construct_unscaling_layer()
unscaling_layer_pointer = neural_network.get_unscaling_layer()
unscaling_layer_pointer.set_statistics(targets_statistics)

# Pasar Vector< Statistics<double> > a Matrix<double> y luego a Eigen::Matrix<double>

#inputs_statistics = data_set.scale_inputs_minimum_maximum();
#targets_statistics = data_set.scale_targets_minimum_maximum();

