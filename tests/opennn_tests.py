import unittest
import numpy as np
import opennn

""" DATA SET """

class TestDataSet(unittest.TestCase):

    def setUp(self):
        self.data = np.matrix('1 2; 3 4; 5 6')
        self.data_set = opennn.DataSet(self.data)
        self.variables = self.data_set.get_variables()
        pass


    def test_data_set_empty_constructor(self):

        self.data_set = opennn.DataSet()

        self.assertTrue(0 == self.data_set.get_data().size)


    def test_data_set_matrix_constructor(self):

        self.data = np.matrix('1 2; 3 4; 5 6')

        self.data_set = opennn.DataSet(self.data)

        self.assertTrue(np.array_equal(self.data, self.data_set.get_data()))


    def test_data_set_training_and_target_constructor(self):

        self.data_set = opennn.DataSet(2, 3)

        self.assertTrue(np.array_equal([[0, 0, 0],[0, 0, 0]], self.data_set.get_data()))


    def test_data_set_instances_training_and_target_constructor(self):

        self.data_set = opennn.DataSet(3, 1, 2)

        self.assertTrue(np.array_equal([[0, 0, 0], [0, 0, 0], [0, 0, 0]], self.data_set.get_data()))


    def test_data_set_string_url_constructor(self):

        self.data_set = opennn.DataSet("./simple_function_regression.dat")

        self.assertTrue(
            np.array_equal([[0., 0.454], [0.1, 0.723], [0.2, 0.908], [0.3, 0.854], [0.4, 0.587], [0.5, 0.545],
                                        [0.6, 0.306], [0.7, 0.129], [0.8, 0.185], [0.9, 0.263], [1., 0.503]], self.data_set.get_data()))


    def test_data_set_get_instances(self):

        instances_number = self.data_set.get_instances().get_instances_number()

        self.assertEqual(3, instances_number)


    def test_data_set_get_variables(self):

        variables_number = self.data_set.get_variables().get_variables_number()

        self.assertEqual(2, variables_number)


    def test_data_set_set_data_file_name_and_load_data(self):

        self.data_set = opennn.DataSet()
        self.data_set.set_data_file_name("./simple_function_regression.dat")
        self.data_set.load_data()

        self.assertTrue(
            np.array_equal([[0., 0.454], [0.1, 0.723], [0.2, 0.908], [0.3, 0.854], [0.4, 0.587], [0.5, 0.545],
                            [0.6, 0.306], [0.7, 0.129], [0.8, 0.185], [0.9, 0.263], [1., 0.503]],
                           self.data_set.get_data()))


    def test_data_set_get_data(self):

        self.assertTrue(np.array_equal(self.data, self.data_set.get_data()))


    def test_data_set_scale_inputs_minimum_maximum(self):

        self.data_set.scale_inputs_minimum_maximum()

        self.assertTrue(np.array_equal([[-1, 2], [0, 4], [1, 6]], self.data_set.get_data()))


    def test_data_set_scale_targets_minimum_maximum(self):

        self.data_set.scale_targets_minimum_maximum()

        self.assertTrue(np.array_equal([[1, -1], [3, 0], [5, 1]], self.data_set.get_data()))


    def test_variables_empty_constructor(self):

        self.variables = opennn.Variables()

        self.assertTrue(np.array_equal([], self.variables.get_inputs_information()))


    def test_variables_size_one_constructor(self):

        self.variables = opennn.Variables(1)

        self.assertEqual(1, self.variables.get_variables_number())


    def test_variables_size_two_constructor(self):

        self.variables = opennn.Variables(1, 1)

        self.assertEqual(2, self.variables.get_variables_number())


    def test_variables_get_variables_number(self):

        self.assertEqual(2, self.variables.get_variables_number())


    #def test_set_use(self):
#
# class TestInstances(unittest.TestCase):
#
#     def setUp(self):
#         pass
#
#     def test_instances_constructor(self):
#
#         data = np.matrix('1 2; 3 4; 5 6')
#
#         data_set = opennn.DataSet(data)
#
#         instances = opennn.Instances(data_set.get_instances())
#         instances.set_training()
#
#         self.assertEqual(3, instances.get_instances_number())
#         self.assertEqual(3, instances.get_training_instances_number())
#         self.assertEqual(0, instances.get_selection_instances_number())
#         self.assertEqual(0, instances.get_testing_instances_number())
#
#
# """ NEURAL NETWORK """
#
# class TestNeuralNetwork(unittest.TestCase):
#
#     def setUp(self):
#         pass
#
#     def test_neuralNetwork_constructor(self):
#
#         neuralNetwork = opennn.NeuralNetwork(1, 2, 1)
#
#         print(neuralNetwork.get_inputs())
#
#     def test_construct_scaling_layer(self):
#
#         neural_network = opennn.NeuralNetwork(1, 2, 1)
#
#         neural_network.construct_scaling_layer()
#
#         self.assertEqual(neural_network.has_scaling_layer(), true)
#
#         self.assertEqual(neural_network.get_scaling_layer().get_scaling_neurons_number(), 1)
#
#     def test_get_inputs(self):
#
#         neural_network = opennn.NeuralNetwork(2, 3)
#
#         #self.assertNotEqual(neural_network.get_inputs(), NULL)
#
#         #self.assertEqual(neural_network.get_inputs().get_inputs_number, 2)
#
#
#
# class TestInputs(unittest.TestCase):
#
#     def setUp(self):
#         pass
#
#     def test_Inputs_constructor(self):
#
#         neuralNetwork = opennn.NeuralNetwork(1, 2, 1)
#
#         inputs = neuralNetwork.get_inputs()


""" TRAINING STRATEGY """



""" MODEL SELECTION??? """


""" TESTING ANALYSIS """



if __name__ == '__main__':
    unittest.main()
	
