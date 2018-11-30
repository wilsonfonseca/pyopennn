// System includes

#include <iostream>
#include <stdio.h>
#include <string>
#include <fstream>
#include <algorithm>
#include <functional>
#include <limits>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <sstream>
#include <cassert>
#include <iomanip>
#include <iterator>
#include <istream>
#include <map>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <vector>
#include <climits>
#include <time.h>

// Pybind includes

#include "../pybind11/include/pybind11/eigen.h"
#include "../pybind11/include/pybind11/operators.h"
#include "../pybind11/include/pybind11/pybind11.h"
#include "../pybind11/include/pybind11/pytypes.h"
#include "../pybind11/include/pybind11/stl.h"

// OpenNN includes

#include "./opennn/opennn.h"

// Namespaces

using namespace OpenNN;
using namespace std;
namespace py = pybind11;

// OpenNN module

PYBIND11_MODULE(opennn, m) {

    // Data set

    py::class_<DataSet>(m, "DataSet")
        .def(py::init<>())
        .def(py::init<const Eigen::MatrixXd&>())
        .def(py::init<const size_t&, const size_t&>())
        .def(py::init<const size_t&, const size_t&, const size_t&>())
        .def(py::init<const string&>())
        .def(py::init<const DataSet&>())
        .def(py::self == py::self)
        .def("get_variables", (const Variables& (DataSet::*) () const) &DataSet::get_variables)
        .def("set_data_file_name", &DataSet::set_data_file_name)
        .def("load_data", &DataSet::load_data)
        .def("print_data", &DataSet::print_data)
        .def("scale_inputs_minimum_maximum", &DataSet::scale_inputs_minimum_maximum_eigen)
        .def("scale_targets_minimum_maximum", &DataSet::scale_targets_minimum_maximum_eigen)
        .def("get_instances", &DataSet::get_instances)
        .def("get_data", &DataSet::get_data_eigen);

    // Variables

    py::class_<Variables> variables(m, "Variables");

    variables.def(py::init<>())
        .def(py::init<const size_t&>())
        .def(py::init<const size_t&, const size_t&>())
        .def(py::init<const Variables&>())
        .def(py::self == py::self)
        .def("get_variables_number", &Variables::get_variables_number)
        .def("get_inputs_number", &Variables::get_inputs_number)
        .def("get_targets_number", &Variables::get_targets_number)
        .def("get_inputs_name", &Variables::get_inputs_name_std)
        .def("get_targets_name", &Variables::get_targets_name_std)
        .def("get_inputs_information", &Variables::get_inputs_information_vector_of_vector)
        .def("get_targets_information", &Variables::get_targets_information_vector_of_vector)
        .def("set_use", (void (Variables::*) (const size_t&, const string&)) &Variables::set_use)
        .def("set_name", &Variables::set_name)
        .def("set_description", &Variables::set_description)
        .def("set_units", (void (Variables::*) (const size_t&, const string&)) &Variables::set_units);

    py::enum_<Variables::Use>(variables, "Use")
        .value("Input", Variables::Use::Input)
        .value("Target", Variables::Use::Target)
        .value("Time", Variables::Use::Time)
        .value("Unused", Variables::Use::Unused)
        .export_values();

    // Struct Variables::Item

    py::class_<Variables::Item>(m, "VariablesItem")
        .def(py::init<>())
        .def_readwrite("name", &Variables::Item::name)
        .def_readwrite("units", &Variables::Item::units)
        .def_readwrite("description", &Variables::Item::description)
        .def_readwrite("use", &Variables::Item::use);

    // Instances

    py::class_<Instances>(m, "Instances")
        .def(py::init<>())
        .def(py::init<const size_t&>())
        .def(py::init<const Instances&>())
        .def(py::self == py::self)
        .def("get_instances_number", &Instances::get_instances_number)
        .def("get_training_instances_number", &Instances::get_training_instances_number)
        .def("get_selection_instances_number", &Instances::get_selection_instances_number)
        .def("get_testing_instances_number", &Instances::get_testing_instances_number)
        .def("set_training", (void (Instances::*) ()) &Instances::set_training);

    // Neural network

    py::class_<NeuralNetwork>(m, "NeuralNetwork")
        .def(py::init<>())
        .def(py::init<const MultilayerPerceptron&>())
        .def(py::init<const vector<size_t>&>())
        .def(py::init<const size_t&, const size_t&>())
        .def(py::init<const size_t&, const size_t&, const size_t&>())
        .def(py::init<const string&>())
        .def(py::init<const NeuralNetwork&>())
        .def(py::self == py::self)
        .def("get_inputs", &NeuralNetwork::get_inputs_pointer)
        .def("get_outputs", &NeuralNetwork::get_outputs_pointer)
        .def("construct_scaling_layer", &NeuralNetwork::construct_scaling_layer)
        .def("get_scaling_layer", &NeuralNetwork::get_scaling_layer_pointer)
        .def("construct_unscaling_layer", &NeuralNetwork::construct_unscaling_layer)
        .def("get_unscaling_layer", &NeuralNetwork::get_unscaling_layer_pointer);

    // Inputs

    py::class_<Inputs>(m, "Inputs")
        .def(py::init<>())
        .def(py::init<const size_t&>())
        .def(py::init<const Inputs&>())
        .def(py::self == py::self)
        .def("set_information", &Inputs::set_information_vector_of_vector);

    // Outputs

    py::class_<Outputs>(m, "Outputs")
        .def(py::init<>())
        .def(py::init<const size_t&>())
        .def(py::init<const Outputs&>())
        .def(py::self == py::self)
        .def("set_information", &Outputs::set_information_vector_of_vector);

    // ScalingLayer

    py::class_<ScalingLayer>(m, "ScalingLayer")
        .def(py::init<>())
        .def(py::init<const size_t&>())
        .def(py::init<const ScalingLayer&>())
        .def(py::self == py::self)
        .def("set_statistics", &ScalingLayer::set_statistics_eigen)
        .def("set_scaling_methods", (void (ScalingLayer::*) (const string&)) &ScalingLayer::set_scaling_methods);

    // UnscalingLayer

    py::class_<UnscalingLayer>(m, "UnscalingLayer")
        .def(py::init<>())
        .def(py::init<const size_t&>())
        .def(py::init<const UnscalingLayer&>())
        .def(py::self == py::self)
        .def("set_statistics", &UnscalingLayer::set_statistics_eigen)
        .def("set_unscaling_method", (void (UnscalingLayer::*) (const string&)) &UnscalingLayer::set_unscaling_method);


    // Training strategy

    py::class_<TrainingStrategy>(m, "TrainingStrategy")
        .def(py::init<>())
        .def(py::init<NeuralNetwork*, DataSet*>())
        .def(py::init<const string&>())
        .def("set_loss_method", (void (TrainingStrategy::*) (const string&)) &TrainingStrategy::set_loss_method)
        .def("set_training_method", (void (TrainingStrategy::*) (const string&)) &TrainingStrategy::set_training_method)
        .def("perform_training", &TrainingStrategy::perform_training);


    // Model selection

    py::class_<ModelSelection>(m, "ModelSelection")
        .def(py::init<>())
        .def(py::init<TrainingStrategy*>())
        .def(py::init<const string&>())
        .def("set_order_selection_method", (void (ModelSelection::*) (const string&)) &ModelSelection::set_order_selection_method)
        .def("set_inputs_selection_method", (void (ModelSelection::*) (const string&)) &ModelSelection::set_inputs_selection_method)
        .def("perform_order_selection", &ModelSelection::perform_order_selection)
        .def("perform_inputs_selection", &ModelSelection::perform_inputs_selection);

    // Testing analysis

    py::class_<TestingAnalysis>(m, "TestingAnalysis")
        .def(py::init<>())
        .def(py::init<NeuralNetwork*, DataSet*>())
        .def(py::init<const string&>());


//    py::class_<SumSquaredError>(m, "SumSquaredError")
//        .def(py::init<>())
//        .def(py::init<NeuralNetwork*, DataSet*>())
//        .def("calculate_error", (double (SumSquaredError::*) (const Vector<size_t>&)) &SumSquaredError::calculate_error);

//    // Loss index

//    py::class_<LossIndex>(m, "LossIndex")
//        .def(py::init<>());


//    // QuasiNewtonMethod

//    py::class_<QuasiNewtonMethod>(m, "QuasiNewtonMethod")
//        .def(py::init<>())
//        .def(py::init<LossIndex*>())
//        .def(py::init<const tinyxml2::XMLDocument&>())
//        .def("set_display_period", &QuasiNewtonMethod::set_display_period)
//        .def("set_maximum_iterations_number", &QuasiNewtonMethod::set_maximum_iterations_number);

//    // Model selection

//    // Testing analysis


// AntColonyOptimization

//    py::class_<AntColonyOptimization>(m, "AntColonyOptimization")
//        .def(py::init<>())
//        .def(py::init<TrainingStrategy*>())
//        .def(py::init<const tinyxml2::XMLDocument&>())
//        .def(py::init<const string&>())
//        .def("get_maximum_selection_failures", &AntColonyOptimization::get_maximum_selection_failures)
//        .def("set_default", &AntColonyOptimization::set_default)
//        .def("set_maximum_selection_failures", &AntColonyOptimization::set_maximum_selection_failures)
//        .def("perform_minimum_model_evaluation", &AntColonyOptimization::perform_minimum_model_evaluation)
//        .def("perform_maximum_model_evaluation", &AntColonyOptimization::perform_maximum_model_evaluation)
//        .def("perform_mean_model_evaluation", &AntColonyOptimization::perform_mean_model_evaluation)
//        .def("perform_model_evaluation", &AntColonyOptimization::perform_model_evaluation)
//        .def("chose_paths", &AntColonyOptimization::chose_paths)
//        .def("evaluate_ants", &AntColonyOptimization::evaluate_ants)
//        .def("perform_order_selection", &AntColonyOptimization::perform_order_selection)
//        .def("to_string_matrix", &AntColonyOptimization::to_string_matrix)
//        .def("to_XML", &AntColonyOptimization::to_XML)
//        .def("from_XML", &AntColonyOptimization::from_XML)
//        .def("write_XML", &AntColonyOptimization::write_XML)
//        .def("save", &AntColonyOptimization::save)
//        .def("load", &AntColonyOptimization::load);

// AssociationRules

//    py::class_<AssociationRules>(m, "AssociationRules")
//        .def(py::init<>())
//        .def("get_sparse_matrix", &AssociationRules::get_sparse_matrix)
//        .def("get_minimum_support", &AssociationRules::get_minimum_support)
//        //.def("get_maximum_time", &AssociationRules::get_maximum_time)
//        .def("get_display", &AssociationRules::get_display)
//        .def("set_sparse_matrix", &AssociationRules::set_sparse_matrix)
//        .def("set_minimum_support", &AssociationRules::set_minimum_support)
//        //.def("set_maximum_time", &AssociationRules::set_maximum_time)
//        .def("set_display", &AssociationRules::set_display)
//        .def("calculate_combinations_number", &AssociationRules::calculate_combinations_number)
//        .def("calculate_combinations", &AssociationRules::calculate_combinations)
//        .def("calculate_support", &AssociationRules::calculate_support)
//        .def("calculate_confidence", &AssociationRules::calculate_confidence)
//        .def("calculate_lift", &AssociationRules::calculate_lift)
//        .def("perform_a_priori_algorithm", &AssociationRules::perform_a_priori_algorithm);

// BoundingLayer

//    py::class_<BoundingLayer>(m, "BoundingLayer")
//        .def(py::init<>())
//        .def(py::init<const size_t&>())
//        .def(py::init<const tinyxml2::XMLDocument&>())
//        .def(py::init<const BoundingLayer&>())
//        .def(py::self == py::self)
//        .def("is_empty", &BoundingLayer::get_bounding_neurons_number)
//        .def("get_bounding_method", &BoundingLayer::get_bounding_method)
//        .def("write_bounding_method", &BoundingLayer::write_bounding_method)
//        .def("get_lower_bounds", &BoundingLayer::get_lower_bounds)
//        .def("get_lower_bound", &BoundingLayer::get_lower_bound)
//        .def("get_upper_bounds", &BoundingLayer::get_upper_bounds)
//        .def("get_upper_bound", &BoundingLayer::get_upper_bound)
//        .def("get_bounds", &BoundingLayer::get_bounds)
//        .def("set", (void (BoundingLayer::*) ()) &BoundingLayer::set)
//        .def("set", (void (BoundingLayer::*) (const size_t&)) &BoundingLayer::set)
//        .def("set", (void (BoundingLayer::*) (const tinyxml2::XMLDocument&)) &BoundingLayer::set)
//        .def("set", (void (BoundingLayer::*) (const BoundingLayer&)) &BoundingLayer::set)
//        .def("set_bounding_method", (void (BoundingLayer::*) (const BoundingLayer::BoundingMethod&)) &BoundingLayer::set_bounding_method)
//        .def("set_bounding_method", (void (BoundingLayer::*) (const string&)) &BoundingLayer::set_bounding_method)
//        .def("set_lower_bounds", &BoundingLayer::set_lower_bounds)
//        .def("set_lower_bound", &BoundingLayer::set_lower_bound)
//        .def("set_upper_bounds", &BoundingLayer::set_upper_bounds)
//        .def("set_upper_bound", &BoundingLayer::set_upper_bound)
//        .def("set_bounds", &BoundingLayer::set_bounds)
//        .def("set_display", &BoundingLayer::set_display)
//        .def("set_default", &BoundingLayer::set_default)
//        .def("prune_bounding_neuron", &BoundingLayer::prune_bounding_neuron)
//        .def("initialize_random", &BoundingLayer::initialize_random)
//        .def("calculate_outputs", &BoundingLayer::calculate_outputs)
//        .def("calculate_derivative", &BoundingLayer::calculate_derivative)
//        .def("calculate_second_derivative", &BoundingLayer::calculate_second_derivative)
//        .def("arrange_Jacobian", &BoundingLayer::arrange_Jacobian)
//        .def("arrange_Hessian_form", &BoundingLayer::arrange_Hessian_form)
//        .def("write_expression", &BoundingLayer::write_expression)
//        .def("write_expression_php", &BoundingLayer::write_expression_php)
//        .def("object_to_string", &BoundingLayer::object_to_string)
//        .def("to_XML", &BoundingLayer::to_XML)
//        .def("from_XML", &BoundingLayer::from_XML)
//        .def("write_XML", &BoundingLayer::write_XML);

//    py::enum_<BoundingLayer::BoundingMethod>(m, "BoundingMethod")
//        .value("NoBounding", BoundingLayer::BoundingMethod::NoBounding)
//        .value("Bounding", BoundingLayer::BoundingMethod::Bounding)
//        .export_values();

//    py::class_<String>(m, "String")
//        .def(py::init<>())
//        .def(py::init<const string&>())
//        .def(py::init<const char*>())
//        .def("split", &String::split)
//        .def("trim", &String::trim)
//        .def("get_trimmed", &String::get_trimmed);

//    py::class_<ConditionsLayer>(m, "ConditionsLayer")
//        .def(py::init<>())
//        .def(py::init<const size_t&, const size_t&>())
//        .def(py::init<const tinyxml2::XMLDocument&>())
//        .def(py::init<const ConditionsLayer&>())
//        .def(py::self == py::self)
//        .def("get_external_inputs_number", &ConditionsLayer::get_external_inputs_number)
//        .def("get_conditions_neurons_number", &ConditionsLayer::get_conditions_neurons_number)
//        .def("get_conditions_method", &ConditionsLayer::get_conditions_method)
//        .def("write_conditions_method", &ConditionsLayer::write_conditions_method)
//        .def("get_external_input_values", &ConditionsLayer::get_external_input_values)
//        .def("get_external_input_value", &ConditionsLayer::get_external_input_value)
//        .def("get_output_values", &ConditionsLayer::get_output_values)
//        .def("get_output_value", &ConditionsLayer::get_output_value)
//        .def("get_display", &ConditionsLayer::get_display)
//        .def("set", (void (ConditionsLayer::*) ()) &ConditionsLayer::set)
//        .def("set", (void (ConditionsLayer::*) (const size_t&, const size_t&)) &ConditionsLayer::set)
//        .def("set", (void (ConditionsLayer::*) (const ConditionsLayer&)) &ConditionsLayer::set)
//        .def("set_external_inputs_number", &ConditionsLayer::set_external_inputs_number)
//        .def("set_conditions_neurons_number", &ConditionsLayer::set_conditions_neurons_number)
//        .def("set_conditions_method", (void (ConditionsLayer::*) (const ConditionsLayer::ConditionsMethod&)) &ConditionsLayer::set_conditions_method)
//        .def("set_conditions_method", (void (ConditionsLayer::*) (const string&)) &ConditionsLayer::set_conditions_method)
//        .def("set_external_input_values", &ConditionsLayer::set_external_input_values)
//        .def("set_external_input_value", &ConditionsLayer::set_external_input_value)
//        .def("set_output_values", &ConditionsLayer::set_output_values)
//        .def("set_output_value", &ConditionsLayer::set_output_value)
//        .def("set_display", &ConditionsLayer::set_display)
//        .def("set_default", &ConditionsLayer::set_default)
//        .def("initialize_random", &ConditionsLayer::initialize_random)
//        .def("check", &ConditionsLayer::check)
//        .def("calculate_particular_solution", &ConditionsLayer::calculate_particular_solution)
//        .def("calculate_particular_solution_Jacobian", &ConditionsLayer::calculate_particular_solution_Jacobian)
//        .def("calculate_particular_solution_Hessian_form", &ConditionsLayer::calculate_particular_solution_Hessian_form)
//        .def("calculate_homogeneous_solution", &ConditionsLayer::calculate_homogeneous_solution)
//        .def("calculate_homogeneous_solution_Jacobian", &ConditionsLayer::calculate_homogeneous_solution_Jacobian)
//        .def("calculate_homogeneous_solution_Hessian_form", &ConditionsLayer::calculate_homogeneous_solution_Hessian_form)
//        .def("calculate_outputs", &ConditionsLayer::calculate_outputs)
//        .def("calculate_Jacobian", &ConditionsLayer::calculate_Jacobian)
//        .def("calculate_Hessian_form", &ConditionsLayer::calculate_Hessian_form)
//        .def("calculate_one_condition_particular_solution", &ConditionsLayer::calculate_one_condition_particular_solution)
//        .def("calculate_one_condition_particular_solution_Jacobian", &ConditionsLayer::calculate_one_condition_particular_solution_Jacobian)
//        .def("calculate_one_condition_particular_solution_Hessian_form", &ConditionsLayer::calculate_one_condition_particular_solution_Hessian_form)
//        .def("calculate_one_condition_homogeneous_solution", &ConditionsLayer::calculate_one_condition_homogeneous_solution)
//        .def("calculate_one_condition_homogeneous_solution_Jacobian", &ConditionsLayer::calculate_one_condition_homogeneous_solution_Jacobian)
//        .def("calculate_one_condition_homogeneous_solution_Hessian_form", &ConditionsLayer::calculate_one_condition_homogeneous_solution_Hessian_form)
//        .def("calculate_two_conditions_particular_solution", &ConditionsLayer::calculate_two_conditions_particular_solution)
//        .def("calculate_two_conditions_particular_solution_Jacobian", &ConditionsLayer::calculate_two_conditions_particular_solution_Jacobian)
//        .def("calculate_two_conditions_particular_solution_Hessian_form", &ConditionsLayer::calculate_two_conditions_particular_solution_Hessian_form)
//        .def("calculate_two_conditions_homogeneous_solution", &ConditionsLayer::calculate_two_conditions_homogeneous_solution)
//        .def("calculate_two_conditions_homogeneous_solution_Jacobian", &ConditionsLayer::calculate_two_conditions_homogeneous_solution_Jacobian)
//        .def("calculate_two_conditions_homogeneous_solution_Hessian_form", &ConditionsLayer::calculate_two_conditions_homogeneous_solution_Hessian_form)
//        .def("write_particular_solution_expression", &ConditionsLayer::write_particular_solution_expression)
//        .def("write_homogeneous_solution_expression", &ConditionsLayer::write_homogeneous_solution_expression)
//        .def("write_one_condition_particular_solution_expression", &ConditionsLayer::write_one_condition_particular_solution_expression)
//        .def("write_one_condition_homogeneous_solution_expression", &ConditionsLayer::write_one_condition_homogeneous_solution_expression)
//        .def("write_two_conditions_particular_solution_expression", &ConditionsLayer::write_two_conditions_particular_solution_expression)
//        .def("write_two_conditions_homogeneous_solution_expression", &ConditionsLayer::write_two_conditions_homogeneous_solution_expression)
//        .def("write_output_expression", &ConditionsLayer::write_output_expression)
//        .def("write_expression", &ConditionsLayer::write_expression)
//        .def("object_to_string", &ConditionsLayer::object_to_string)
//        .def("to_XML", &ConditionsLayer::to_XML)
//        .def("from_XML", &ConditionsLayer::from_XML)
//        .def("write_XML", &ConditionsLayer::write_XML);


}


