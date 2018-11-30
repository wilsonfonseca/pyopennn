/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   S U M   S Q U A R E D   E R R O R   C L A S S                                                              */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "sum_squared_error.h"

namespace OpenNN
{

/// Default constructor. 
/// It creates a sum squared error term not associated to any neural network and not measured on any data set.
/// It also initializes all the rest of class members to their default values.

SumSquaredError::SumSquaredError() : LossIndex()
{
}


/// Neural network constructor. 
/// It creates a sum squared error term associated to a neural network but not measured on any data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

SumSquaredError::SumSquaredError(NeuralNetwork* new_neural_network_pointer) 
: LossIndex(new_neural_network_pointer)
{
}


/// Data set constructor. 
/// It creates a sum squared error not associated to any neural network but to be measured on a data set object.
/// It also initializes all the rest of class members to their default values.
/// @param new_data_set_pointer Pointer to a data set object.

SumSquaredError::SumSquaredError(DataSet* new_data_set_pointer)
: LossIndex(new_data_set_pointer)
{
}


/// Neural network and data set constructor. 
/// It creates a sum squared error associated to a neural network and measured on a data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

SumSquaredError::SumSquaredError(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
 : LossIndex(new_neural_network_pointer, new_data_set_pointer)
{
}


/// XML constructor. 
/// It creates a sum squared error not associated to any neural network and not measured on any data set.
/// It also sets all the rest of class members from a TinyXML document.
/// @param sum_squared_error_document XML document with the class members.

SumSquaredError::SumSquaredError(const tinyxml2::XMLDocument& sum_squared_error_document)
 : LossIndex(sum_squared_error_document)
{
    from_XML(sum_squared_error_document);
}


/// Copy constructor. 
/// It creates a sum squared error not associated to any neural network and not measured on any data set.
/// It also sets all the rest of class members from another sum squared error object.
/// @param new_sum_squared_error Object to be copied. 

SumSquaredError::SumSquaredError(const SumSquaredError& new_sum_squared_error)
 : LossIndex(new_sum_squared_error)
{

}


// DESTRUCTOR

/// Destructor.

SumSquaredError::~SumSquaredError() 
{
}


// METHODS

/// Returns the loss value of a neural network according to the sum squared error on a data set.

double SumSquaredError:: calculate_error(const Matrix<double>& outputs, const Matrix<double>& targets) const
{
   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   return outputs.calculate_sum_squared_error(targets);
}


double SumSquaredError::calculate_error(const Vector<size_t>& instances_indices, const Vector<double>& parameters) const
{
   // Neural network stuff

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   // Sum squared error stuff

   #ifdef __OPENNN_DEBUG__

   ostringstream buffer;

   const size_t size = parameters.size();

   const size_t parameters_number = neural_network_pointer->get_parameters_number();

   if(size != parameters_number)
   {
      buffer << "OpenNN Exception: SumSquaredError class." << endl
             << "double calculate_error(const Vector<double>&) const method." << endl
             << "Size(" << size << ") must be equal to number of parameters(" << parameters_number << ")." << endl;

      throw logic_error(buffer.str());
   }

   #endif

   // Data set stuff

   const Matrix<double> inputs = data_set_pointer->get_input_data(instances_indices);

   const Matrix<double> targets = data_set_pointer->get_target_data(instances_indices);

   // Neural network stuff

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs, parameters);

   return calculate_error(outputs, targets);
}


// Test combination

Vector<double> SumSquaredError::calculate_output_gradient(const Vector<size_t>&, const Vector<double>& output, const Vector<double>& target) const
{
    return (output-target)*2.0;
}


Matrix<double> SumSquaredError::calculate_output_gradient(const Matrix<double>& outputs, const Matrix<double>& targets) const
{
    return (outputs-targets)*2.0;
}


Matrix<double> SumSquaredError::calculate_output_Hessian(const Vector<size_t>&, const Vector<double>&, const Vector<double>&) const
{
    const size_t outputs_number = neural_network_pointer->get_multilayer_perceptron_pointer()->get_outputs_number();

    Matrix<double> output_Hessian(outputs_number, outputs_number);
    output_Hessian.initialize_diagonal(2.0);

    return(output_Hessian);
}


Vector<double> SumSquaredError::calculate_points_errors_layer_combinations(const size_t& layer_index, const Matrix<double>& layer_combinations) const
{
    MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t instances_number = data_set_pointer->get_instances_pointer()->get_instances_number();
    const Vector<size_t> instances_indices(0, 1, instances_number-1);

    const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs_layer_combinations(layer_index, layer_combinations);
    const Matrix<double> targets = data_set_pointer->get_target_data(instances_indices);

    return outputs.calculate_sum_squared_error_rows(targets);
}


/// Calculates the Hessian matrix for a neural network with one hidden layer and an arbitrary number of
/// inputs, perceptrons in the hidden layer and outputs.

Matrix<double> SumSquaredError::calculate_single_hidden_layer_Hessian() const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif
/*
    // Neural network stuff

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();
    const size_t layers_number = 2;

    const size_t parameters_number = multilayer_perceptron_pointer->get_parameters_number();

    Vector< Vector< Vector<double> > > second_order_forward_propagation(3);

    Vector < Vector< Vector<double> > > perceptrons_combination_parameters_gradient(layers_number);
    Matrix < Matrix<double> > interlayers_combination_combination_Jacobian;

    // Data set stuff

    const Instances& instances = data_set_pointer->get_instances();

    const Vector<size_t> training_indices = instances.get_training_indices();

    size_t training_index;

    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> inputs_indices = variables.get_inputs_indices();
    const Vector<size_t> targets_indices = variables.get_targets_indices();

    Vector<double> inputs(inputs_number);
    Vector<double> targets(outputs_number);

    // Sum squared error stuff

    Vector< Vector<double> > layers_delta(layers_number);

    Vector<double> output_gradient(outputs_number);
    Matrix<double> output_Hessian(outputs_number, outputs_number);

    Matrix<double> single_hidden_layer_Hessian(parameters_number, parameters_number, 0.0);

    const size_t i = 0;

    training_index = training_indices[i];

    inputs = data_set_pointer->get_instance(training_index, inputs_indices);

    targets = data_set_pointer->get_instance(training_index, targets_indices);

    second_order_forward_propagation = multilayer_perceptron_pointer->calculate_second_order_forward_propagation(inputs);

    const Vector< Vector<double> >& layers_activation = second_order_forward_propagation[0];
    const Vector< Vector<double> >& layers_activation_derivative = second_order_forward_propagation[1];
    const Vector< Vector<double> >& layers_activation_second_derivative = second_order_forward_propagation[2];

    Vector< Vector<double> > layers_inputs(layers_number);

    layers_inputs[0] = inputs;

    for(size_t j = 1; j < layers_number; j++)
    {
        layers_inputs[j] = layers_activation[j-1];
    }

    perceptrons_combination_parameters_gradient = multilayer_perceptron_pointer->calculate_perceptrons_combination_parameters_gradient(layers_inputs);

    interlayers_combination_combination_Jacobian = multilayer_perceptron_pointer->calculate_interlayers_combination_combination_Jacobian(inputs);

    output_gradient = calculate_output_gradient(layers_activation[layers_number-1], targets);

    output_Hessian = calculate_output_Hessian(layers_activation[layers_number-1], targets);

    layers_delta = calculate_layers_delta(layers_activation_derivative, output_gradient);

    const size_t first_layer_parameters_number = multilayer_perceptron_pointer->get_layer(0).get_parameters().size();
    const size_t second_layer_parameters_number = multilayer_perceptron_pointer->get_layer(1).get_parameters().size();

    Vector<size_t> parameter_indices(3);

    size_t layer_index_i;
    size_t neuron_index_i;
    size_t parameter_index_i;

    size_t layer_index_j;
    size_t neuron_index_j;
    size_t parameter_index_j;

    const Matrix<double> output_interlayers_Delta =
   (output_Hessian
     * layers_activation_derivative[layers_number-1]
     * layers_activation_derivative[layers_number-1]
     + output_gradient
     * layers_activation_second_derivative[layers_number-1]);

    // Both weights in the second layer

    for(size_t i = first_layer_parameters_number; i < second_layer_parameters_number + first_layer_parameters_number; i++)
    {
        parameter_indices = multilayer_perceptron_pointer->get_parameter_indices(i);
        layer_index_i = parameter_indices[0];
        neuron_index_i = parameter_indices[1];
        parameter_index_i = parameter_indices[2];

        for(size_t j = first_layer_parameters_number; j < second_layer_parameters_number + first_layer_parameters_number; j++)
        {
            parameter_indices = multilayer_perceptron_pointer->get_parameter_indices(j);
            layer_index_j = parameter_indices[0];
            neuron_index_j = parameter_indices[1];
            parameter_index_j = parameter_indices[2];           

            single_hidden_layer_Hessian(i,j) =
            perceptrons_combination_parameters_gradient[layer_index_i][neuron_index_i][parameter_index_i]
            *perceptrons_combination_parameters_gradient[layer_index_j][neuron_index_j][parameter_index_j]
            *calculate_Kronecker_delta(neuron_index_i,neuron_index_j)
            *output_interlayers_Delta(neuron_index_j,neuron_index_i);
        }
    }

    // One weight in each layer

    Matrix<double> second_layer_weights = multilayer_perceptron_pointer->get_layer(1).get_synaptic_weights();

    for(size_t i = 0; i < first_layer_parameters_number; i++)
    {
        parameter_indices = multilayer_perceptron_pointer->get_parameter_indices(i);
        layer_index_i = parameter_indices[0];
        neuron_index_i = parameter_indices[1];
        parameter_index_i = parameter_indices[2];

        for(size_t j = first_layer_parameters_number; j < first_layer_parameters_number + second_layer_parameters_number; j++)
        {
            parameter_indices = multilayer_perceptron_pointer->get_parameter_indices(j);
            layer_index_j = parameter_indices[0];
            neuron_index_j = parameter_indices[1];
            parameter_index_j = parameter_indices[2];

            single_hidden_layer_Hessian(i,j) =
            (perceptrons_combination_parameters_gradient[layer_index_i][neuron_index_i][parameter_index_i]
             *perceptrons_combination_parameters_gradient[layer_index_j][neuron_index_j][parameter_index_j]
             *layers_activation_derivative[layer_index_i][neuron_index_i]
             *second_layer_weights(neuron_index_j, neuron_index_i)
             *output_interlayers_Delta(neuron_index_j, neuron_index_j)
             +perceptrons_combination_parameters_gradient[layer_index_i][neuron_index_i][parameter_index_i]
             *layers_activation_derivative[layer_index_i][neuron_index_i]
             *layers_delta[layer_index_j][neuron_index_j]
             *calculate_Kronecker_delta(parameter_index_j,neuron_index_i+1));
        }
    }

    // Both weights in the first layer

    for(size_t i = 0; i < first_layer_parameters_number; i++)
    {
        parameter_indices = multilayer_perceptron_pointer->get_parameter_indices(i);
        layer_index_i = parameter_indices[0];
        neuron_index_i = parameter_indices[1];
        parameter_index_i = parameter_indices[2];

        for(size_t j = 0; j < first_layer_parameters_number; j++)
        {
            parameter_indices = multilayer_perceptron_pointer->get_parameter_indices(j);
            layer_index_j = parameter_indices[0];
            neuron_index_j = parameter_indices[1];
            parameter_index_j = parameter_indices[2];

            double sum = 0.0;

            for(size_t k = 0; k < outputs_number; k++)
            {
                sum += second_layer_weights(k, neuron_index_i)
                       *second_layer_weights(k, neuron_index_j)
                       *output_interlayers_Delta(k,k);
            }

            single_hidden_layer_Hessian(i, j) =
                    perceptrons_combination_parameters_gradient[layer_index_i][neuron_index_i][parameter_index_i]
                    *perceptrons_combination_parameters_gradient[layer_index_j][neuron_index_j][parameter_index_j]
                    *(layers_activation_derivative[layer_index_i][neuron_index_i]
                    *layers_activation_derivative[layer_index_j][neuron_index_j]
                    *sum
                    +layers_activation_second_derivative[layer_index_j][neuron_index_j]
                    *calculate_Kronecker_delta(neuron_index_j,neuron_index_i)
                    *second_layer_weights.get_column(neuron_index_j).dot(layers_delta[1]));
        }
    }

    // Hessian

    for(size_t i = 0; i < parameters_number; i++)
    {
        for(size_t j = 0; j < parameters_number; j++)
        {
            single_hidden_layer_Hessian(j,i) = single_hidden_layer_Hessian(i,j);
        }
    }

    return single_hidden_layer_Hessian;
*/
    return Matrix<double>();
}


/// Calculates the squared error terms for each instance, and returns it in a vector of size the number training instances. 

Vector<double> SumSquaredError::calculate_error_terms(const Vector<size_t>&) const
{
   // Control sentence(if debug)
/*
   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   // Neural network stuff

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   // Data set stuff

   const Instances& instances = data_set_pointer->get_instances();

   const Vector<size_t> training_indices = instances.get_training_indices();

   const size_t training_instances_number = training_indices.size();

   const Variables& variables = data_set_pointer->get_variables();

   const Vector<size_t> inputs_indices = variables.get_inputs_indices();
   const Vector<size_t> targets_indices = variables.get_targets_indices();

   // Loss index stuff

   Vector<double> error_terms(training_instances_number);

   #pragma omp parallel for

   for(int i = 0; i < static_cast<int>(training_instances_number); i++)
   {
       const size_t training_index = training_indices[i];

      // Input vector

      const Vector<double> inputs = data_set_pointer->get_instance(training_index, inputs_indices);

      // Output vector

      const Vector<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

      // Target vector

      const Vector<double> targets = data_set_pointer->get_instance(training_index, targets_indices);

      // Error

      error_terms[i] = outputs.calculate_euclidean_distance(targets);
   }

   return(error_terms);
*/
    return Vector<double>();
}

/*
/// Returns the error terms vector for a hypotetical vector of parameters. 
/// @param parameters Neural network parameters for which the error terms vector is to be computed. 

Vector<double> SumSquaredError::calculate_training_terms(const Vector<double>& parameters) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__
   
   check();

   const size_t size = parameters.size();

   const size_t parameters_number = neural_network_pointer->get_parameters_number();

   if(size != parameters_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: SumSquaredError class." << endl
             << "double calculate_error_terms(const Vector<double>&) const method." << endl
             << "Size(" << size << ") must be equal to number of neural network parameters(" << parameters_number << ")." << endl;

      throw logic_error(buffer.str());	  
   }

   #endif

   NeuralNetwork neural_network_copy(*neural_network_pointer);

   neural_network_copy.set_parameters(parameters);

   SumSquaredError sum_squared_error_copy(*this);

   sum_squared_error_copy.set_neural_network_pointer(&neural_network_copy);

   return(sum_squared_error_copy.calculate_error_terms());
}
*/

/// Returns the terms_Jacobian matrix of the sum squared error function, whose elements are given by the 
/// derivatives of the squared errors data set with respect to the multilayer perceptron parameters.
/// The terms_Jacobian matrix here is computed using a back-propagation algorithm.

Matrix<double> SumSquaredError::calculate_error_terms_Jacobian(const Vector<size_t>&) const
{
   #ifdef __OPENNN_DEBUG__

   check();

   #endif 
/*
   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();
   const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

   const size_t neural_parameters_number = multilayer_perceptron_pointer->get_parameters_number();

   Vector< Vector< Vector<double> > > first_order_forward_propagation(2);

   Vector< Vector<double> > layers_inputs(layers_number);
   Vector< Matrix<double> > layers_combination_parameters_Jacobian(layers_number);

   // Data set

   const Instances& instances = data_set_pointer->get_instances();

   const Vector<size_t> training_indices = instances.get_training_indices();

   const size_t training_instances_number = training_indices.size();

   const Variables& variables = data_set_pointer->get_variables();

   const Vector<size_t> inputs_indices = variables.get_inputs_indices();
   const Vector<size_t> targets_indices = variables.get_targets_indices();

   // Loss index

   Vector<double> output_gradient(outputs_number);

   Vector< Vector<double> > layers_delta(layers_number);
   Vector<double> point_gradient(neural_parameters_number);

   Matrix<double> terms_Jacobian(training_instances_number, neural_parameters_number);

   // Main loop

   #pragma omp parallel for private(first_order_forward_propagation, layers_inputs, \
    layers_combination_parameters_Jacobian, output_gradient, layers_delta, point_gradient)

   for(int i = 0; i < static_cast<int>(training_instances_number); i++)
   {
       const size_t training_index = training_indices[i];

      const Vector<double> inputs = data_set_pointer->get_instance(training_index, inputs_indices);

      const Vector<double> targets = data_set_pointer->get_instance(training_index, targets_indices);

      first_order_forward_propagation = multilayer_perceptron_pointer->calculate_first_order_forward_propagation(inputs);

      layers_inputs = multilayer_perceptron_pointer->get_layers_input(inputs, first_order_forward_propagation);

      layers_combination_parameters_Jacobian = multilayer_perceptron_pointer->calculate_layers_combination_parameters_Jacobian(layers_inputs);

      Vector<double> term;
      double term_norm;

         term = first_order_forward_propagation[0][layers_number-1] - targets;
         term_norm = term.calculate_L2_norm();

         if(term_norm == 0.0)
   	     {
             output_gradient.set(outputs_number, 0.0);
	     }
         else
	     {
            output_gradient = term/term_norm;
	     }

         layers_delta = calculate_layers_delta(first_order_forward_propagation[1], output_gradient);

      point_gradient = calculate_point_gradient(layers_combination_parameters_Jacobian, layers_delta);

      terms_Jacobian.set_row(i, point_gradient);
  }

   return(terms_Jacobian);
*/
   return Matrix<double>();
}


// FirstOrderTerms calculate_first_order_terms() const method

/// Returns the first order loss of the terms loss function.
/// This is a structure containing the error terms vector and the error terms Jacobian.

LossIndex::FirstOrderTerms SumSquaredError::calculate_first_order_terms() const
{
   FirstOrderTerms first_order_terms;

//   first_order_terms.terms = calculate_error_terms();
//   first_order_terms.Jacobian = calculate_error_terms_Jacobian();

   return(first_order_terms);
}


/// Returns the squared errors of the training instances. 

Vector<double> SumSquaredError::calculate_squared_errors() const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   // Neural network stuff

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   // Data set stuff

   const Instances& instances = data_set_pointer->get_instances();

   const Vector<size_t> training_indices = instances.get_training_indices();

   const size_t training_instances_number = training_indices.size();

   const Variables& variables = data_set_pointer->get_variables();

   const Vector<size_t> inputs_indices = variables.get_inputs_indices();
   const Vector<size_t> targets_indices = variables.get_targets_indices();

   const MissingValues missing_values = data_set_pointer->get_missing_values();

   // Loss index

   Vector<double> squared_errors(training_instances_number);

   #pragma omp parallel for

   for(int i = 0; i < static_cast<int>(training_instances_number); i++)
   {
       const size_t training_index = training_indices[i];

      // Input vector

      const Vector<double> inputs = data_set_pointer->get_instance(training_index, inputs_indices);

      // Output vector

      const Vector<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

      // Target vector

      const Vector<double> targets = data_set_pointer->get_instance(training_index, targets_indices);

      // Error

      squared_errors[i] = outputs.calculate_sum_squared_error(targets);
   }

   return(squared_errors);
}


// string write_error_term_type() const method

/// Returns a string with the name of the sum squared error loss type, "SUM_SQUARED_ERROR".

string SumSquaredError::write_error_term_type() const
{
   return("SUM_SQUARED_ERROR");
}


// tinyxml2::XMLDocument* to_XML() method method 

/// Returns a representation of the sum squared error object, in XML format. 

tinyxml2::XMLDocument* SumSquaredError::to_XML() const
{
   ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Sum squared error

   tinyxml2::XMLElement* root_element = document->NewElement("SumSquaredError");

   document->InsertFirstChild(root_element);

   // Display

//   {
//      tinyxml2::XMLElement* display_element = document->NewElement("Display");
//      root_element->LinkEndChild(display_element);

//      buffer.str("");
//      buffer << display;

//      tinyxml2::XMLText* display_text = document->NewText(buffer.str().c_str());
//      display_element->LinkEndChild(display_text);
//   }

   return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

void SumSquaredError::write_XML(tinyxml2::XMLPrinter&) const
{
    //file_stream.OpenElement("SumSquaredError");

    //file_stream.CloseElement();
}


// void load(const tinyxml2::XMLDocument&) method

/// Loads a sum squared error object from a XML document.
/// @param document TinyXML document containing the members of the object.

void SumSquaredError::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("SumSquaredError");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SumSquaredError class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Sum squared error element is nullptr.\n";

        throw logic_error(buffer.str());
    }

  // Display
  {
     const tinyxml2::XMLElement* element = root_element->FirstChildElement("Display");

     if(element)
     {
        const string new_display_string = element->GetText();

        try
        {
           set_display(new_display_string != "0");
        }
        catch(const logic_error& e)
        {
           cerr << e.what() << endl;
        }
     }
  }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2018 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
