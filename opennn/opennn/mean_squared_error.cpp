/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M E A N   S Q U A R E D   E R R O R   C L A S S                                                            */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "mean_squared_error.h"

namespace OpenNN
{
// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates a mean squared error term not associated to any 
/// neural network and not measured on any data set.
/// It also initializes all the rest of class members to their default values.

MeanSquaredError::MeanSquaredError() : LossIndex()
{
}


// NEURAL NETWORK CONSTRUCTOR

/// Neural network constructor. 
/// It creates a mean squared error term object associated to a 
/// neural network object but not measured on any data set object.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

MeanSquaredError::MeanSquaredError(NeuralNetwork* new_neural_network_pointer)
: LossIndex(new_neural_network_pointer)
{
}


// DATA SET CONSTRUCTOR

/// Data set constructor. 
/// It creates a mean squared error term not associated to any 
/// neural network but to be measured on a given data set object.
/// It also initializes all the rest of class members to their default values.
/// @param new_data_set_pointer Pointer to a data set object.

MeanSquaredError::MeanSquaredError(DataSet* new_data_set_pointer)
: LossIndex(new_data_set_pointer)
{
}


// NEURAL NETWORK AND DATA SET CONSTRUCTOR

/// Neural network and data set constructor. 
/// It creates a mean squared error term object associated to a 
/// neural network and measured on a data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

MeanSquaredError::MeanSquaredError(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
: LossIndex(new_neural_network_pointer, new_data_set_pointer)
{
}


// XML CONSTRUCTOR

/// XML constructor. 
/// It creates a mean squared error object with all pointers set to nullptr. 
/// The object members are loaded by means of a XML document.
/// Please be careful with the format of that file, which is specified in the OpenNN manual.
/// @param mean_squared_error_document TinyXML document with the mean squared error elements.

MeanSquaredError::MeanSquaredError(const tinyxml2::XMLDocument& mean_squared_error_document)
 : LossIndex(mean_squared_error_document)
{
    from_XML(mean_squared_error_document);
}


// COPY CONSTRUCTOR

/// Copy constructor. 
/// It creates a copy of an existing mean squared error object. 
/// @param other_mean_squared_error Mean squared error object to be copied.

MeanSquaredError::MeanSquaredError(const MeanSquaredError& other_mean_squared_error)
: LossIndex(other_mean_squared_error)
{
}


// DESTRUCTOR

/// Destructor.

MeanSquaredError::~MeanSquaredError()
{
}


// METHODS


/// Checks that there are a neural network and a data set associated to the mean squared error, 
/// and that the numbers of inputs and outputs in the neural network are equal to the numbers of inputs and targets in the data set. 
/// If some of the above conditions is not hold, the method throws an exception. 

void MeanSquaredError::check() const
{
   ostringstream buffer;

   // Neural network stuff

   if(!neural_network_pointer)
   {
      buffer << "OpenNN Exception: MeanSquaredError class.\n"
             << "void check() const method.\n"
             << "Pointer to neural network is nullptr.\n";

      throw logic_error(buffer.str());	  
   }

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   if(!multilayer_perceptron_pointer)
   {
      buffer << "OpenNN Exception: MeanSquaredError class.\n"
             << "void check() const method.\n"
             << "Pointer to multilayer perceptron is nullptr.\n";

      throw logic_error(buffer.str());	  
   }

   const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   if(inputs_number == 0)
   {
      buffer << "OpenNN Exception: MeanSquaredError class.\n"
             << "void check() const method.\n"
             << "Number of inputs in multilayer perceptron object is zero.\n";

      throw logic_error(buffer.str());	  
   }

   if(outputs_number == 0)
   {
      buffer << "OpenNN Exception: MeanSquaredError class.\n"
             << "void check() const method.\n"
             << "Number of outputs in multilayer perceptron object is zero.\n";

      throw logic_error(buffer.str());	  
   }

   // Data set stuff

   if(!data_set_pointer)
   {
      buffer << "OpenNN Exception: MeanSquaredError class.\n"
             << "void check() const method.\n"
             << "Pointer to data set is nullptr.\n";

      throw logic_error(buffer.str());	  
   }

   // Sum squared error stuff

   const Variables& variables = data_set_pointer->get_variables();

   const size_t data_set_inputs_number = variables.get_inputs_number();
   const size_t data_set_targets_number = variables.get_targets_number();

   if(inputs_number != data_set_inputs_number)
   {
      buffer << "OpenNN Exception: MeanSquaredError class.\n"
             << "void check() const method.\n"
             << "Number of inputs in multilayer perceptron must be equal to number of inputs in data set.\n";

      throw logic_error(buffer.str());	  
   }

   if(outputs_number != data_set_targets_number)
   {
      buffer << "OpenNN Exception: MeanSquaredError class.\n"
             << "void check() const method.\n"
             << "Number of outputs in multilayer perceptron must be equal to number of targets in data set.\n";

      throw logic_error(buffer.str());
   }
}


double MeanSquaredError::calculate_error(const Matrix<double>& outputs, const Matrix<double>& targets) const
{
    // Control sentence

    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    const size_t instances_number = targets.get_columns_number();

    return outputs.calculate_sum_squared_error(targets)/static_cast<double>(instances_number);
}


double MeanSquaredError::calculate_error(const Vector<double>& parameters) const
{
    const size_t instances_number = data_set_pointer->get_instances().get_instances_number();

    return calculate_error(Vector<size_t>(0,1,instances_number-1), parameters);
}


/// Returns which would be the error term of a neural network for an hypothetical
/// vector of parameters. It does not set that vector of parameters to the neural network. 
/// @param parameters Vector of potential parameters for the neural network associated to the error term.

double MeanSquaredError::calculate_error(const Vector<size_t>& instances_indices, const Vector<double>& parameters) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   check();

   const size_t size = parameters.size();

   const size_t parameters_number = neural_network_pointer->get_parameters_number();

   if(size != parameters_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: MeanSquaredError class.\n"
             << "double calculate_error(const Vector<double>&) const method.\n"
             << "Size(" << size << ") must be equal to number of parameters(" << parameters_number << ").\n";

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


Vector<double> MeanSquaredError::calculate_output_gradient(const Vector<size_t>& instances_indices, const Vector<double>& output, const Vector<double>& target) const
{
    const size_t instances_number = instances_indices.size();

    const Vector<double> output_gradient = (output-target)*(2.0/static_cast<double>(instances_number));

    return(output_gradient);
}


Matrix<double> MeanSquaredError::calculate_output_gradient(const Matrix<double>& outputs, const Matrix<double>& targets) const
{
    const size_t instances_number = targets.get_rows_number();

    return (outputs-targets)*2.0/static_cast<double>(instances_number);
}


Matrix<double> MeanSquaredError::calculate_output_Hessian(const Vector<size_t>& instances_indices, const Vector<double>&, const Vector<double>&) const
{
    const size_t instances_number = instances_indices.size();

    const size_t outputs_number = neural_network_pointer->get_multilayer_perceptron_pointer()->get_outputs_number();

    Matrix<double> output_Hessian(outputs_number, outputs_number);
    output_Hessian.initialize_diagonal(2.0/static_cast<double>(instances_number));

    return(output_Hessian);
}


/// Returns loss vector of the error terms function for the mean squared error.
/// It uses the error back-propagation method.

Vector<double> MeanSquaredError::calculate_error_terms(const Vector<size_t>& instances_indices) const
{
   // Control sentence

   #ifdef __OPENNN_DEBUG__ 

   check();

   #endif

   // Neural network stuff

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   // Data set stuff

   const size_t instances_number = instances_indices.size();

   const Variables& variables = data_set_pointer->get_variables();

   const Vector<size_t> inputs_indices = variables.get_inputs_indices();
   const Vector<size_t> targets_indices = variables.get_targets_indices();

   // Mean squared error stuff

   Vector<double> error_terms(instances_number);

   #pragma omp parallel for

   for(int i = 0; i < static_cast<int>(instances_number); i++)
   {
       const size_t instance_index = instances_indices[static_cast<size_t>(i)];

      // Input vector

      const Vector<double> inputs = data_set_pointer->get_instance(instance_index, inputs_indices);

      // Output vector

      const Vector<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

      // Target vector

      const Vector<double> targets = data_set_pointer->get_instance(instance_index, targets_indices);

      // Error

      error_terms[static_cast<size_t>(i)] = outputs.calculate_euclidean_distance(targets);
   }

   return error_terms/sqrt(static_cast<double>(instances_number));
}


/// Returns the Jacobian matrix of the mean squared error function, whose elements are given by the 
/// derivatives of the squared errors data set with respect to the multilayer perceptron parameters.

Matrix<double> MeanSquaredError::calculate_error_terms_Jacobian(const Vector<size_t>& instances_indices) const
{
   // Control sentence

   #ifdef __OPENNN_DEBUG__ 

   check();

   #endif

   // Neural network stuff

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

   const size_t neural_parameters_number = multilayer_perceptron_pointer->get_parameters_number();

   Vector< Vector< Vector<double> > > first_order_forward_propagation(2);

   // Data set stuff

   const size_t instances_number = instances_indices.size();

   const Variables& variables = data_set_pointer->get_variables();

   const Vector<size_t> inputs_indices = variables.get_inputs_indices();
   const Vector<size_t> targets_indices = variables.get_targets_indices();

   // Loss index

   Vector<double> output_gradient(outputs_number);

   Vector< Vector<double> > layers_delta(layers_number);
   Vector<double> point_gradient(neural_parameters_number);

   Matrix<double> terms_Jacobian(instances_number, neural_parameters_number);

   // Main loop

#pragma omp parallel for private(first_order_forward_propagation, output_gradient, layers_delta, point_gradient)

   for(int i = 0; i < static_cast<int>(instances_number); i++)
   {
       const size_t instance_index = instances_indices[static_cast<size_t>(i)];

      const Vector<double> inputs = data_set_pointer->get_instance(instance_index, inputs_indices);

      const Vector<double> targets = data_set_pointer->get_instance(instance_index, targets_indices);

      first_order_forward_propagation = multilayer_perceptron_pointer->calculate_first_order_forward_propagation(inputs);

      const Vector< Vector<double> >& layers_activation = first_order_forward_propagation[0];
      const Vector< Vector<double> >& layers_activation_derivative = first_order_forward_propagation[1];

      Vector<double> term;
      double term_norm;

         const Vector<double>& outputs = first_order_forward_propagation[0][layers_number-1]; 

         term = (outputs-targets);
         term_norm = term.calculate_L2_norm();

         if(term_norm == 0.0)
         {
             output_gradient.set(outputs_number, 0.0);
         }
         else
         {
            output_gradient = term/term_norm;
         }

         layers_delta = calculate_layers_delta(layers_activation_derivative, output_gradient);

      point_gradient = calculate_point_gradient(inputs, layers_activation, layers_delta);

      terms_Jacobian.set_row(static_cast<size_t>(i), point_gradient);
  }

   return terms_Jacobian/sqrt(static_cast<double>(instances_number));
}


/// Returns a string with the name of the mean squared error loss type, "MEAN_SQUARED_ERROR".

string MeanSquaredError::write_error_term_type() const
{
   return("MEAN_SQUARED_ERROR");
}


/// Serializes the mean squared error object into a XML document of the TinyXML library. 
/// See the OpenNN manual for more information about the format of this document-> 

tinyxml2::XMLDocument* MeanSquaredError::to_XML() const
{
   ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Mean squared error

   tinyxml2::XMLElement* mean_squared_error_element = document->NewElement("MeanSquaredError");

   document->InsertFirstChild(mean_squared_error_element);

   // Display
//   {
//      tinyxml2::XMLElement* element = document->NewElement("Display");
//      mean_squared_error_element->LinkEndChild(element);

//      buffer.str("");
//      buffer << display;

//      tinyxml2::XMLText* text = document->NewText(buffer.str().c_str());
//      element->LinkEndChild(text);
//   }

   return(document);
}


void MeanSquaredError::write_XML(tinyxml2::XMLPrinter&) const
{
    //file_stream.OpenElement("MeanSquaredError");

    //file_stream.CloseElement();
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
