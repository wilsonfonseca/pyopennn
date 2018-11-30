/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   C R O S S   E N T R O P Y   E R R O R   C L A S S                                                          */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "cross_entropy_error.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates a default cross entropy error term object, 
/// which is not associated to any neural network and not measured on any data set.
/// It also initializes all the rest of class members to their default values.

CrossEntropyError::CrossEntropyError() : LossIndex()
{
}


// NEURAL NETWORK CONSTRUCTOR

/// Neural network constructor. 
/// It creates a cross entropy error term associated to a neural network but not measured on any data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

CrossEntropyError::CrossEntropyError(NeuralNetwork* new_neural_network_pointer)
 : LossIndex(new_neural_network_pointer)
{
}


// DATA SET CONSTRUCTOR

/// Data set constructor. 
/// It creates a cross entropy error not associated to any neural network but to be measured on a data set object.
/// It also initializes all the rest of class members to their default values.
/// @param new_data_set_pointer Pointer to a data set object.

CrossEntropyError::CrossEntropyError(DataSet* new_data_set_pointer) 
: LossIndex(new_data_set_pointer)
{
}


// NEURAL NETWORK AND DATA SET CONSTRUCTOR

/// Neural network and data set constructor. 
/// It creates a cross entropy error term object associated to a neural network and measured on a data set.
/// It also initializes all the rest of class members to their default values:
/// @param new_neural_network_pointer: Pointer to a neural network object.
/// @param new_data_set_pointer: Pointer to a data set object.

CrossEntropyError::CrossEntropyError(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
: LossIndex(new_neural_network_pointer, new_data_set_pointer)
{
}


// XML CONSTRUCTOR

/// XML constructor. 
/// It creates a cross entropy error not associated to any neural network and not measured on any data set.
/// It also sets all the rest of class members from a TinyXML document->
/// @param sum_squared_error_document XML document with the class members. 

CrossEntropyError::CrossEntropyError(const tinyxml2::XMLDocument& sum_squared_error_document)
 : LossIndex(sum_squared_error_document)
{
    from_XML(sum_squared_error_document);
}


// COPY CONSTRUCTOR

/// Copy constructor. 
/// It creates a cross entropy error not associated to any neural network and not measured on any data set.
/// It also sets all the rest of class members from another cross-entropy error object.
/// @param new_cross_entropy_error Object to be copied. 

CrossEntropyError::CrossEntropyError(const CrossEntropyError& new_cross_entropy_error)
 : LossIndex(new_cross_entropy_error)
{

}


// DESTRUCTOR

/// Destructor.

CrossEntropyError::~CrossEntropyError() 
{
}


// ASSIGNMENT OPERATOR

/// Assignment operator. 
/// @param other_cross_entropy_error Object to be copied. 

CrossEntropyError& CrossEntropyError::operator = (const CrossEntropyError& other_cross_entropy_error)
{
   if(this != &other_cross_entropy_error) 
   {
      *neural_network_pointer = *other_cross_entropy_error.neural_network_pointer;
      *data_set_pointer = *other_cross_entropy_error.data_set_pointer;
      display = other_cross_entropy_error.display;
   }

   return(*this);

}

// EQUAL TO OPERATOR

/// Equal to operator. 
/// If compares this object with another object of the same class, and returns true if they are equal, and false otherwise. 
/// @param other_cross_entropy_error Object to be compared with. 

bool CrossEntropyError::operator == (const CrossEntropyError& other_cross_entropy_error) const
{
   if(*neural_network_pointer == *other_cross_entropy_error.neural_network_pointer
   && display == other_cross_entropy_error.display)    
   {
      return(true);
   }
   else
   {
      return(false);  
   }

}


// METHODS


/// Checks that there are a neural network and a data set associated to the cross entropy error, 
/// and that the numbers of inputs and outputs in the neural network are equal to the numbers of inputs and targets in the data set. 

void CrossEntropyError::check() const
{
   ostringstream buffer;

   // Neural network stuff

   if(!neural_network_pointer)
   {
      buffer << "OpenNN Exception: CrossEntropyError class.\n"
             << "void check() const method.\n"
             << "Pointer to neural network is nullptr.\n";

      throw logic_error(buffer.str());	  
   }

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   if(!multilayer_perceptron_pointer)
   {
      buffer << "OpenNN Exception: CrossEntropyError class.\n"
             << "void check() const method.\n"
             << "Pointer to multilayer perceptron is nullptr.\n";

      throw logic_error(buffer.str());	  
   }

   const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   if(inputs_number == 0)
   {
      buffer << "OpenNN Exception: CrossEntropyError class.\n"
             << "void check() const method.\n"
             << "Number of inputs in multilayer perceptron object is zero.\n";

      throw logic_error(buffer.str());	  
   }

   if(outputs_number == 0)
   {
      buffer << "OpenNN Exception: CrossEntropyError class.\n"
             << "void check() const method.\n"
             << "Number of outputs in multilayer perceptron object is zero.\n";

      throw logic_error(buffer.str());	  
   }

   const ProbabilisticLayer* probabilistic_layer_pointer = neural_network_pointer->get_probabilistic_layer_pointer();

   if(!probabilistic_layer_pointer)
   {
      buffer << "OpenNN Exception: CrossEntropyError class.\n"
             << "void check() const method.\n"
             << "Pointer to probabilistic layer is nullptr.\n";

      throw logic_error(buffer.str());	  
   }

   const ProbabilisticLayer::ProbabilisticMethod& outputs_probabilizing_method = probabilistic_layer_pointer->get_probabilistic_method();

   if(outputs_probabilizing_method != ProbabilisticLayer::Softmax)
   {
      buffer << "OpenNN Exception: CrossEntropyError class.\n"
             << "void check() const method.\n"
             << "Probabilistic method is not Softmax.\n";

      throw logic_error(buffer.str());
   }

   // Data set stuff

   if(!data_set_pointer)
   {
      buffer << "OpenNN Exception: CrossEntropyError class.\n"
             << "void check() const method.\n"
             << "Pointer to data set is nullptr.\n";

      throw logic_error(buffer.str());	  
   }

   // Cross-entropy error stuff

   const Variables& variables = data_set_pointer->get_variables();

   const size_t data_set_inputs_number = variables.get_inputs_number();
   const size_t targets_number = variables.get_targets_number();

   if(inputs_number != data_set_inputs_number)
   {
      buffer << "OpenNN Exception: CrossEntropyError class.\n"
             << "void check() const method.\n"
             << "Number of inputs in neural network (" << inputs_number << ") must be equal to number of inputs in data set (" << data_set_inputs_number << ").\n";

      throw logic_error(buffer.str());	  
   }

   if(outputs_number != targets_number)
   {
      buffer << "OpenNN Exception: CrossEntropyError class.\n"
             << "void check() const method.\n"
             << "Number of outputs in neural network must be equal to number of targets in data set.\n";

      throw logic_error(buffer.str());
   }
}


double CrossEntropyError::calculate_error(const Matrix<double>& outputs, const Matrix<double>& targets) const
{
   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   double cross_entropy_error = 0.0; 

//    #pragma omp parallel for reduction(+ : cross_entropy_error)

//   for(int i = 0; i < static_cast<int>(instances_number); i++)
//   {
//       // Cross entropy error

//      for(size_t j = 0; j < outputs_number; j++)
//      {
//          if(outputs[j] == 0.0)
//          {
//              outputs[j] = 1.0e-6;
//          }
//          else if(outputs[j] == 1.0)
//          {
//              outputs[j] = 1.0 - 1.0e-6;
//          }

//          if(targets[j] == 0.0)
//          {
//              cross_entropy_error -= log(1.0 - outputs[j]);
//          }
//          else if(targets[j] == 1.0)
//          {
//              cross_entropy_error -= log(outputs[j]);
//          }
//      }
//   }

//   return(cross_entropy_error/static_cast<double>(instances_number));


   return 0.0;
}


/// Returns which would be the cross-entropy loss of a neural network for an hypothetical vector of parameters.
/// It does not set that vector of parameters to the neural network.
/// @param parameters Vector of potential parameters for the neural network associated to the error term.

double CrossEntropyError::calculate_error(const Vector<double>& parameters) const
{
    // Neural network stuff

    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    // Cross-entropy error stuff

    #ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    const size_t size = parameters.size();

    const size_t parameters_number = neural_network_pointer->get_parameters_number();

    if(size != parameters_number)
    {
       buffer << "OpenNN Exception: CrossEntropyError class." << endl
              << "double calculate_error(const Vector<double>&) const method." << endl
              << "Size(" << size << ") must be equal to number of parameters(" << parameters_number << ")." << endl;

       throw logic_error(buffer.str());
    }

    #endif

    // Neural network stuff

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    // Data set stuff

    const Instances& instances = data_set_pointer->get_instances();

    const Vector<size_t> training_indices = instances.get_training_indices();

    const size_t training_instances_number = training_indices.size();

    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> inputs_indices = variables.get_inputs_indices();
    const Vector<size_t> targets_indices = variables.get_targets_indices();

    // Cross-entropy error stuff

    double cross_entropy_error = 0.0;

    #pragma omp parallel for reduction(+ : cross_entropy_error)

    for(int i = 0; i < static_cast<int>(training_instances_number); i++)
    {
        const size_t training_index = training_indices[static_cast<size_t>(i)];

       // Input vector

       const Vector<double> inputs = data_set_pointer->get_instance(training_index, inputs_indices);

       // Output vector

       Vector<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs, parameters);

       // Target vector

       Vector<double> targets = data_set_pointer->get_instance(training_index, targets_indices);

       // Cross-entropy error

       for(size_t j = 0; j < outputs_number; j++)
       {
           if(outputs[j] == 0.0)
           {
               outputs[j] = 1.0e-6;
           }
           else if(fabs(outputs[j] - 1) < numeric_limits<double>::epsilon())
           {
               outputs[j] = 1.0 - 1.0e-6;
           }

           if(targets[j] == 0.0)
           {
               cross_entropy_error -= log(1.0 - outputs[j]);
           }
           else if(targets[j] == 1.0)
           {
               cross_entropy_error -= log(outputs[j]);
           }
//           cross_entropy_error -= (targets[j]*log(outputs[j]) + (1.0 - targets[j])*log(1.0 - outputs[j]));
       }
    }

    return(cross_entropy_error/static_cast<double>(training_instances_number));
}


/// Returns the minimum achieveable cross entropy for the training data. 
/// It occurs when all the targets are equal to the outputs for the training data.

double CrossEntropyError::calculate_minimum_loss() const
{
    // Neural network stuff

    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    // Neural network stuff

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    // Data set stuff

    const Instances& instances = data_set_pointer->get_instances();

    const Vector<size_t> training_indices = instances.get_training_indices();

    const size_t training_instances_number = training_indices.size();

    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> inputs_indices = variables.get_inputs_indices();
    const Vector<size_t> targets_indices = variables.get_targets_indices();

    // Cross-entropy error stuff

    double minimum_cross_entropy_error = 0.0;

    #pragma omp parallel for reduction(+ : minimum_cross_entropy_error)

    for(int i = 0; i < static_cast<int>(training_instances_number); i++)
    {
        const size_t training_index = training_indices[static_cast<size_t>(i)];

        // Input vector

       const Vector<double> inputs = data_set_pointer->get_instance(training_index, inputs_indices);

       // Output vector

       Vector<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

       // Target vector

       Vector<double> targets = data_set_pointer->get_instance(training_index, targets_indices);

       // Cross-entropy error

       for(size_t j = 0; j < outputs_number; j++)
       {
           if(outputs[j] == 0.0)
           {
               outputs[j] = 1.0e-6;
           }
           else if(fabs(outputs[j] - 1) < numeric_limits<double>::epsilon())
           {
               outputs[j] = 0.99999;
           }

           if(targets[j] == 0.0)
           {
               targets[j] = 1.0e-6;
           }
           else if(targets[j] == 1.0)
           {
               targets[j] = 0.999999;
           }

           minimum_cross_entropy_error -= (targets[j]*log(outputs[j]/targets[j]) + (1.0 - targets[j])*log((1.0 - outputs[j])/(1.0 - targets[j])));
       }
    }

    return(minimum_cross_entropy_error/static_cast<double>(training_instances_number));
}


/// Returns the minimum achieveable cross entropy for the selection data. 
/// It occurs when all the targets are equal to the outputs for the selection data.

double CrossEntropyError::calculate_minimum_selection_error() const
{
    // Control sentence

    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    // Neural network stuff

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    // Data set stuff

    const Instances& instances = data_set_pointer->get_instances();

    const size_t selection_instances_number = instances.get_selection_instances_number();

    const Vector<size_t> selection_indices = instances.get_selection_indices();

    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> inputs_indices = variables.get_inputs_indices();
    const Vector<size_t> targets_indices = variables.get_targets_indices();

    // Loss index

    double minimum_selection_loss = 0.0;

    #pragma omp parallel for reduction(- : minimum_selection_loss)

    for(int i = 0; i < static_cast<int>(selection_instances_number); i++)
    {
        const size_t selection_index = selection_indices[static_cast<size_t>(i)];

       // Input vector

       const Vector<double> inputs = data_set_pointer->get_instance(selection_index, inputs_indices);

       // Output vector

       Vector<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

       // Target vector

       Vector<double> targets = data_set_pointer->get_instance(selection_index, targets_indices);

       // Cross entropy error

       for(size_t j = 0; j < outputs_number; j++)
       {
           if(outputs[j] == 0.0)
           {
               outputs[j] = 1.0e-6;
           }
           else if(outputs[j] == 1.0)
           {
               outputs[j] = 0.999999;
           }

           if(targets[j] == 0.0)
           {
               targets[j] = 1.0e-6;
           }
           else if(targets[j] == 1.0)
           {
               targets[j] = 0.999999;
           }

           minimum_selection_loss -= (targets[j]*log(outputs[j]/targets[j]) + (1.0 - targets[j])*log((1.0 - outputs[j])/(1.0 - targets[j])));
       }
    }

    return(minimum_selection_loss/static_cast<double>(selection_instances_number));
}


/// Returns the loss value of a neural network according to the cross-entropy error on a data set without the normalization.

double CrossEntropyError::calculate_error_unnormalized() const
{
   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   // Neural network stuff

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   // Data set stuff

   const Instances& instances = data_set_pointer->get_instances();

   const Vector<size_t> training_indices = instances.get_training_indices();

   const size_t training_instances_number = training_indices.size();

   const Variables& variables = data_set_pointer->get_variables();

   const Vector<size_t> inputs_indices = variables.get_inputs_indices();
   const Vector<size_t> targets_indices = variables.get_targets_indices();

   // Cross entropy error

   double cross_entropy_error = 0.0;

    #pragma omp parallel for reduction(+ : cross_entropy_error)

   for(int i = 0; i < static_cast<int>(training_instances_number); i++)
   {
       const size_t training_index = training_indices[static_cast<size_t>(i)];

      // Input vector

      const Vector<double> inputs = data_set_pointer->get_instance(training_index, inputs_indices);

      // Output vector

      Vector<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

      // Target vector

      Vector<double> targets = data_set_pointer->get_instance(training_index, targets_indices);

      // Cross entropy error

      for(size_t j = 0; j < outputs_number; j++)
      {
          if(outputs[j] == 0.0)
          {
              outputs[j] = 1.0e-6;
          }
          else if(outputs[j] == 1.0)
          {
              outputs[j] = 0.999999;
          }

          if(targets[j] == 0.0)
          {
              cross_entropy_error -= (1.0 - targets[j])*log(1.0 - outputs[j]);
          }
          else if(targets[j] == 1.0)
          {
              cross_entropy_error -= targets[j]*log(outputs[j]);
          }
      }
   }

   return(cross_entropy_error);
}


/// Returns which would be the cross-entropy loss without the normalization of a neural network for an hypothetical vector of parameters.
/// It does not set that vector of parameters to the neural network.
/// @param parameters Vector of potential parameters for the neural network associated to the error term.

double CrossEntropyError::calculate_error_unnormalized(const Vector<double>& parameters) const
{
    // Neural network stuff

    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    // Cross-entropy error stuff

    #ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    const size_t size = parameters.size();

    const size_t parameters_number = neural_network_pointer->get_parameters_number();

    if(size != parameters_number)
    {
       buffer << "OpenNN Exception: CrossEntropyError class." << endl
              << "double calculate_error(const Vector<double>&) const method." << endl
              << "Size(" << size << ") must be equal to number of parameters(" << parameters_number << ")." << endl;

       throw logic_error(buffer.str());
    }

    #endif

    // Neural network stuff

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    // Data set stuff

    const Instances& instances = data_set_pointer->get_instances();

    const Vector<size_t> training_indices = instances.get_training_indices();

    const size_t training_instances_number = training_indices.size();

    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> inputs_indices = variables.get_inputs_indices();
    const Vector<size_t> targets_indices = variables.get_targets_indices();

    // Cross-entropy error stuff

    double cross_entropy_error = 0.0;

    #pragma omp parallel for reduction(+ : cross_entropy_error)

    for(int i = 0; i < static_cast<int>(training_instances_number); i++)
    {
        const size_t training_index = training_indices[static_cast<size_t>(i)];

       // Input vector

       const Vector<double> inputs = data_set_pointer->get_instance(training_index, inputs_indices);

       // Output vector

       Vector<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs, parameters);

       // Target vector

       Vector<double> targets = data_set_pointer->get_instance(training_index, targets_indices);

       // Cross-entropy error

       for(size_t j = 0; j < outputs_number; j++)
       {
           if(outputs[j] == 0.0)
           {
               outputs[j] = 1.0e-6;
           }
           else if(fabs(outputs[j] - 1) < numeric_limits<double>::epsilon())
           {
               outputs[j] = 0.99999;
           }

           cross_entropy_error -= (targets[j]*log(outputs[j]) + (1.0 - targets[j])*log(1.0 - outputs[j]));
       }
    }

    return(cross_entropy_error);
}


/// Returns the minimum achieveable cross entropy without the normalization for the training data.
/// It occurs when all the targets are equal to the outputs for the training data.

double CrossEntropyError::calculate_minimum_loss_unnormalized() const
{
    // Neural network stuff

    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    // Neural network stuff

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    // Data set stuff

    const Instances& instances = data_set_pointer->get_instances();

    const Vector<size_t> training_indices = instances.get_training_indices();

    const size_t training_instances_number = training_indices.size();

    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> inputs_indices = variables.get_inputs_indices();
    const Vector<size_t> targets_indices = variables.get_targets_indices();

    // Cross-entropy error stuff

    double minimum_cross_entropy_error = 0.0;

    #pragma omp parallel for reduction(+ : minimum_cross_entropy_error)

    for(int i = 0; i < static_cast<int>(training_instances_number); i++)
    {
       const size_t training_index = training_indices[static_cast<size_t>(i)];

        // Input vector

       const Vector<double> inputs = data_set_pointer->get_instance(training_index, inputs_indices);

       // Output vector

       Vector<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

       // Target vector

       Vector<double> targets = data_set_pointer->get_instance(training_index, targets_indices);

       // Cross-entropy error

       for(size_t j = 0; j < outputs_number; j++)
       {
           if(outputs[j] == 0.0)
           {
               outputs[j] = 1.0e-6;
           }
           else if(fabs(outputs[j] - 1) < numeric_limits<double>::epsilon())
           {
               outputs[j] = 0.99999;
           }

           if(targets[j] == 0.0)
           {
               targets[j] = 1.0e-6;
           }
           else if(targets[j] == 1.0)
           {
               targets[j] = 0.999999;
           }

           minimum_cross_entropy_error -= (targets[j]*log(outputs[j]/targets[j]) + (1.0 - targets[j])*log((1.0 - outputs[j])/(1.0 - targets[j])));
       }
    }

    return(minimum_cross_entropy_error);
}


/// Returns the cross entropy error of the neural network measured on the selection instances of the data set.

double CrossEntropyError::calculate_selection_error_unnormalized() const
{
   // Control sentence

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   // Neural network stuff

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   // Data set stuff

   const Instances& instances = data_set_pointer->get_instances();

   const size_t selection_instances_number = instances.get_selection_instances_number();

   if(selection_instances_number == 0)
   {
       return(0.0);
   }

   const Vector<size_t> selection_indices = instances.get_selection_indices();

   const Variables& variables = data_set_pointer->get_variables();

   const Vector<size_t> inputs_indices = variables.get_inputs_indices();
   const Vector<size_t> targets_indices = variables.get_targets_indices();

   // Loss index

   double selection_loss = 0.0;

   #pragma omp parallel for reduction(- : selection_loss)

   for(int i = 0; i < static_cast<int>(selection_instances_number); i++)
   {
       const size_t selection_index = selection_indices[static_cast<size_t>(i)];

      // Input vector

      const Vector<double> inputs = data_set_pointer->get_instance(selection_index, inputs_indices);

      // Output vector

      Vector<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

      // Target vector

      Vector<double> targets = data_set_pointer->get_instance(selection_index, targets_indices);

      // Cross entropy error

      for(size_t j = 0; j < outputs_number; j++)
      {
          if(outputs[j] == 0.0)
          {
              outputs[j] = 1.0e-6;
          }
          else if(outputs[j] == 1.0)
          {
              outputs[j] = 0.999999;
          }

          selection_loss -= (targets[j]*log(outputs[j]) + (1.0 - targets[j])*log(1.0 - outputs[j]));
      }
   }

   return(selection_loss);
}


/// Returns the minimum achieveable cross entropy without the normalization for the selection data with no normalization.
/// It occurs when all the targets are equal to the outputs for the selection data.

double CrossEntropyError::calculate_minimum_selection_error_unnormalized() const
{
    // Control sentence

    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    // Neural network stuff

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    // Data set stuff

    const Instances& instances = data_set_pointer->get_instances();

    const size_t selection_instances_number = instances.get_selection_instances_number();

    const Vector<size_t> selection_indices = instances.get_selection_indices();

    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> inputs_indices = variables.get_inputs_indices();
    const Vector<size_t> targets_indices = variables.get_targets_indices();

    // Loss index

    double minimum_selection_loss = 0.0;

    #pragma omp parallel for reduction(- : minimum_selection_loss)

    for(int i = 0; i < static_cast<int>(selection_instances_number); i++)
    {
        const size_t selection_index = selection_indices[static_cast<size_t>(i)];

       // Input vector

       const Vector<double> inputs = data_set_pointer->get_instance(selection_index, inputs_indices);

       // Output vector

        Vector<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

       // Target vector

       Vector<double> targets = data_set_pointer->get_instance(selection_index, targets_indices);

       // Cross entropy error

       for(size_t j = 0; j < outputs_number; j++)
       {
           if(outputs[j] == 0.0)
           {
               outputs[j] = 1.0e-6;
           }
           else if(outputs[j] == 1.0)
           {
               outputs[j] = 0.999999;
           }

           if(targets[j] == 0.0)
           {
               targets[j] = 1.0e-6;
           }
           else if(targets[j] == 1.0)
           {
               targets[j] = 0.999999;
           }

           minimum_selection_loss -= (targets[j]*log(outputs[j]/targets[j]) + (1.0 - targets[j])*log((1.0 - outputs[j])/(1.0 - targets[j])));
       }
    }

    return(minimum_selection_loss);
}


/// Returns the cross-entropy error function output gradient of a multilayer perceptron on a data set.
/// It uses the error back-propagation method.
/// @param output Vector of outputs of the neural network.
/// @param target Vector of targets of the data set.

Vector<double> CrossEntropyError::calculate_output_gradient(const Vector<double>& output, const Vector<double>& target) const
{
    const size_t outputs_number = output.size();

    const size_t training_instances_number = data_set_pointer->get_instances().get_training_instances_number();

    Vector<double> output_gradient(outputs_number, 0.0);

    for(size_t j = 0; j < outputs_number; j++)
    {
        if(output[j] == 0.0)
        {
            output_gradient[j] = (-target[j]/1.e-6 + (1.0 - target[j])/(1.0 - 1.e-6));
        }
        else if(output[j] == 1.0)
        {
            output_gradient[j] = (-target[j]/0.999999 + (1.0 - target[j])/(1.0 - 0.999999));
        }
        else
        {
           output_gradient[j] = (-target[j]/output[j] + (1.0 - target[j])/(1.0 - output[j]));
        }
    }

    return(output_gradient/static_cast<double>(training_instances_number));
}


/// Returns the cross-entropy error function otuput Hessian of a multilayer perceptron on a data set.
/// It uses the error back-propagation method.

Matrix<double> CrossEntropyError::calculate_output_Hessian(const Vector<double>& output, const Vector<double>& target) const
{
    const size_t outputs_number = output.size();

    const size_t training_instances_number = data_set_pointer->get_instances().get_training_instances_number();

    Matrix<double> output_Hessian(outputs_number, outputs_number, 0.0);

    for(size_t i = 0; i < outputs_number; i++)
    {
        if(output[i] == 0.0)
        {
            output_Hessian(i,i) = (1.0 - target[i])/((1.0 - 1.e-6)*(1.0 - 1.e-6)) + target[i]/((1.e-6)*(1.e-6))/static_cast<double>(training_instances_number);
        }
        else if(output[i] == 1.0)
        {
            output_Hessian(i,i) = (1.0 - target[i])/((1.0 - 0.999999)*(1.0 - 0.999999))+target[i]/(0.999999*0.999999)/static_cast<double>(training_instances_number);
        }
        else
        {
           output_Hessian(i,i) = (1.0 - target[i])/((1.0 - output[i])*(1.0 - output[i])) + target[i]/(output[i]*output[i])/static_cast<double>(training_instances_number);
        }
    }

    return output_Hessian;
}


/// Returns the cross entropy error function gradient of a multilayer perceptron on a data set.
/// It uses the error back-propagation method.

Vector<double> CrossEntropyError::calculate_gradient() const
{
   // Control sentence(if debug)

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

   const Instances& instances = data_set_pointer->get_instances();

   const Vector<size_t> training_indices = instances.get_training_indices();

   const size_t training_instances_number = training_indices.size();

   const Variables& variables = data_set_pointer->get_variables();

   const Vector<size_t> inputs_indices = variables.get_inputs_indices();
   const Vector<size_t> targets_indices = variables.get_targets_indices();

   // Sum squared error stuff

   Vector<double> output_gradient(outputs_number);

   Vector< Matrix<double> > layers_combination_parameters_Jacobian;

   Vector< Vector<double> > layers_inputs(layers_number);
   Vector< Vector<double> > layers_delta;

   Vector<double> point_gradient(neural_parameters_number, 0.0);

   Vector<double> gradient(neural_parameters_number, 0.0);

   #pragma omp parallel for private(first_order_forward_propagation, layers_inputs, layers_combination_parameters_Jacobian,\
    output_gradient, layers_delta, point_gradient)

   for(int i = 0; i < static_cast<int>(training_instances_number); i++)
   {
       const size_t training_index = training_indices[static_cast<size_t>(i)];

       // Data set

       const Vector<double> inputs = data_set_pointer->get_instance(training_index, inputs_indices);

       const Vector<double> targets = data_set_pointer->get_instance(training_index, targets_indices);

       // Multilayer perceptron

       first_order_forward_propagation = multilayer_perceptron_pointer->calculate_first_order_forward_propagation(inputs);

       const Vector< Vector<double> >& layers_activation = first_order_forward_propagation[0];
       const Vector< Vector<double> >& layers_activation_derivative = first_order_forward_propagation[1];

       layers_inputs = multilayer_perceptron_pointer->get_layers_input(inputs, layers_activation);

       layers_combination_parameters_Jacobian = multilayer_perceptron_pointer->calculate_layers_combination_parameters_Jacobian(layers_inputs);

       // Loss index

           output_gradient = calculate_output_gradient_unnormalized(layers_activation[layers_number-1], targets)/static_cast<double>(training_instances_number);

           layers_delta = calculate_layers_delta(layers_activation_derivative, output_gradient);

       point_gradient = calculate_point_gradient(layers_combination_parameters_Jacobian, layers_delta);

       #pragma omp critical
       gradient += point_gradient;
   }

   return(gradient);
}


/// Returns the cross-entropy error function output gradient without the normalization of a multilayer perceptron on a data set.
/// It uses the error back-propagation method.
/// @param output Vector of outputs of the neural network.
/// @param target Vector of targets of the data set.

Vector<double> CrossEntropyError::calculate_output_gradient_unnormalized(const Vector<double>& output, const Vector<double>& target) const
{
    const size_t outputs_number = output.size();

    Vector<double> output_gradient(outputs_number, 0.0);

    for(size_t j = 0; j < outputs_number; j++)
    {
        if(output[j] == 0.0)
        {
            output_gradient[j] = (-target[j]/1.e-6 + (1.0 - target[j])/(1.0 - 1.e-6));
        }
        else if(output[j] == 1.0)
        {
            output_gradient[j] = (-target[j]/0.999999 + (1.0 - target[j])/(1.0 - 0.999999));
        }
        else
        {
           output_gradient[j] = (-target[j]/output[j] + (1.0 - target[j])/(1.0 - output[j]));
        }
    }

    return(output_gradient);
}


/// Returns the cross-entropy error function output gradient without the normalization of a multilayer perceptron on a data set.
/// It uses the error back-propagation method.

Vector<double> CrossEntropyError::calculate_gradient_unnormalized() const
{
#ifdef __OPENNN_DEBUG__

    check();

#endif

    // Neural network stuff

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

    const size_t neural_parameters_number = multilayer_perceptron_pointer->get_parameters_number();

    Vector< Vector< Vector<double> > > first_order_forward_propagation(2);

    // Data set stuff

    const Instances& instances = data_set_pointer->get_instances();

    const Vector<size_t> training_indices = instances.get_training_indices();

    const size_t training_instances_number = training_indices.size();

    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> inputs_indices = variables.get_inputs_indices();
    const Vector<size_t> targets_indices = variables.get_targets_indices();

    Vector<double> inputs(inputs_number);
    Vector<double> targets(outputs_number);

    // Sum squared error stuff

    Vector<double> output_gradient(outputs_number);

    Vector< Matrix<double> > layers_combination_parameters_Jacobian;

    Vector< Vector<double> > layers_inputs(layers_number);
    Vector< Vector<double> > layers_delta;

    Vector<double> point_gradient(neural_parameters_number, 0.0);

    Vector<double> gradient(neural_parameters_number, 0.0);

#pragma omp parallel for private(first_order_forward_propagation, layers_inputs, layers_combination_parameters_Jacobian,\
    output_gradient, layers_delta, point_gradient)

    for(int i = 0; i < static_cast<int>(training_instances_number); i++)
    {
        const size_t training_index = training_indices[static_cast<size_t>(i)];

        const Vector<double> inputs = data_set_pointer->get_instance(training_index, inputs_indices);

        const Vector<double> targets = data_set_pointer->get_instance(training_index, targets_indices);

        first_order_forward_propagation = multilayer_perceptron_pointer->calculate_first_order_forward_propagation(inputs);

        const Vector< Vector<double> >& layers_activation = first_order_forward_propagation[0];
        const Vector< Vector<double> >& layers_activation_derivative = first_order_forward_propagation[1];

        layers_inputs = multilayer_perceptron_pointer->get_layers_input(inputs, layers_activation);

        layers_combination_parameters_Jacobian = multilayer_perceptron_pointer->calculate_layers_combination_parameters_Jacobian(layers_inputs);

            output_gradient = calculate_output_gradient_unnormalized(layers_activation[layers_number-1], targets);

            layers_delta = calculate_layers_delta(layers_activation_derivative, output_gradient);

        point_gradient = calculate_point_gradient(layers_combination_parameters_Jacobian, layers_delta);

#pragma omp critical
        gradient += point_gradient;
    }

    return(gradient);
}


/// Returns a string with the name of the cross entropy error loss type, "CROSS_ENTROPY_ERROR".

string CrossEntropyError::write_error_term_type() const
{
   return("CROSS_ENTROPY_ERROR");
}


/// Serializes the cross entropy error object into a XML document of the TinyXML library. 
/// See the OpenNN manual for more information about the format of this document-> 

tinyxml2::XMLDocument* CrossEntropyError::to_XML() const
{
   ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Cross entropy error 

   tinyxml2::XMLElement* cross_entropy_error_element = document->NewElement("CrossEntropyError");

   document->InsertFirstChild(cross_entropy_error_element);

   // Display

//   {
//      tinyxml2::XMLElement* display_element = document->NewElement("Display");
//      cross_entropy_error_element->LinkEndChild(display_element);

//      buffer.str("");
//      buffer << display;

//      tinyxml2::XMLText* display_text = document->NewText(buffer.str().c_str());
//      display_element->LinkEndChild(display_text);
//   }

   return(document);
}


/// Serializes the cross entropy error object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document

void CrossEntropyError::write_XML(tinyxml2::XMLPrinter&) const
{
    //file_stream.OpenElement("CrossEntropyError");

    //file_stream.CloseElement();
}


/// Deserializes a TinyXML document into this cross entropy object.
/// @param document TinyXML document containing the member data.

void CrossEntropyError::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("CrossEntropyError");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: CrossEntropyError class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Cross entropy error element is nullptr.\n";

        throw logic_error(buffer.str());
    }

  // Display
  {
     const tinyxml2::XMLElement* display_element = root_element->FirstChildElement("Display");

     if(display_element)
     {
        const string new_display_string = display_element->GetText();

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
