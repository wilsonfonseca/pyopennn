/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   E R R O R   T E R M   C L A S S                                                                            */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "loss_index.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates a default error term object, with all pointers initialized to nullptr.
/// It also initializes all the rest of class members to their default values.

LossIndex::LossIndex()
 : neural_network_pointer(nullptr), 
   data_set_pointer(nullptr)
{
   set_default();
}


// NEURAL NETWORK CONSTRUCTOR

/// Neural network constructor. 
/// It creates a error term object associated to a neural network object.
/// The rest of pointers are initialized to nullptr.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

LossIndex::LossIndex(NeuralNetwork* new_neural_network_pointer)
 : neural_network_pointer(new_neural_network_pointer), 
   data_set_pointer(nullptr)
{
   set_default();
}


// DATA SET CONSTRUCTOR

/// Data set constructor. 
/// It creates a error term object associated to a given data set object.
/// The rest of pointers are initialized to nullptr.
/// It also initializes all the rest of class members to their default values.
/// @param new_data_set_pointer Pointer to a data set object.

LossIndex::LossIndex(DataSet* new_data_set_pointer)
 : neural_network_pointer(nullptr), 
   data_set_pointer(new_data_set_pointer)
{
   set_default();
}


// NEURAL NETWORK AND DATA SET CONSTRUCTOR

/// Neural network and data set constructor. 
/// It creates a error term object associated to a neural network and to be measured on a data set.
/// The rest of pointers are initialized to nullptr.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

LossIndex::LossIndex(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
 : neural_network_pointer(new_neural_network_pointer), 
   data_set_pointer(new_data_set_pointer)
{
   set_default();
}


// XML CONSTRUCTOR

/// XML constructor. 
/// It creates a default error term object, with all pointers initialized to nullptr.
/// It also loads all the rest of class members from a XML document.
/// @param error_term_document Pointer to a TinyXML document with the object data.

LossIndex::LossIndex(const tinyxml2::XMLDocument& error_term_document)
 : neural_network_pointer(nullptr), 
   data_set_pointer(nullptr)
{
   set_default();

   from_XML(error_term_document);
}


// COPY CONSTRUCTOR

/// Copy constructor. 
/// It creates a copy of an existing error term object.
/// @param other_error_term Error term object to be copied.

LossIndex::LossIndex(const LossIndex& other_error_term)
 : neural_network_pointer(nullptr), 
   data_set_pointer(nullptr)
{
   neural_network_pointer = other_error_term.neural_network_pointer;

   data_set_pointer = other_error_term.data_set_pointer;

   display = other_error_term.display;
}


// DESTRUCTOR

/// Destructor.

LossIndex::~LossIndex()
{
}


// ASSIGNMENT OPERATOR


/// Assignment operator. 
/// It assigns to this error term object the members from another error term object.
/// @param other_error_term Error term object to be copied.

LossIndex& LossIndex::operator = (const LossIndex& other_error_term)
{
   if(this != &other_error_term)
   {
      neural_network_pointer = other_error_term.neural_network_pointer;

      data_set_pointer = other_error_term.data_set_pointer;

      display = other_error_term.display;
   }

   return(*this);
}


// EQUAL TO OPERATOR


/// Equal to operator. 
/// It compares this object to another object. 
/// The return is true if both objects have the same member data, and false otherwise. 

bool LossIndex::operator == (const LossIndex& other_error_term) const
{
   if(neural_network_pointer != other_error_term.neural_network_pointer
   || data_set_pointer != other_error_term.data_set_pointer)
   {
       return(false);
   }

   else if(display != other_error_term.display)
   {
      return(false);
   }

   return(true);

}


// METHODS

const double& LossIndex::get_regularization_weight() const
{
   return(regularization_weight);
}


/// Returns true if messages from this class can be displayed on the screen, or false if messages
/// from this class can't be displayed on the screen.

const bool& LossIndex::get_display() const
{
   return(display);
}


/// Returns true if this error term has a neural network associated,
/// and false otherwise.

bool LossIndex::has_neural_network() const
{
    if(neural_network_pointer)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


/// Returns true if this error term has a data set associated,
/// and false otherwise.

bool LossIndex::has_data_set() const
{
    if(data_set_pointer)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


/// Sets all the member pointers to nullptr(neural network, data set, mathematical model).
/// It also initializes all the rest of class members to their default values.

void LossIndex::set()
{
   neural_network_pointer = nullptr;
   data_set_pointer = nullptr;

   set_default();
}


/// Sets all the member pointers to nullptr, but the neural network, which set to a given pointer.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

void LossIndex::set(NeuralNetwork* new_neural_network_pointer)
{
   neural_network_pointer = new_neural_network_pointer;
   data_set_pointer = nullptr;

   set_default();
}


/// Sets all the member pointers to nullptr, but the data set, which set to a given pointer.
/// It also initializes all the rest of class members to their default values.
/// @param new_data_set_pointer Pointer to a data set object.

void LossIndex::set(DataSet* new_data_set_pointer)
{
   neural_network_pointer = nullptr;
   data_set_pointer = new_data_set_pointer;

   set_default();
}


/// Sets new neural network and data set pointers.
/// Finally, it initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

void LossIndex::set(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
{
   neural_network_pointer = new_neural_network_pointer;

   data_set_pointer = new_data_set_pointer;

   set_default();
}


/// Sets to this error term object the members of another error term object.
/// @param other_error_term Error term to be copied.

void LossIndex::set(const LossIndex& other_error_term)
{
   neural_network_pointer = other_error_term.neural_network_pointer;

   data_set_pointer = other_error_term.data_set_pointer;

   regularization_method = other_error_term.regularization_method;

   display = other_error_term.display;
}


/// Sets a pointer to a neural network object which is to be associated to the error term.
/// @param new_neural_network_pointer Pointer to a neural network object to be associated to the error term.

void LossIndex::set_neural_network_pointer(NeuralNetwork* new_neural_network_pointer)
{
   neural_network_pointer = new_neural_network_pointer;
}


/// Sets a new data set on which the error term is to be measured.

void LossIndex::set_data_set_pointer(DataSet* new_data_set_pointer)
{
   data_set_pointer = new_data_set_pointer;
}


/// Sets the members of the error term to their default values:
/// <ul>
/// <li> Display: true.
/// </ul>

void LossIndex::set_default()
{
   regularization_method = L2;
   display = true;
}


void LossIndex::set_regularization_method(const LossIndex::RegularizationMethod& new_regularization_method)
{
    regularization_method = new_regularization_method;
}

void LossIndex::set_regularization_weight(const double& new_regularization_weight)
{
    regularization_weight = new_regularization_weight;
}


/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void LossIndex::set_display(const bool& new_display)
{
   display = new_display;
}


/// Checks that there is a neural network associated to the error term.
/// If some of the above conditions is not hold, the method throws an exception. 

void LossIndex::check() const
{
   ostringstream buffer;

   if(!neural_network_pointer)
   {
      buffer << "OpenNN Exception: LossIndex class.\n"
             << "void check() const.\n"
             << "Pointer to neural network is nullptr.\n";

      throw logic_error(buffer.str());	  
   }

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

  if(!multilayer_perceptron_pointer)
  {
        ostringstream buffer;

        buffer << "OpenNN Exception: LossIndex class.\n"
            << "Vector< Vector<double> > calculate_layers_delta(const Vector< Vector<double> >&, const Vector<double>&) const method.\n"
            << "Pointer to multilayer perceptron in neural network is nullptr.\n";

        throw logic_error(buffer.str());
  }
}



double LossIndex::calculate_error(const Vector<size_t>& instances_indices) const
{
   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   // Data set stuff

   const Matrix<double> inputs = data_set_pointer->get_input_data(instances_indices);

   const Matrix<double> targets = data_set_pointer->get_target_data(instances_indices);

   // Neural network stuff

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

   return calculate_error(outputs, targets);
}


double LossIndex::calculate_error(const Vector<size_t>& instances_indices, const Vector<double>& parameters) const
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


double LossIndex::calculate_training_error() const
{
    const Vector< Vector<size_t> > training_batchs_indices
            = data_set_pointer->get_instances_pointer()->get_training_batchs_indices(batch_size);

    const size_t batchs_number = training_batchs_indices.size();

    double training_error = 0.0;

    #pragma omp parallel for reduction(+ : training_error)

    for(int i = 0; i < static_cast<int>(batchs_number); i++)
    {
        const double training_batch_error = calculate_error(training_batchs_indices[static_cast<unsigned>(i)]);

        training_error += training_batch_error;
    }

    return training_error;
}


Vector<double> LossIndex::calculate_training_error_gradient() const
{
    cout << "Start gradient" << endl;

    const size_t parameters_number = neural_network_pointer->get_multilayer_perceptron_pointer()->get_parameters_number();

    const Vector< Vector<size_t> > training_batchs_indices
            = data_set_pointer->get_instances_pointer()->get_training_batchs_indices(batch_size);

    const size_t batchs_number = training_batchs_indices.size();

    Vector<double> training_error_gradient(parameters_number, 0.0);

    #pragma omp parallel for

    for(int i = 0; i < static_cast<int>(batchs_number); i++)
    {
        const Vector<double> training_batch_gradient = calculate_error_gradient(training_batchs_indices[static_cast<unsigned>(i)]);

        #pragma omp critical

        training_error_gradient += training_batch_gradient;
    }

    cout << "End gradient" << endl;

    return training_error_gradient;
}



double LossIndex::calculate_selection_error() const
{
    const Vector< Vector<size_t> > selection_batchs_indices
            = data_set_pointer->get_instances_pointer()->get_selection_batchs_indices(batch_size);

    const size_t batchs_number = selection_batchs_indices.size();

    double selection_error = 0.0;

    #pragma omp parallel for reduction(+ : selection_error)

    for(int i = 0; i < static_cast<int>(batchs_number); i++)
    {
        const double selection_batch_error = calculate_error(selection_batchs_indices[static_cast<unsigned>(i)]);

        selection_error += selection_batch_error;
    }

    return selection_error;
}


double LossIndex::calculate_testing_error() const
{
    return 0.0;
}


double LossIndex::calculate_all_instances_error() const
{
    const size_t instances_number = data_set_pointer->get_instances_pointer()->get_instances_number();

    const Vector<size_t> instances_indices(0, 1, instances_number-1);

    return calculate_error(instances_indices);
}


double LossIndex::calculate_all_instances_error(const Vector<double>& parameters) const
{
    const size_t instances_number = data_set_pointer->get_instances_pointer()->get_instances_number();

    const Vector<size_t> instances_indices(0, 1, instances_number-1);

    return calculate_error(instances_indices, parameters);
}



double LossIndex::calculate_training_error(const Vector<double>& parameters) const
{
    const Vector< Vector<size_t> > training_batchs_indices
            = data_set_pointer->get_instances_pointer()->get_training_batchs_indices(batch_size);

    const size_t batchs_number = training_batchs_indices.size();

    double training_error = 0.0;

//    #pragma omp parallel for reduction(+ : training_error)

    for(int i = 0; i < static_cast<int>(batchs_number); i++)
    {
        const double training_batch_error = calculate_error(training_batchs_indices[static_cast<unsigned>(i)], parameters);

        training_error += training_batch_error;
    }

    return training_error;
}


double LossIndex::calculate_selection_error(const Vector<double>& parameters) const
{
    const Vector< Vector<size_t> > selection_batchs_indices
            = data_set_pointer->get_instances_pointer()->get_selection_batchs_indices(batch_size);

    const size_t batchs_number = selection_batchs_indices.size();

    double selection_error = 0.0;

    #pragma omp parallel for reduction(+ : selection_error)

    for(int i = 0; i < static_cast<int>(batchs_number); i++)
    {
        const double selection_batch_error = calculate_error(selection_batchs_indices[static_cast<unsigned>(i)], parameters);

        selection_error += selection_batch_error;
    }

    return selection_error;
}


double LossIndex::calculate_testing_error(const Vector<double>& parameters) const
{
    const Vector< Vector<size_t> > testing_batchs_indices
            = data_set_pointer->get_instances_pointer()->get_testing_batchs_indices(batch_size);

    const size_t batchs_number = testing_batchs_indices.size();

    double testing_error = 0.0;

    #pragma omp parallel for reduction(+ : testing_error)

    for(int i = 0; i < static_cast<int>(batchs_number); i++)
    {
        const double testing_batch_error = calculate_error(testing_batchs_indices[static_cast<unsigned>(i)], parameters);

        testing_error += testing_batch_error;
    }

    return testing_error;
}

Vector<double> LossIndex::calculate_error_outputs(const Matrix<double>&) const
{
    return Vector<double>();
}


Vector<double> LossIndex::calculate_error_gradient() const
{
    const size_t instances_number = data_set_pointer->get_instances_pointer()->get_instances_number();

    const Vector<size_t> instances_indices(0, 1, instances_number-1);

    return calculate_error_gradient(instances_indices);
}


/// Returns the delta vector for all the layers in the multilayer perceptron.
/// The format of this quantity is a vector of vectors. 
/// @param layers_activation_derivative Forward propagation activation derivative. 
/// @param output_gradient Gradient of the outputs error function.

Vector< Vector<double> > LossIndex::calculate_layers_delta
(const Vector< Vector<double> >& layers_activation_derivative, 
 const Vector<double>& output_gradient) const
{
   // Neural network stuff

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   #ifdef __OPENNN_DEBUG__
   
   check();

   #endif

   const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

   const Vector<size_t> layers_perceptrons_number = multilayer_perceptron_pointer->get_layers_perceptrons_numbers();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   // Forward propagation activation derivative size

   const size_t layers_activation_derivative_size = layers_activation_derivative.size();

   if(layers_activation_derivative_size != layers_number)
   {
       ostringstream buffer;

      buffer << "OpenNN Exception: LossIndex class.\n"
             << "Vector< Vector<double> > calculate_layers_delta(const Vector< Vector<double> >&, const Vector<double>&) method.\n"
             << "Size of forward propagation activation derivative vector must be equal to number of layers.\n";

      throw logic_error(buffer.str());	  
   }

   if(layers_number > 0)
   {
      const size_t output_gradient_size = output_gradient.size();

      if(output_gradient_size != layers_perceptrons_number[layers_number-1])
      {
          ostringstream buffer;

         buffer << "OpenNN Exception: LossIndex class.\n"
                << "Vector<double> calculate_layers_delta(const Vector< Vector<double> >&, const Vector<double>&) method.\n"
                << "Size of outputs error gradient (" << output_gradient_size << ") must be equal to "
                << "number of outputs (" << layers_perceptrons_number[layers_number-1] << ").\n";

         throw logic_error(buffer.str());	     
      }
   }

   #endif

   // Neural network stuff

   Matrix<double> layer_synaptic_weights;

   // Loss index stuff

   Vector< Vector<double> > layers_delta(layers_number);

   // Output layer

   if(layers_number > 0)
   {
      layers_delta[layers_number-1] = layers_activation_derivative[layers_number-1]*output_gradient;

      // Rest of hidden layers

      for(int i = static_cast<int>(layers_number)-2; i >= 0; i--)
      {
         layer_synaptic_weights = neural_network_pointer->get_multilayer_perceptron_pointer()->get_layer(static_cast<size_t>(i+1)).get_synaptic_weights();

         layers_delta[static_cast<size_t>(i)] = layers_activation_derivative[static_cast<size_t>(i)]*(layers_delta[static_cast<size_t>(i+1)].dot(layer_synaptic_weights));
      }
   }

   return(layers_delta);
}


Vector< Matrix<double> > LossIndex::calculate_layers_delta
(const Vector< Matrix<double> >& layers_activation_derivative,
 const Matrix<double>& output_gradient) const
{
    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   // Neural network stuff

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const Vector<size_t> layers_perceptrons_number = multilayer_perceptron_pointer->get_layers_perceptrons_numbers();

   // Forward propagation activation derivative size

   const size_t layers_activation_derivative_size = layers_activation_derivative.size();

   if(layers_activation_derivative_size != layers_number)
   {
       ostringstream buffer;

      buffer << "OpenNN Exception: LossIndex class.\n"
             << "Vector< Matrix<double> > calculate_layers_delta(const Vector< Matrix<double> >&, const Matrix<double>&) method.\n"
             << "Size of forward propagation activation derivative vector must be equal to number of layers.\n";

      throw logic_error(buffer.str());
   }

   if(layers_number > 0)
   {
      const size_t output_gradient_columns_number = output_gradient.get_columns_number();

      if(output_gradient_columns_number != layers_perceptrons_number[layers_number-1])
      {
          ostringstream buffer;

         buffer << "OpenNN Exception: LossIndex class.\n"
                << "Vector<double> calculate_layers_delta(const Vector< Vector<double> >&, const Vector<double>&) method.\n"
                << "Size of outputs error gradient (" << output_gradient_columns_number << ") must be equal to "
                << "number of outputs (" << layers_perceptrons_number[layers_number-1] << ").\n";

         throw logic_error(buffer.str());
      }
   }

   #endif

   // Neural network stuff

   Matrix<double> layer_synaptic_weights_transpose;

   // Loss index stuff

   Vector< Matrix<double> > layers_delta(layers_number);

   // Output layer

   if(layers_number > 0)
   {
      layers_delta[layers_number-1] = layers_activation_derivative[layers_number-1]*output_gradient;

      // Rest of hidden layers

      for(int i = static_cast<int>(layers_number)-2; i >= 0; i--)
      {
         layer_synaptic_weights_transpose = neural_network_pointer->get_multilayer_perceptron_pointer()->get_layer(static_cast<size_t>(i+1)).get_synaptic_weights().calculate_transpose();

         layers_delta[static_cast<size_t>(i)] = layers_activation_derivative[static_cast<size_t>(i)]*(layers_delta[static_cast<size_t>(i+1)].dot(layer_synaptic_weights_transpose));
      }
   }

   return layers_delta;
}


/// Returns the gradient of the error term function at some input point.
/// @param inputs Input vector. 
/// @param layers_activation Activations of all layers in the multilayer perceptron 
/// @param layers_delta Vector of vectors containing the partial derivatives of the outputs error function with respect to all the combinations of all layers. 

Vector<double> LossIndex::calculate_point_gradient
(const Vector<double>& inputs, 
 const Vector< Vector<double> >& layers_activation, 
 const Vector< Vector<double> >& layers_delta) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
   const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();
   const Vector<size_t> layers_perceptrons_number = multilayer_perceptron_pointer->get_layers_perceptrons_numbers();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__
 
   // Input size

   const size_t inputs_size = inputs.size();

   if(inputs_size != inputs_number)
   {
       ostringstream buffer;

      buffer << "OpenNN Exception: LossIndex class.\n"
             << "Vector< Vector<double> > calculate_layers_error_gradient(const Vector< Vector<double> >&, const Vector<double>&, const Vector<double>&) method.\n"
             << "Size of inputs(" << inputs_size << ") must be equal to inputs number(" << inputs_number << ").\n";

      throw logic_error(buffer.str());  
   }

   // Forward propagation activation size

   const size_t layers_activation_size = layers_activation.size();

   if(layers_activation_size != layers_number)
   {
       ostringstream buffer;

      buffer << "OpenNN Exception: LossIndex class.\n"
             << "Vector< Vector<double> > calculate_layers_error_gradient(const Vector< Vector<double> >&, const Vector<double>&, const Vector<double>&) method.\n"
             << "Size of forward propagation activation(" << layers_activation_size << ") must be equal to number of layers(" << layers_number << ").\n";

      throw logic_error(buffer.str());	  
   }

   // Hidden errors size

   const size_t layers_delta_size = layers_delta.size();
      
   if(layers_delta_size != layers_number)
   {
       ostringstream buffer;

      buffer << "OpenNN Exception: LossIndex class.\n"
             << "Vector< Vector<double> > calculate_layers_error_gradient(const Vector< Vector<double> >&, const Vector<double>&) method.\n"
             << "Size of layers delta("<< layers_delta_size << ") must be equal to number of layers(" << layers_number << ").\n";

      throw logic_error(buffer.str());	  
   }

   #endif

   const size_t parameters_number = neural_network_pointer->get_parameters_number();

   Vector<double> point_gradient(parameters_number);

   size_t index = 0;

   const Vector< Vector<double> > layers_inputs = multilayer_perceptron_pointer->get_layers_input(inputs, layers_activation);

   const Vector< Matrix<double> > layers_combination_parameters_Jacobian = multilayer_perceptron_pointer->calculate_layers_combination_parameters_Jacobian(layers_inputs);

   for(size_t i = 0; i < layers_number; i++)
   {                  
      point_gradient.tuck_in(index, layers_delta[i].dot(layers_combination_parameters_Jacobian[i]));

      index += multilayer_perceptron_pointer->get_layer(i).get_parameters_number();
   }

   if(layers_number != 0)
   {
        return Vector<double>(parameters_number, 0.0);
   }

      Vector<double> synaptic_weights;

      index = 0;

      // First layer

  for(size_t i = 0; i < layers_perceptrons_number[0]; i++)
  {
     // Bias

     point_gradient[index] = layers_delta[0][i];
     index++;

     // Synaptic weights

     for(size_t j = 0; j < inputs_number; j++)
     {
        point_gradient[index] = layers_delta[0][i]*inputs[j];
        index++;
     }
  }

      // Rest of layers	
    
      for(size_t h = 1; h < layers_number; h++)
      {      
         for(size_t i = 0; i < layers_perceptrons_number[h]; i++)
         {
            // Bias

            point_gradient[index] = layers_delta[h][i];
            index++;

            // Synaptic weights

            for(size_t j = 0; j < layers_perceptrons_number[h-1]; j++)
            {
               point_gradient[index] = layers_delta[h][i]*layers_activation[h-1][j];
               index++;   
            }
         }
      }

   return(point_gradient);
}


/// Returns the gradient of the error term function at some input point.
/// @param layers_combination_parameters_Jacobian
/// @param layers_delta
/// @todo

Vector<double> LossIndex::calculate_point_gradient
(const Vector< Matrix<double> >& layers_combination_parameters_Jacobian, 
 const Vector< Vector<double> >& layers_delta) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__
 
   // Input size

   const size_t layers_combination_parameters_Jacobian_size = layers_combination_parameters_Jacobian.size();

   if(layers_combination_parameters_Jacobian_size != layers_number)
   {
       ostringstream buffer;

      buffer << "OpenNN Exception: LossIndex class.\n"
             << "Vector< Vector<double> > calculate_layers_error_gradient(const Vector< Vector<double> >&, const Vector<double>&, const Vector<double>&) method.\n"
             << "Size of forward propagation activation(" << layers_combination_parameters_Jacobian_size << ") must be equal to number of layers(" << layers_number << ").\n";

      throw logic_error(buffer.str());	  
   }

   // Hidden errors size

   const size_t layers_delta_size = layers_delta.size();
      
   if(layers_delta_size != layers_number)
   {
       ostringstream buffer;

      buffer << "OpenNN Exception: LossIndex class.\n"
             << "Vector< Vector<double> > calculate_layers_error_gradient(const Vector< Vector<double> >&, const Vector<double>&) method.\n"
             << "Size of layers delta("<< layers_delta_size << ") must be equal to number of layers(" << layers_number << ").\n";

      throw logic_error(buffer.str());	  
   }

   #endif

   const Vector<size_t> layers_parameters_number = multilayer_perceptron_pointer->get_layers_parameters_number();

   const size_t parameters_number = multilayer_perceptron_pointer->get_parameters_number();

   Vector<double> point_gradient(parameters_number);

   size_t index = 0;

   for(size_t i = 0; i < layers_number; i++)
   {
      const Vector<double> layer_point_gradient = layers_delta[i].dot(layers_combination_parameters_Jacobian[i]);

      point_gradient.tuck_in(index, layer_point_gradient);

      index += layers_parameters_number[i];
   }

   return(point_gradient);
}



Vector<double> LossIndex::calculate_error_gradient
(const Matrix<double>& inputs,
 const Vector< Matrix<double> >& layers_activations,
 const Vector< Matrix<double> >& layers_delta) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   // Input size

//   const size_t layers_combination_parameters_Jacobian_size = layers_combination_parameters_Jacobian.size();

//   if(layers_combination_parameters_Jacobian_size != layers_number)
//   {
//       ostringstream buffer;

//      buffer << "OpenNN Exception: LossIndex class.\n"
//             << "Vector< Vector<double> > calculate_layers_error_gradient(const Vector< Vector<double> >&, const Vector<double>&, const Vector<double>&) method.\n"
//             << "Size of forward propagation activation(" << layers_combination_parameters_Jacobian_size << ") must be equal to number of layers(" << layers_number << ").\n";

//      throw logic_error(buffer.str());
//   }

   // Hidden errors size

   const size_t layers_delta_size = layers_delta.size();

   if(layers_delta_size != layers_number)
   {
       ostringstream buffer;

      buffer << "OpenNN Exception: LossIndex class.\n"
             << "Vector< Vector<double> > calculate_layers_error_gradient(const Vector< Vector<double> >&, const Vector<double>&) method.\n"
             << "Size of layers delta("<< layers_delta_size << ") must be equal to number of layers(" << layers_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   const size_t parameters_number = multilayer_perceptron_pointer->get_parameters_number();

   const Vector<size_t> layers_parameters_number = multilayer_perceptron_pointer->get_layers_parameters_number();

   Vector<double> error_gradient(parameters_number);

   size_t index = 0;

   Vector< Matrix<double> > layer_combination_parameters_Jacobian;

   for(size_t i = 0; i < layers_number; i++)
   {
      if(i == 0)
      {
          layer_combination_parameters_Jacobian
                  = multilayer_perceptron_pointer->get_layer(i).calculate_combinations_parameters_Jacobian(inputs);
      }
      else
      {
          layer_combination_parameters_Jacobian
                  = multilayer_perceptron_pointer->get_layer(i).calculate_combinations_parameters_Jacobian(layers_activations[i-1]);
      }

      const Vector<double> layer_error_gradient = layers_delta[i].dot(layer_combination_parameters_Jacobian).calculate_columns_sum();

      error_gradient.tuck_in(index, layer_error_gradient);

      index += layers_parameters_number[i];
   }

   return error_gradient;
}

/// @todo

double LossIndex::calculate_point_error_output_layer_combinations(const Vector<double>& output_layer_combinations) const
{
    const size_t outputs_number = neural_network_pointer->get_multilayer_perceptron_pointer()->get_outputs_number();

    Vector<double> targets(outputs_number, 1.0);

//    const Instances& instances = data_set_pointer->get_instances();

//    const Variables& variables = data_set_pointer->get_variables();

//    const Vector<size_t> targets_indices = variables.get_targets_indices();

//    targets = data_set_pointer->get_instance(0, targets_indices);

    Vector<double> activations = neural_network_pointer->get_multilayer_perceptron_pointer()->get_layer(0).calculate_activations(output_layer_combinations);

    return activations.calculate_sum_squared_error(targets);
}


double LossIndex::calculate_point_error_layer_combinations(const size_t& layer_index, const size_t& instance_index, const Vector<double>& layer_combination) const
{
    MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

//    const size_t instances_number = data_set_pointer->get_instances_pointer()->get_instances_number();
    const Vector<size_t> instances_indices(1, instance_index);

    const Vector<double> outputs = multilayer_perceptron_pointer->calculate_outputs_layer_combinations(layer_index, layer_combination);
    const Vector<double> targets = data_set_pointer->get_target_data(instances_indices);

   return outputs.calculate_sum_squared_error(targets);
}

Vector<double> LossIndex::calculate_points_errors_output_layer_combinations(const Matrix<double>& output_layer_combinations) const
{
    const size_t outputs_number = neural_network_pointer->get_multilayer_perceptron_pointer()->get_outputs_number();

    Matrix<double> targets(outputs_number, 1);

    Matrix<double> activations = neural_network_pointer->get_multilayer_perceptron_pointer()->get_layer(0).calculate_activations(output_layer_combinations);

    return activations.calculate_sum_squared_error_rows(targets);
}


Vector<double> LossIndex::calculate_points_errors_layer_combinations(const size_t& layer_index, const Matrix<double>& layer_combinations) const
{
    MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t instances_number = data_set_pointer->get_instances_pointer()->get_instances_number();
    const Vector<size_t> instances_indices(0, 1, instances_number-1);

    const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs_layer_combinations(layer_index, layer_combinations);
    const Matrix<double> targets = data_set_pointer->get_target_data(instances_indices);

   return outputs.calculate_sum_squared_error_rows(targets);
}

/// @todo

Matrix<double> LossIndex::calculate_output_interlayers_Delta(const Vector<double>& output_layer_activation_derivative,
                                                                   const Vector<double>& output_layer_activation_second_derivative,
                                                                   const Vector<double>& output_gradient,
                                                                   const Matrix<double>& output_Hessian) const
{
//    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

//    const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

//    const Matrix<double> output_interlayers_Delta =
//   (output_Hessian
//     * output_layer_activation_derivative[layers_number-1]
//     * output_layer_activation_derivative[layers_number-1]
//     + output_gradient
//     * output_layer_activation_second_derivative[layers_number-1]);

//    return(output_interlayers_Delta);

    return Matrix<double>();
}

/// @todo

Matrix<double> LossIndex::calculate_interlayers_Delta(
        const size_t& index_1,
        const size_t& index_2,
        const Vector<double>& layer_1_activation_derivative,
        const Vector<double>& layer_2_activation_derivative,
        const Vector<double>& layer_1_activation_second_derivative,
        const Vector<double>& layer_2_activation_second_derivative,
        const Vector< Vector<double> >& layers_activation_derivative,
        const Vector<double>& layers_delta,
        const Matrix<double>& interlayers_combination_combination_Jacobian,
        const Matrix<double>& previous_interlayers_Delta,
        const Vector< Vector<double> >& complete_layers_delta) const
{
    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();
    const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

    Matrix<double> layer_1_weights  = multilayer_perceptron_pointer->get_layer(index_1).get_synaptic_weights();
    Matrix<double> layer_2_weights  = multilayer_perceptron_pointer->get_layer(index_2).get_synaptic_weights();
    Matrix<double> output_layer_weights = multilayer_perceptron_pointer->get_layer(layers_number-1).get_synaptic_weights();

    if(index_1 == 0 && index_2 == 0)
    {
        layer_2_weights = multilayer_perceptron_pointer->get_layer(index_2+1).get_synaptic_weights();
    }

    const size_t layer_1_perceptrons_number = multilayer_perceptron_pointer->get_layer(index_1).get_perceptrons_number();
    const size_t layer_2_perceptrons_number = multilayer_perceptron_pointer->get_layer(index_2).get_perceptrons_number();

    Matrix<double> interlayers_Delta(layer_1_perceptrons_number, layer_2_perceptrons_number, 0.0);

    for(size_t i = 0; i < layer_1_perceptrons_number; i++)
    {
        for(size_t j = 0; j < layer_2_perceptrons_number; j++)
        {
            if(index_2 == multilayer_perceptron_pointer->get_layers_number()-1)
            {
                if(index_1 == 0 && index_2 == 2)
                {
                    interlayers_Delta(i,j) +=
                            previous_interlayers_Delta(0,0)
                            *layers_activation_derivative[1][j]
                            *multilayer_perceptron_pointer->get_layer(index_2).get_synaptic_weights()(0,0)
                            *multilayer_perceptron_pointer->get_layer(index_1+1).get_synaptic_weights()(0,0)
                            *layer_1_activation_derivative[i]
                            +layer_1_activation_second_derivative[i]
                            *interlayers_combination_combination_Jacobian(0,0)
                            *multilayer_perceptron_pointer->get_layer(index_1).get_synaptic_weights().get_column(j).dot(layers_delta);
                }
                else
                {
                    interlayers_Delta(i,j) =
                            layer_2_weights(/*neuron_index_*/j, /*neuron_index_*/i)
                            *layer_1_activation_derivative[/*neuron_index_*/i]
                            *previous_interlayers_Delta(/*neuron_index_*/j, /*neuron_index_*/j);
                }
            }
            else
            {
                if(index_1 == 0 && index_2 == 0)
                {
                    interlayers_Delta(i,j) +=
                            layer_2_weights(0,0)
                            *layer_2_weights(0,0)
                            *previous_interlayers_Delta(0,0)
                            +layer_2_activation_second_derivative[/*neuron_index_j*/j]
                            *calculate_Kronecker_delta(i,j)
                            *output_layer_weights.get_column(j).dot(layers_delta);
                }
                else if(index_1 == 0 && index_2 == 1)
                {
                    cout << "------------" << endl;

                    cout << "Previous interlayers Delta: " << previous_interlayers_Delta << endl;

                    cout << "layers delta: " << layers_delta << endl;

                    cout << "complete layers delta: " << complete_layers_delta << endl;

                    cout << "interlayers_combination_combination_Jacobian(0,0): " << interlayers_combination_combination_Jacobian(0,0) << endl;

                    double interlayers_Delta02 =
                            previous_interlayers_Delta(0,0)
                            *layers_activation_derivative[1][j]
                            *multilayer_perceptron_pointer->get_layer(2).get_synaptic_weights()(0,0)
                            *multilayer_perceptron_pointer->get_layer(1).get_synaptic_weights()(0,0)
                            *layers_activation_derivative[0][j]
                            +layer_1_activation_second_derivative[i]
                            *interlayers_combination_combination_Jacobian(0,0)
                            *multilayer_perceptron_pointer->get_layer(0).get_synaptic_weights().get_column(j).dot(layers_delta);

                    cout << "Interlayers Delta(0,2): " << interlayers_Delta02 << endl;

                    // Previous interlayers Delta:   interlayers_Delta02

                    interlayers_Delta(i,j) +=
                            layers_activation_derivative[0][j]
                            *output_layer_weights(0,0)
                            *interlayers_Delta02
                            +
                            layer_1_activation_second_derivative[i]
                            *interlayers_combination_combination_Jacobian(0,0)
                            *multilayer_perceptron_pointer->get_layer(0).get_synaptic_weights().get_column(j).dot(complete_layers_delta[0]);

                    cout << "------------" << endl;
                }
                else
                {
                    if(index_1 == 1 && index_2 == 1)
                    {
                        for(size_t k = 0; k < outputs_number; k++)
                        {
                            interlayers_Delta(i,j) +=
                                    output_layer_weights(k, /*neuron_index_*/i)
                                    *output_layer_weights(k, /*neuron_index_*/j)
                                    *previous_interlayers_Delta(k,k);
                        }

                        interlayers_Delta(i,j) =
                                interlayers_Delta(i,j)
                                *layer_1_activation_derivative[/*neuron_index_*/i]
                                *layer_2_activation_derivative[/*neuron_index_*/j]
                                +layer_2_activation_second_derivative[/*neuron_index_j*/j]
                                *calculate_Kronecker_delta(i,j)
                                *output_layer_weights.get_column(j).dot(layers_delta);
                    }
                    else
                    {
                        for(size_t k = 0; k < outputs_number; k++)
                        {
                            interlayers_Delta(i,j) +=
                                    output_layer_weights(k, /*neuron_index_*/i)
                                    *output_layer_weights(k, /*neuron_index_*/j)
                                    *previous_interlayers_Delta(k,k);
                        }

                        interlayers_Delta(i,j) =
                                interlayers_Delta(i,j)
                                *layer_1_activation_derivative[/*neuron_index_*/i]
                                *layer_2_activation_derivative[/*neuron_index_*/j]
                                +layer_2_activation_second_derivative[/*neuron_index_j*/j]
                                *calculate_Kronecker_delta(i,j)
                                *output_layer_weights.get_column(j).dot(layers_delta);
                    }
                }
            }
        }
    }

//    cout << "-----------" << endl;

    return(interlayers_Delta);
}


/// Returns the second partial derivatives of the outputs error function with respect to the combinations of two layers.
/// That quantity is called interlayers Delta, and it is represented as a matrix of matrices. 
/// @param layers_activation_derivative Activation derivatives of all layers in the multilayer perceptron
/// @param layers_activation_second_derivative Activation second derivatives of all layers in the multilayer perceptron
/// @param interlayers_combination_combination_Jacobian_form Matrix of matrices containing the partial derivatives of all layers combinations with respect to all layers combinations. 
/// @param output_gradient Gradient vector of the outputs error function.
/// @param output_Hessian Hessian matrix of the outputs error function.
/// @param layers_delta Vector of vectors containing the partial derivatives of the outputs error function with respect to the combinations of all layers. 
/// @todo

Matrix< Matrix<double> > LossIndex::calculate_interlayers_Delta
(const Vector< Vector<double> >& layers_activation_derivative,
 const Vector< Vector<double> >& layers_activation_second_derivative,
 const Matrix< Matrix<double> >& interlayers_combination_combination_Jacobian_form, 
 const Vector<double>& output_gradient,
 const Matrix<double>& output_Hessian,
 const Vector< Vector<double> >& layers_delta) const
{
   // Neural network stuff

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();
   const Vector<size_t> layers_size = multilayer_perceptron_pointer->get_layers_perceptrons_numbers();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(layers_number != 0)
   {
      const size_t output_gradient_size = output_gradient.size();

      if(output_gradient_size != layers_size[layers_number-1])
      {
          ostringstream buffer;

         buffer << "OpenNN Exception: LossIndex class.\n"
                << "Vector<double> calculate_interlayers_Delta() method.\n"
                << "Size of layer " << layers_number-1 << " must be equal to size of output error gradient(" << output_gradient_size << ")."
                << endl;

         throw logic_error(buffer.str());
      }

      const size_t output_Hessian_rows_number = output_Hessian.get_rows_number();
      const size_t output_Hessian_columns_number = output_Hessian.get_columns_number();

      if(output_Hessian_rows_number != layers_size[layers_number-1])
      {
          ostringstream buffer;

         buffer << "OpenNN Exception: LossIndex class.\n"
                << "Vector<double> calculate_interlayers_Delta() method.\n"
                << "Size of layer " << layers_number-1 << " must be equal to number of rows in output error Hessian(" << output_Hessian_rows_number << ")."
                << endl;

         throw logic_error(buffer.str());
      }

      if(output_Hessian_columns_number != layers_size[layers_number-1])
      {
          ostringstream buffer;

         buffer << "OpenNN Exception: LossIndex class.\n"
                << "Vector<double> calculate_interlayers_Delta() method.\n"
                << "Size of layer " << layers_number-1 << ") must be equal to number of columns in output error Hessian(" << output_Hessian_columns_number << ")."
                << endl;

         throw logic_error(buffer.str());
      }
   }

   #endif

   const Vector< Matrix<double> > layers_synaptic_weights = multilayer_perceptron_pointer->get_layers_synaptic_weights();

   // Objective functional stuff

   Matrix< Matrix<double> > interlayers_Delta(layers_number, layers_number);

   Matrix<double> previous_interlayers_Delta;

   for(size_t i = 0; i < layers_number; i++)
   {
      for(size_t j = 0; j < layers_number; j++)
      {
         interlayers_Delta(i,j).set(layers_size[i], layers_size[j], 0.0);
      }
   }

   if(layers_number == 0)
   {
        return(interlayers_Delta);
   }

   Matrix<double> output_interlayers_Delta = calculate_output_interlayers_Delta(layers_activation_derivative[layers_number-1],
                                                                                layers_activation_second_derivative[layers_number-1],
                                                                                output_gradient,
                                                                                output_Hessian);

   interlayers_Delta(layers_number-1, layers_number-1) = output_interlayers_Delta;
//           calculate_output_interlayers_Delta(layers_activation_derivative[layers_number-1],
//                                              layers_activation_second_derivative[layers_number-1],
//                                              output_gradient,
//                                              output_Hessian);

   for(size_t i = layers_number-1; i == 0; i--)
   {
       for(size_t j = layers_number - 1; j == i; j--)
       {
           if(i == layers_number-1 &&  j == layers_number-1) // Output-outputs layer
           {
               interlayers_Delta(i,j) = calculate_output_interlayers_Delta(layers_activation_derivative[layers_number-1],
                                                                           layers_activation_second_derivative[layers_number-1],
                                                                           output_gradient,
                                                                           output_Hessian);
               previous_interlayers_Delta = interlayers_Delta(i,j);
           }
           else //Rest of hidden layers
           {
               cout << "layers_delta[i+1]: " << layers_delta[i+1] << endl;

               interlayers_Delta(i,j) = calculate_interlayers_Delta(i,
                                                                    j,
                                                                    layers_activation_derivative[i],
                                                                    layers_activation_derivative[j],
                                                                    layers_activation_second_derivative[i],
                                                                    layers_activation_second_derivative[j],
                                                                    layers_activation_derivative,
                                                                    layers_delta[i+1],
                                                                    interlayers_combination_combination_Jacobian_form(i,j),
                                                                    interlayers_Delta(2,2)/*previous_interlayers_Delta*/,
                                                                    layers_delta);
           }
       }
   }

   return interlayers_Delta;
}


/// Returns the Hessian of the error term at some input.
/// @param layers_activation_derivative
/// @param perceptrons_combination_parameters_gradient
/// @param interlayers_combination_combination_Jacobian
/// @param layers_delta
/// @param interlayers_Delta
/// @todo

Matrix<double> LossIndex::calculate_point_Hessian
(const Vector< Vector<double> >& layers_activation_derivative,
 const Vector< Vector< Vector<double> > >& perceptrons_combination_parameters_gradient,
 const Matrix< Matrix<double> >& interlayers_combination_combination_Jacobian,
 const Vector< Vector<double> >& layers_delta,
 const Matrix< Matrix<double> >& interlayers_Delta) const
{
   // Neural network stuff

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

   const size_t parameters_number = multilayer_perceptron_pointer->get_parameters_number();

//   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();
//   const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

   #ifdef __OPENNN_DEBUG__

   const size_t layers_activation_derivative_size = layers_activation_derivative.size();

   if(layers_activation_derivative_size != layers_number)
   {
       ostringstream buffer;

      buffer << "OpenNN Exception: LossIndex class.\n"
             << "Matrix<double> calculate_point_Hessian(const Vector<double>&, const Matrix< Matrix<double> >&, const Vector< Vector<double> >&, const Matrix< Matrix<double> >&) const method.\n"
             << "Size of layers activation derivative must be equal to number of layers in multilayer perceptron.\n";

      throw logic_error(buffer.str());
   }

//   const size_t perceptrons_combination_parameters_gradient_size = perceptrons_combination_parameters_gradient.size();

//   const size_terlayers_combination_combination_Jacobian_rows_number = interlayers_combination_combination_Jacobian.get_rows_number();
//   const size_terlayers_combination_combination_Jacobian_columns_number = interlayers_combination_combination_Jacobian.get_columns_number();

//   const size_t layers_delta_size = layers_delta.size();

//   const size_t interlayers_Delta_rows_number = interlayers_Delta.get_rows_number();
//   const size_t interlayers_Delta_columns_number = interlayers_Delta.get_columns_number();

   #endif

   // Objective functional

   Matrix<double> point_Hessian(parameters_number, parameters_number, 0.0);

   Vector<size_t> parameter_indices(3);

   size_t layer_index_i;
   size_t neuron_index_i;
   size_t parameter_index_i;

   size_t layer_index_j;
   size_t neuron_index_j;
   size_t parameter_index_j;

   const size_t last_layer_parameters_number = multilayer_perceptron_pointer->get_layer(layers_number-1).get_parameters_number();
   //const size_t first_layer_parameters_number = multilayer_perceptron_pointer->get_layer(0).get_parameters_number();

 //  Matrix<double> second_layer_weights = multilayer_perceptron_pointer->get_layer(1).get_synaptic_weights();

    // @todo

   cout << "Interlayers Delta: \n" << interlayers_Delta << endl;

   cout << "Layers delta: \n" << layers_delta << endl;
//   cout << "interlayers combination combination Jacobian: \n" << interlayers_combination_combination_Jacobian << endl;

   if(layers_number > 0)
   {
       // Last layer

       cout << "---Last layer---" << endl;

       for(size_t i = parameters_number-last_layer_parameters_number; i < parameters_number; i++)
       {
           parameter_indices = multilayer_perceptron_pointer->get_parameter_indices(i);
           layer_index_i = parameter_indices[0];
           neuron_index_i = parameter_indices[1];
           parameter_index_i = parameter_indices[2];

           for(size_t j = parameters_number-last_layer_parameters_number; j < parameters_number; j++)
           {
               parameter_indices = multilayer_perceptron_pointer->get_parameter_indices(j);
               layer_index_j = parameter_indices[0];
               neuron_index_j = parameter_indices[1];
               parameter_index_j = parameter_indices[2];

               point_Hessian(i,j) =
                       perceptrons_combination_parameters_gradient[layer_index_i][neuron_index_i][parameter_index_i]
                       *perceptrons_combination_parameters_gradient[layer_index_j][neuron_index_j][parameter_index_j]
                       *calculate_Kronecker_delta(neuron_index_i,neuron_index_j)
                       *interlayers_Delta(layer_index_i, layer_index_j)(neuron_index_j,neuron_index_i);
           }
       }

       // First layer

       cout << "---First layer---" << endl;

       for(size_t i = 0; i < parameters_number-last_layer_parameters_number; i++)
       {
           parameter_indices = multilayer_perceptron_pointer->get_parameter_indices(i);
           layer_index_i = parameter_indices[0];
           neuron_index_i = parameter_indices[1];
           parameter_index_i = parameter_indices[2];

           for(size_t j = 0; j < parameters_number-last_layer_parameters_number; j++)
           {
              parameter_indices = multilayer_perceptron_pointer->get_parameter_indices(j);
              layer_index_j = parameter_indices[0];
              neuron_index_j = parameter_indices[1];
              parameter_index_j = parameter_indices[2];


              point_Hessian(i,j) =
              (perceptrons_combination_parameters_gradient[layer_index_i][neuron_index_i][parameter_index_i]
               *perceptrons_combination_parameters_gradient[layer_index_j][neuron_index_j][parameter_index_j]
               //*layers_activation_derivative[layer_index_i][neuron_index_i]
               *interlayers_Delta(layer_index_i, layer_index_j)(neuron_index_i,neuron_index_j)
               + perceptrons_combination_parameters_gradient[layer_index_i][neuron_index_i][parameter_index_i]
               *layers_activation_derivative[layer_index_i][neuron_index_i]
               *layers_delta[layer_index_j][neuron_index_j]
               *calculate_Kronecker_delta(parameter_index_j,neuron_index_i+1));
              // *interlayers_combination_combination_Jacobian(layer_index_j, layer_index_i)(neuron_index_j,neuron_index_i));//(layer_index_i, layer_index_i+1)(neuron_index_j,neuron_index_i));

          }
       }

       // Rest of the layers

       cout << "---Rest of the layers---" << endl;

      for(size_t i = 0; i < parameters_number-last_layer_parameters_number; i++)
      {
          parameter_indices = multilayer_perceptron_pointer->get_parameter_indices(i);
          layer_index_i = parameter_indices[0];
          neuron_index_i = parameter_indices[1];
          parameter_index_i = parameter_indices[2];

          for(size_t j = parameters_number-last_layer_parameters_number; j < parameters_number; j++)
          {
              parameter_indices = multilayer_perceptron_pointer->get_parameter_indices(j);
              layer_index_j = parameter_indices[0];
              neuron_index_j = parameter_indices[1];
              parameter_index_j = parameter_indices[2];

              point_Hessian(i,j) =
              (perceptrons_combination_parameters_gradient[layer_index_i][neuron_index_i][parameter_index_i]
               *perceptrons_combination_parameters_gradient[layer_index_j][neuron_index_j][parameter_index_j]
               //*layers_activation_derivative[layer_index_i][neuron_index_i]
               *interlayers_Delta(layer_index_i, layer_index_j)(neuron_index_i,neuron_index_j)
               + perceptrons_combination_parameters_gradient[layer_index_i][neuron_index_i][parameter_index_i]
               *layers_activation_derivative[layer_index_i][neuron_index_i]
               *layers_delta[layer_index_j][neuron_index_j]
               *calculate_Kronecker_delta(parameter_index_j,neuron_index_i+1)
               *interlayers_combination_combination_Jacobian(layer_index_j-1, layer_index_i)(neuron_index_j,neuron_index_i));
          }
      }
   }

   for(size_t i = 0; i < parameters_number; i++)
   {
       for(size_t j = 0; j < i; j++)
       {
           point_Hessian(i,j) = point_Hessian(j,i);
       }
   }

   return(point_Hessian);
}


/// Returns the Hessian of the error term at some input for only one hidden layer.
/// @param layers_activation_derivative
/// @param layers_activation_second_derivative
/// @param perceptrons_combination_parameters_gradient
/// @param layers_delta
/// @param output_interlayers_Delta
/// @todo

Matrix<double> LossIndex::calculate_single_hidden_layer_point_Hessian
(const Vector< Vector<double> >& layers_activation_derivative,
 const Vector< Vector<double> >& layers_activation_second_derivative,
 const Vector< Vector< Vector<double> > >& perceptrons_combination_parameters_gradient,
 const Vector< Vector<double> >& layers_delta,
 const Matrix<double>& output_interlayers_Delta) const
{
    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    const size_t parameters_number = multilayer_perceptron_pointer->get_parameters_number();

    const size_t first_layer_parameters_number = multilayer_perceptron_pointer->get_layer(0).get_parameters().size();
    const size_t second_layer_parameters_number = multilayer_perceptron_pointer->get_layer(1).get_parameters().size();

    Matrix<double> single_hidden_layer_point_Hessian(parameters_number, parameters_number, 0.0);

//    Vector<size_t> parameter_indices(3);

    size_t layer_index_i;
    size_t neuron_index_i;
    size_t parameter_index_i;

    size_t layer_index_j;
    size_t neuron_index_j;
    size_t parameter_index_j;

    const Matrix<size_t> parameters_indices = multilayer_perceptron_pointer->get_parameters_indices();

    // Both weights in the second layer

    for(size_t i = first_layer_parameters_number; i < second_layer_parameters_number + first_layer_parameters_number; i++)
    {
        layer_index_i = parameters_indices(i,0);
        neuron_index_i = parameters_indices(i,1);
        parameter_index_i = parameters_indices(i,2);


        for(size_t j = first_layer_parameters_number; j < second_layer_parameters_number + first_layer_parameters_number; j++)
        {
            layer_index_j = parameters_indices(j,0);
            neuron_index_j = parameters_indices(j,1);
            parameter_index_j = parameters_indices(j,2);

            single_hidden_layer_point_Hessian(i,j) =
            perceptrons_combination_parameters_gradient[layer_index_i][neuron_index_i][parameter_index_i]
            *perceptrons_combination_parameters_gradient[layer_index_j][neuron_index_j][parameter_index_j]
            *calculate_Kronecker_delta(neuron_index_i,neuron_index_j)
            *output_interlayers_Delta(neuron_index_j,neuron_index_i);
        }
    }

    // One weight in each layer

    const Matrix<double> second_layer_weights = multilayer_perceptron_pointer->get_layer(1).get_synaptic_weights();

    for(size_t i = 0; i < first_layer_parameters_number; i++)
    {
        layer_index_i = parameters_indices(i,0);
        neuron_index_i = parameters_indices(i,1);
        parameter_index_i = parameters_indices(i,2);

        for(size_t j = first_layer_parameters_number; j < first_layer_parameters_number + second_layer_parameters_number; j++)
        {
            layer_index_j = parameters_indices(j,0);
            neuron_index_j = parameters_indices(j,1);
            parameter_index_j = parameters_indices(j,2);

            single_hidden_layer_point_Hessian(i,j) =
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
        layer_index_i = parameters_indices(i,0);
        neuron_index_i = parameters_indices(i,1);
        parameter_index_i = parameters_indices(i,2);

        for(size_t j = 0; j < first_layer_parameters_number; j++)
        {
            layer_index_j = parameters_indices(j,0);
            neuron_index_j = parameters_indices(j,1);
            parameter_index_j = parameters_indices(j,2);

            double sum = 0.0;

            for(size_t k = 0; k < outputs_number; k++)
            {
                sum += second_layer_weights(k, neuron_index_i)
                       *second_layer_weights(k, neuron_index_j)
                       *output_interlayers_Delta(k,k);
            }

            single_hidden_layer_point_Hessian(i, j) =
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
            single_hidden_layer_point_Hessian(j,i) = single_hidden_layer_point_Hessian(i,j);
        }
    }
    return single_hidden_layer_point_Hessian;
}


double LossIndex::calculate_loss() const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    const size_t instances_number = data_set_pointer->get_instances().get_instances_number();

    return calculate_loss(Vector<size_t>(0,1,instances_number-1));
}


double LossIndex::calculate_loss(const Vector<double>& parameters) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    const size_t instances_number = data_set_pointer->get_instances().get_instances_number();

    return calculate_loss(Vector<size_t>(0,1,instances_number-1), parameters);
}


double LossIndex::calculate_loss(const Vector<size_t>& instances_indices) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    if(regularization_method == None)
    {
        return calculate_error(instances_indices);
    }
    else
    {
        return calculate_error(instances_indices) + regularization_weight*calculate_regularization();
    }
}


double LossIndex::calculate_loss(const Vector<size_t>& instances_indices, const Vector<double>& parameters) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    if(regularization_method == None)
    {
        return calculate_error(instances_indices, parameters);
    }
    else
    {
        return calculate_error(instances_indices, parameters) + regularization_weight*calculate_regularization(parameters);
    }
}


double LossIndex::calculate_loss(const Vector<size_t>& instances_indices, const Vector<double>& direction, const double& rate) const
{
     const Vector<double> parameters = neural_network_pointer->get_parameters();
     const Vector<double> increment = direction*rate;

     return calculate_loss(instances_indices, parameters + increment);
}


double LossIndex::calculate_training_loss(const Vector<double>& direction, const double& rate) const
{
    const Vector<double> parameters = neural_network_pointer->get_parameters();
    const Vector<double> increment = direction*rate;

    if(regularization_method == None)
    {
        return calculate_training_error(parameters + increment);
    }
    else
    {
        return calculate_training_error(parameters + increment) + regularization_weight*calculate_regularization(parameters);
    }
}


double LossIndex::calculate_training_loss() const
{
    if(regularization_method == None)
    {
        return calculate_training_error();
    }
    else
    {
        return calculate_training_error() + regularization_weight*calculate_regularization();
    }
}


Vector<double> LossIndex::calculate_loss_gradient() const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    const size_t instances_number = data_set_pointer->get_instances().get_instances_number();

    return calculate_loss_gradient(Vector<size_t>(0,1,instances_number-1));
}


Vector<double> LossIndex::calculate_loss_gradient(const Vector<size_t>& instances_indices) const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    if(regularization_method == None)
    {
        return calculate_error_gradient(instances_indices);
    }
    else
    {
        return calculate_error_gradient(instances_indices) + regularization_weight*calculate_regularization();
    }
}


Vector<double> LossIndex::calculate_training_loss_gradient() const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    if(regularization_method == None)
    {
        return calculate_training_error_gradient();
    }
    else
    {
        return calculate_training_error_gradient() + regularization_weight*calculate_regularization();
    }
}


Matrix<double> LossIndex::calculate_loss_Hessian() const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    const size_t instances_number = data_set_pointer->get_instances().get_instances_number();

    return calculate_loss_Hessian(Vector<size_t>(0,1,instances_number-1));
}


Matrix<double> LossIndex::calculate_loss_Hessian(const Vector<size_t>& instances_indices) const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    const Matrix<double> error_Hessian = calculate_error_Hessian(instances_indices);

    const Matrix<double> regularization_Hessian = calculate_regularization_Hessian();

    return error_Hessian + regularization_Hessian*regularization_weight;
}


Vector<double> LossIndex::calculate_error_gradient(const Vector<size_t>& instances_indices) const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    // Data set stuff

    const Matrix<double> inputs = data_set_pointer->get_input_data(instances_indices);
    const Matrix<double> targets = data_set_pointer->get_target_data(instances_indices);

    // Neural network stuff

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const MultilayerPerceptron::FirstOrderForwardPropagation first_order_forward_propagation
            = multilayer_perceptron_pointer->calculate_first_order_forward_propagation(inputs);

    // Loss index stuff

    const Matrix<double> output_gradient = calculate_output_gradient(first_order_forward_propagation.layers_activations.get_last(), targets);

    const Vector< Matrix<double> > layers_delta = calculate_layers_delta(first_order_forward_propagation.layers_activation_derivatives, output_gradient);

    return calculate_error_gradient(inputs, first_order_forward_propagation.layers_activations, layers_delta);
}


/// @todo

Matrix<double> LossIndex::calculate_error_Hessian(const Vector<size_t>&) const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif
/*
    // Neural network stuff

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

    if(layers_number == 1)
    {
        return(calculate_Hessian_one_layer());
    }
    else if(layers_number == 2)
    {
        return(calculate_Hessian_two_layers());
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SumSquaredError class.\n"
               << "Matrix<double> calculate_Hessian() method.\n"
               << "This method is under development for more than one hidden layer.\n";

        throw logic_error(buffer.str());
    }

    // @todo General method for the Hessian matrix

    const size_t parameters_number = multilayer_perceptron_pointer->get_parameters_number();

    const Vector<size_t> layers_perceptrons_number = multilayer_perceptron_pointer->get_layers_perceptrons_numbers();

    Vector< Vector< Vector<double> > > second_order_forward_propagation(3);

    Vector < Vector< Vector<double> > > perceptrons_combination_parameters_gradient(layers_number);
    Matrix < Matrix<double> > interlayers_combination_combination_Jacobian;

    // Data set stuff

    const Instances& instances = data_set_pointer->get_instances();

    const Vector<size_t> training_indices = instances.get_training_indices();

    const size_t training_instances_number = training_indices.size();

    size_t training_index;

    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> inputs_indices = variables.get_inputs_indices();
    const Vector<size_t> targets_indices = variables.get_targets_indices();

    Vector<double> inputs(inputs_number);
    Vector<double> targets(outputs_number);

    // Sum squared error stuff

    Vector< Vector<double> > layers_delta(layers_number);
    Matrix<double> output_interlayers_Delta;

    Vector<double> output_gradient(outputs_number);
    Matrix<double> output_Hessian(outputs_number, outputs_number);

    Matrix<double> Hessian(parameters_number, parameters_number, 0.0);

    for(size_t i = 0; i < training_instances_number; i++)
    {
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

       output_gradient = calculate_output_gradient(instances_indices, layers_activation[layers_number-1], targets);

       output_Hessian = calculate_output_Hessian(layers_activation[layers_number-1], targets);

       layers_delta = calculate_layers_delta(layers_activation_derivative, output_gradient);

       output_interlayers_Delta = calculate_output_interlayers_Delta(layers_activation_derivative[layers_number-1],
                                                                 layers_activation_second_derivative[layers_number-1],
                                                                 output_gradient,
                                                                 output_Hessian);

       Hessian += calculate_single_hidden_layer_point_Hessian(layers_activation_derivative,
                                                              layers_activation_second_derivative,
                                                              perceptrons_combination_parameters_gradient,
                                                              layers_delta,
                                                              output_interlayers_Delta);
    }

    return(Hessian);
*/
    return Matrix<double>();

}


/// @todo

Matrix<double> LossIndex::calculate_Hessian_one_layer() const
{
    Matrix<double> Hessian_one_layer;

    return(Hessian_one_layer);
}


/// @todo

Matrix<double> LossIndex::calculate_Hessian_two_layers() const
{
/*
    // Neural network stuff

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

    const size_t parameters_number = multilayer_perceptron_pointer->get_parameters_number();

    const Vector<size_t> layers_perceptrons_number = multilayer_perceptron_pointer->get_layers_perceptrons_numbers();

    Vector< Vector< Vector<double> > > second_order_forward_propagation(3);

    Vector < Vector< Vector<double> > > perceptrons_combination_parameters_gradient(layers_number);
    Matrix < Matrix<double> > interlayers_combination_combination_Jacobian;

    // Data set stuff

    const Instances& instances = data_set_pointer->get_instances();

    const Vector<size_t> training_indices = instances.get_training_indices();

    const size_t training_instances_number = training_indices.size();

    size_t training_index;

    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> inputs_indices = variables.get_inputs_indices();
    const Vector<size_t> targets_indices = variables.get_targets_indices();

    Vector<double> inputs(inputs_number);
    Vector<double> targets(outputs_number);

    Vector< Vector<double> > layers_delta(layers_number);
    Matrix<double> output_interlayers_Delta;

    Vector<double> output_gradient(outputs_number);
    Matrix<double> output_Hessian(outputs_number, outputs_number);

    Matrix<double> Hessian_two_layers(parameters_number, parameters_number, 0.0);

    for(size_t i = 0; i < training_instances_number; i++)
    {
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

       output_gradient = calculate_output_gradient(instances_indices, layers_activation[layers_number-1], targets);

       output_Hessian = calculate_output_Hessian(layers_activation[layers_number-1], targets);

       layers_delta = calculate_layers_delta(layers_activation_derivative, output_gradient);

       output_interlayers_Delta = calculate_output_interlayers_Delta(layers_activation_derivative[layers_number-1],
                                                                     layers_activation_second_derivative[layers_number-1],
                                                                     output_gradient,
                                                                     output_Hessian);

       Hessian_two_layers += calculate_single_hidden_layer_point_Hessian(layers_activation_derivative,
                                                                         layers_activation_second_derivative,
                                                                         perceptrons_combination_parameters_gradient,
                                                                         layers_delta,
                                                                         output_interlayers_Delta);
    }

    return(Hessian_two_layers);
*/
    return Matrix<double>();
}


/// Returns a string with the default type of error term, "USER_PERFORMANCE_TERM".

string LossIndex::write_error_term_type() const
{
   return("USER_ERROR_TERM");
}


/// Returns a string with the default information of the error term.
/// It will be used by the training strategy to monitor the training process. 
/// By default this information is empty. 

string LossIndex::write_information() const
{
   return string();
}


double LossIndex::calculate_regularization() const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    switch(regularization_method)
    {
       case L1:
       {
            return neural_network_pointer->get_parameters().calculate_L1_norm();;
       }
       case L2:
       {
            return neural_network_pointer->get_parameters().calculate_L2_norm();
       }
       case None:
       {
            return 0.0;
       }
    }

    return 0.0;
}


double LossIndex::calculate_regularization(const Vector<double>& parameters) const
{
    switch(regularization_method)
    {
       case L1:
       {
            return parameters.calculate_L1_norm();
       }
       case L2:
       {
            return parameters.calculate_L2_norm();
       }
       case None:
       {
            return 0.0;
       }
    }

    return 0.0;
}


Vector<double> LossIndex::calculate_regularization_gradient() const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    switch(regularization_method)
    {
       case L1:
       {
            return neural_network_pointer->get_parameters().calculate_L1_norm_gradient();
       }
       case L2:
       {
            return neural_network_pointer->get_parameters().calculate_L2_norm_gradient();
       }
       case None:
       {
            return Vector<double>(neural_network_pointer->get_parameters_number(), 0.0);
       }
    }

    return Vector<double>();
}


Vector<double> LossIndex::calculate_regularization_gradient(const Vector<double>& parameters) const
{
    switch(regularization_method)
    {
       case L1:
       {
            return parameters.calculate_L1_norm_gradient();
       }
       case L2:
       {
            return parameters.calculate_L2_norm_gradient();
       }
       case None:
       {
            return Vector<double>(parameters.size(), 0.0);
       }
    }

    return Vector<double>();
}


Matrix<double> LossIndex::calculate_regularization_Hessian() const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    switch(regularization_method)
    {
       case L1:
       {
            return neural_network_pointer->get_parameters().calculate_L1_norm_Hessian();
       }
       case L2:
       {
            return neural_network_pointer->get_parameters().calculate_L2_norm_Hessian();
       }
       case None:
       {
            const size_t parameters_number = neural_network_pointer->get_parameters_number();

            return Matrix<double>(parameters_number,parameters_number,0.0);
       }
    }

    return Matrix<double>();
}



Matrix<double> LossIndex::calculate_regularization_Hessian(const Vector<double>& parameters) const
{
    switch(regularization_method)
    {
       case L1:
       {
            return parameters.calculate_L1_norm_Hessian();
       }
       case L2:
       {
            return parameters.calculate_L2_norm_Hessian();
       }
       case None:
       {
            const size_t parameters_number = parameters.size();

            return Matrix<double>(parameters_number,parameters_number,0.0);
       }
    }

    return Matrix<double>();
}


/// Returns the default string representation of a error term.

string LossIndex::object_to_string() const
{
   ostringstream buffer;

   buffer << "Error term\n";
          //<< "Display: " << display << "\n";

   return(buffer.str());
}


/// Serializes a default error term object into a XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this document.

tinyxml2::XMLDocument* LossIndex::to_XML() const
{
   ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Error term

   tinyxml2::XMLElement* root_element = document->NewElement("LossIndex");

   document->InsertFirstChild(root_element);

   return(document);
}


/// Serializes a default error term object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void LossIndex::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("LossIndex");

    file_stream.CloseElement();
}


/// Loads a default error term from a XML document.
/// @param document TinyXML document containing the error term members.

void LossIndex::from_XML(const tinyxml2::XMLDocument& document)
{
   // Display warnings

   const tinyxml2::XMLElement* display_element = document.FirstChildElement("Display");

   if(display_element)
   {
      string new_display_string = display_element->GetText();           

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


/// Returns the Knronecker delta of two integers a and b, which equals 1 if they are equal and 0 otherwise.
/// @param a First integer.
/// @param b Second integer.

size_t LossIndex::calculate_Kronecker_delta(const size_t& a, const size_t& b) const
{
   if(a == b)
   {
      return(1);
   }
   else
   {
      return(0);
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
