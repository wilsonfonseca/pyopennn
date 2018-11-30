/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   P E R C E P T R O N   L A Y E R   C L A S S                                                                */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "perceptron_layer.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates a empty layer object, with no perceptrons.
/// This constructor also initializes the rest of class members to their default values.

PerceptronLayer::PerceptronLayer()
{
   set();
}


// ARCHITECTURE CONSTRUCTOR

/// Layer architecture constructor. 
/// It creates a layer object with given numbers of inputs and perceptrons. 
/// The parameters are initialized at random. 
/// This constructor also initializes the rest of class members to their default values.
/// @param new_inputs_number Number of inputs in the layer.
/// @param new_perceptrons_number Number of perceptrons in the layer. 

PerceptronLayer::PerceptronLayer(const size_t& new_inputs_number, const size_t& new_perceptrons_number,
                                 const PerceptronLayer::ActivationFunction& new_activation_function)
{
   set(new_inputs_number, new_perceptrons_number, new_activation_function);
}
 

// COPY CONSTRUCTOR

/// Copy constructor. 
/// It creates a copy of an existing perceptron layer object. 
/// @param other_perceptron_layer Perceptron layer object to be copied.

PerceptronLayer::PerceptronLayer(const PerceptronLayer& other_perceptron_layer)
{
   set(other_perceptron_layer);
}


// DESTRUCTOR

/// Destructor.
/// This destructor does not delete any pointer.

PerceptronLayer::~PerceptronLayer()
{
}


// ASSIGNMENT OPERATOR

/// Assignment operator. 
/// It assigns to this object the members of an existing perceptron layer object.
/// @param other_perceptron_layer Perceptron layer object to be assigned.

PerceptronLayer& PerceptronLayer::operator = (const PerceptronLayer& other_perceptron_layer)
{
   if(this != &other_perceptron_layer) 
   {
      display = other_perceptron_layer.display;
   }

   return(*this);
}


// EQUAL TO OPERATOR


/// Equal to operator. 
/// It compares this object with another object of the same class. 
/// It returns true if the members of the two objects have the same values, and false otherwise.
/// @ param other_perceptron_layer Perceptron layer to be compared with.

bool PerceptronLayer::operator == (const PerceptronLayer& other_perceptron_layer) const
{
   if(display == other_perceptron_layer.display)
   {
      return(true);
   }
   else
   {
      return(false);
   }
}


// METHODS


/// Returns the number of inputs to the layer.

size_t PerceptronLayer::get_inputs_number() const
{
    return synaptic_weights.get_rows_number();
}


/// Returns the size of the perceptrons vector.

size_t PerceptronLayer::get_perceptrons_number() const
{
   return biases.size();
}


/// Returns the number of parameters(biases and synaptic weights) of the layer.

size_t PerceptronLayer::get_parameters_number() const
{
   return biases.size() + synaptic_weights.size();
}


/// Returns the biases from all the perceptrons in the layer. 
/// The format is a vector of real values. 
/// The size of this vector is the number of neurons in the layer.

const Vector<double>& PerceptronLayer::get_biases() const
{   
   return(biases);
}


/// Returns the synaptic weights from the perceptrons. 
/// The format is a matrix of real values. 
/// The number of rows is the number of neurons in the layer. 
/// The number of columns is the number of inputs to the layer. 

const Matrix<double>& PerceptronLayer::get_synaptic_weights() const
{
   return(synaptic_weights);
}


Matrix<double> PerceptronLayer::get_synaptic_weights(const Vector<double>& parameters) const
{
    const size_t inputs_number = get_inputs_number();
    const size_t perceptrons_number = get_perceptrons_number();

    const size_t synaptic_weights_number = synaptic_weights.size();

    return parameters.get_first(synaptic_weights_number).to_matrix(inputs_number, perceptrons_number);
}


Vector<double> PerceptronLayer::get_biases(const Vector<double>& parameters) const
{
    const size_t biases_number = biases.size();

    return parameters.get_last(biases_number);
}



/// Returns a single vector with all the layer parameters. 
/// The format is a vector of real values. 
/// The size is the number of parameters in the layer. 

Vector<double> PerceptronLayer::get_parameters() const
{
/*
    const size_t parameters_number = get_parameters_number();

    const size_t inputs_number = get_inputs_number();
    const size_t perceptrons_number = get_perceptrons_number();

    Vector<double> parameters(parameters_number);

    size_t index = 0;

    for(size_t i = 0; i < perceptrons_number; i++)
    {
        parameters[index] = biases[i];

        index++;

        for(size_t j = 0; j < inputs_number; j++)
        {
            parameters[index] = synaptic_weights(j,i);

            index++;
        }
    }
*/    

    return synaptic_weights.to_vector().assemble(biases);
}


/// Returns the activation function of the layer.
/// The activation function of a layer is the activation function of all perceptrons in it.

const PerceptronLayer::ActivationFunction& PerceptronLayer::get_activation_function() const
{
    return activation_function;
}


/// Returns a string with the name of the layer activation function.
/// This can be: Logistic, HyperbolicTangent, Threshold, SymmetricThreshold or Linear.

string PerceptronLayer::write_activation_function() const
{
   switch(activation_function)
   {
      case Logistic:
      {
         return("Logistic");
      }

      case HyperbolicTangent:
      {
         return("HyperbolicTangent");
      }

      case Threshold:
      {
         return("Threshold");
      }

      case SymmetricThreshold:
      {
         return("SymmetricThreshold");
      }

      case Linear:
      {
         return("Linear");
      }
   }

    return string();
}


/// Returns true if messages from this class are to be displayed on the screen, 
/// or false if messages from this class are not to be displayed on the screen.

const bool& PerceptronLayer::get_display() const
{
   return(display);
}


/// Sets an empty layer, wihtout any perceptron.
/// It also sets the rest of members to their default values. 

void PerceptronLayer::set()
{
   set_default();
}


/// Sets new numbers of inputs and perceptrons in the layer.
/// It also sets the rest of members to their default values. 
/// @param new_inputs_number Number of inputs.
/// @param new_perceptrons_number Number of perceptron neurons.

void PerceptronLayer::set(const size_t& new_inputs_number, const size_t& new_perceptrons_number,
                          const PerceptronLayer::ActivationFunction& new_activation_function)
{
    biases.set(new_perceptrons_number);

    biases.randomize_normal();

    synaptic_weights.set(new_inputs_number, new_perceptrons_number);

    synaptic_weights.randomize_normal();
   
    activation_function = new_activation_function;

    set_default();
}


/// Sets the members of this perceptron layer object with those from other perceptron layer object. 
/// @param other_perceptron_layer PerceptronLayer object to be copied.

void PerceptronLayer::set(const PerceptronLayer& other_perceptron_layer)
{   
   biases = other_perceptron_layer.biases;

   synaptic_weights = other_perceptron_layer.synaptic_weights;

   activation_function = other_perceptron_layer.activation_function;

   display = other_perceptron_layer.display;
}


/// Sets those members not related to the vector of perceptrons to their default value. 
/// <ul>
/// <li> Display: True.
/// </ul> 

void PerceptronLayer::set_default()
{
   display = true;
}


/// Sets a new number of inputs in the layer. 
/// The new synaptic weights are initialized at random. 
/// @param new_inputs_number Number of layer inputs.
 
void PerceptronLayer::set_inputs_number(const size_t& new_inputs_number)
{
    const size_t perceptrons_number = get_perceptrons_number();

    biases.set(perceptrons_number);

    synaptic_weights.set(new_inputs_number, perceptrons_number);
}


/// Sets a new number perceptrons in the layer. 
/// All the parameters are also initialized at random.
/// @param new_perceptrons_number New number of neurons in the layer.

void PerceptronLayer::set_perceptrons_number(const size_t& new_perceptrons_number)
{    
    const size_t inputs_number = get_inputs_number();

    biases.set(new_perceptrons_number);
    synaptic_weights.set(inputs_number, new_perceptrons_number);
}


/// Sets the biases of all perceptrons in the layer from a single vector.
/// @param new_biases New set of biases in the layer. 

void PerceptronLayer::set_biases(const Vector<double>& new_biases)
{
    biases = new_biases;
}


/// Sets the synaptic weights of this perceptron layer from a single matrix.
/// The format is a matrix of real numbers. 
/// The number of rows is the number of neurons in the corresponding layer. 
/// The number of columns is the number of inputs to the corresponding layer. 
/// @param new_synaptic_weights New set of synaptic weights in that layer. 

void PerceptronLayer::set_synaptic_weights(const Matrix<double>& new_synaptic_weights)
{
    synaptic_weights = new_synaptic_weights;
}


/// Sets the parameters of this layer. 
/// @param new_parameters Parameters vector for that layer. 

void PerceptronLayer::set_parameters(const Vector<double>& new_parameters)
{
    const size_t perceptrons_number = get_perceptrons_number();
    const size_t inputs_number = get_inputs_number();

    const size_t parameters_number = get_parameters_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

    const size_t new_parameters_size = new_parameters.size();

   if(new_parameters_size != parameters_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: PerceptronLayer class.\n"
             << "void set_parameters(const Vector<double>&) method.\n"
             << "Size of new parameters (" << new_parameters_size << ") must be equal to number of parameters (" << parameters_number << ").\n";

	  throw logic_error(buffer.str());
   }

   #endif

//   size_t index = 0;

//   for(size_t i = 0; i < perceptrons_number; i++)
//   {
//       biases[i] = new_parameters[index];
//       index++;

//       for(size_t j = 0; j < inputs_number; j++)
//       {
//           synaptic_weights(j,i) = new_parameters[index];
//           index++;
//       }
//   }

   synaptic_weights = new_parameters.get_subvector(0, inputs_number*perceptrons_number-1).to_matrix(inputs_number, perceptrons_number);

   biases = new_parameters.get_subvector(inputs_number*perceptrons_number, parameters_number-1);
}


/// This class sets a new activation(or transfer) function in a single layer. 
/// @param new_activation_function Activation function for the layer.

void PerceptronLayer::set_activation_function(const PerceptronLayer::ActivationFunction& new_activation_function)
{
    activation_function = new_activation_function;
}


/// Sets a new activation(or transfer) function in a single layer. 
/// The second argument is a string containing the name of the function("Logistic", "HyperbolicTangent", "Threshold", etc).
/// @param new_activation_function Activation function for that layer. 

void PerceptronLayer::set_activation_function(const string& new_activation_function_name)
{
    if(new_activation_function_name == "Logistic")
    {
       activation_function = Logistic;
    }
    else if(new_activation_function_name == "HyperbolicTangent")
    {
       activation_function = HyperbolicTangent;
    }
    else if(new_activation_function_name == "Threshold")
    {
       activation_function = Threshold;
    }
    else if(new_activation_function_name == "SymmetricThreshold")
    {
       activation_function = SymmetricThreshold;
    }
    else if(new_activation_function_name == "Linear")
    {
       activation_function = Linear;
    }
    else
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Perceptron class.\n"
              << "void set_activation_function(const string&) method.\n"
              << "Unknown activation function: " << new_activation_function_name << ".\n";

       throw logic_error(buffer.str());
    }
}


/// Sets a new display value. 
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void PerceptronLayer::set_display(const bool& new_display)
{
   display = new_display;
}


/// Makes the perceptron layer to have one more input.

void PerceptronLayer::grow_input()
{
}


/// Makes the perceptron layer to have one more perceptron.

void PerceptronLayer::grow_perceptron()
{
}


/// Makes the perceptron layer to have perceptrons_added more perceptrons.
/// @param perceptrons_added Number of perceptrons to be added.

void PerceptronLayer::grow_perceptrons(const size_t&)
{
}


/// This method removes a given input from the layer of perceptrons.
/// @param index Index of input to be pruned.

void PerceptronLayer::prune_input(const size_t& index)
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_number = get_inputs_number();

    if(index >= inputs_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: PerceptronLayer class.\n"
              << "void prune_input(const size_t&) method.\n"
              << "Index of input is equal or greater than number of inputs.\n";

       throw logic_error(buffer.str());
    }

    #endif
}


/// This method removes a given perceptron from the layer.
/// @param index Index of perceptron to be pruned.

void PerceptronLayer::prune_perceptron(const size_t& index)
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t perceptrons_number = get_perceptrons_number();

    if(index >= perceptrons_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: PerceptronLayer class.\n"
              << "void prune_perceptron(const size_t&) method.\n"
              << "Index of perceptron is equal or greater than number of perceptrons.\n";

       throw logic_error(buffer.str());
    }

    #endif

}


/// Initializes the perceptron layer with a random number of inputs and a randon number of perceptrons.
/// That can be useful for testing purposes. 

void PerceptronLayer::initialize_random()
{
   const size_t inputs_number = rand()%10 + 1;
   const size_t perceptrons_number = rand()%10 + 1;

   set(inputs_number, perceptrons_number, PerceptronLayer::HyperbolicTangent);
   
   set_display(true);
}


/// Initializes the biases of all the perceptrons in the layer of perceptrons with a given value. 
/// @param value Biases initialization value. 

void PerceptronLayer::initialize_biases(const double& value)
{
    biases.initialize(value);
}


/// Initializes the synaptic weights of all the perceptrons in the layer of perceptrons perceptron with a given value. 
/// @param value Synaptic weights initialization value. 

void PerceptronLayer::initialize_synaptic_weights(const double& value) 
{
    synaptic_weights.initialize(value);
}


/// Initializes all the biases and synaptic weights in the neural newtork with a given value.
/// @param value Parameters initialization value. 

void PerceptronLayer::initialize_parameters(const double& value)
{
    biases.initialize(value);

    synaptic_weights.initialize(value);
}


/// Initializes all the biases and synaptic weights in the neural newtork at random with values comprised 
/// between -1 and +1.

void PerceptronLayer::randomize_parameters_uniform()
{
   biases.randomize_uniform();

   synaptic_weights.randomize_uniform();
}


/// Initializes all the biases and synaptic weights in the layer of perceptrons at random with values 
/// comprised between a minimum and a maximum values.
/// @param minimum Minimum initialization value.
/// @param maximum Maximum initialization value.

void PerceptronLayer::randomize_parameters_uniform(const double& minimum, const double& maximum)
{
    biases.randomize_uniform(minimum, maximum);

    synaptic_weights.randomize_uniform(minimum, maximum);
}


/// Initializes all the biases and synaptic weights in the layer of perceptrons at random, with values 
/// comprised between different minimum and maximum numbers for each parameter.
/// @param minimum Vector of minimum initialization values.
/// @param maximum Vector of maximum initialization values.

void PerceptronLayer::randomize_parameters_uniform(const Vector<double>& minimum, const Vector<double>& maximum)
{
    biases.randomize_uniform(minimum, maximum);

    synaptic_weights.randomize_uniform(minimum, maximum);
}


/// Initializes all the biases and synaptic weights in the layer of perceptrons at random, with values 
/// comprised between a different minimum and maximum numbers for each parameter.
/// All minimum are maximum initialization values must be given from a vector of two real vectors.
/// The first element must contain the minimum inizizalization value for each parameter.
/// The second element must contain the maximum inizizalization value for each parameter.
/// @param minimum_maximum Vector of minimum and maximum initialization values.

void PerceptronLayer::randomize_parameters_uniform(const Vector< Vector<double> >& minimum_maximum)
{
   const size_t parameters_number = get_parameters_number();

   Vector<double> parameters(parameters_number);

   parameters.randomize_uniform(minimum_maximum[0], minimum_maximum[1]);

   set_parameters(parameters);
}


/// Initializes all the biases and synaptic weights in the newtork with random values chosen from a 
/// normal distribution with mean 0 and standard deviation 1.

void PerceptronLayer::randomize_parameters_normal()
{
    biases.randomize_normal();

    synaptic_weights.randomize_normal();
}


/// Initializes all the biases and synaptic weights in the layer of perceptrons with random random values 
/// chosen from a normal distribution with a given mean and a given standard deviation.
/// @param mean Mean of normal distribution.
/// @param standard_deviation Standard deviation of normal distribution.

void PerceptronLayer::randomize_parameters_normal(const double& mean, const double& standard_deviation)
{
    biases.randomize_normal(mean, standard_deviation);

    synaptic_weights.randomize_normal(mean, standard_deviation);
}


/// Initializes all the biases an synaptic weights in the layer of perceptrons with random values chosen 
/// from normal distributions with different mean and standard deviation for each parameter.
/// @param mean Vector of mean values.
/// @param standard_deviation Vector of standard deviation values.

void PerceptronLayer::randomize_parameters_normal(const Vector<double>& mean, const Vector<double>& standard_deviation)
{
    biases.randomize_normal(mean, standard_deviation);

    synaptic_weights.randomize_normal(mean, standard_deviation);
}


/// Initializes all the biases and synaptic weights in the layer of perceptrons with random values chosen 
/// from normal distributions with different mean and standard deviation for each parameter.
/// All mean and standard deviation values are given from a vector of two real vectors.
/// The first element must contain the mean value for each parameter.
/// The second element must contain the standard deviation value for each parameter.
/// @param mean_standard_deviation Vector of mean and standard deviation values.

void PerceptronLayer::randomize_parameters_normal(const Vector< Vector<double> >& mean_standard_deviation)
{
   const size_t parameters_number = get_parameters_number();

   Vector<double> parameters(parameters_number);

   parameters.randomize_normal(mean_standard_deviation[0], mean_standard_deviation[1]);

   set_parameters(parameters);
}


/// Calculates the norm of a layer parameters vector. 

double PerceptronLayer::calculate_parameters_norm() const
{
   return(get_parameters().calculate_L2_norm());
}


/// Returns the combination to every perceptron in the layer as a function of the inputs to that layer. 
/// @param inputs Inputs to the layer.

Vector<double> PerceptronLayer::calculate_combinations(const Vector<double>& inputs) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   const size_t inputs_size = inputs.size();

   const size_t inputs_number = get_inputs_number();

   if(inputs_size != inputs_number) 
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: PerceptronLayer class.\n"
             << "Vector<double> calculate_combinations(const Vector<double>&) const method.\n"
             << "Size of inputs to layer must be equal to number of layer inputs.\n";

	  throw logic_error(buffer.str());
   }   

   #endif

   return inputs.dot(synaptic_weights) + biases;
}


/// Returns the partial derivatives of the combination of a layer with respect to the inputs. 
/// All that partial derivatives are getd in the so called Jacobian matrix of the layer combination function.

Matrix<double> PerceptronLayer::calculate_combinations_Jacobian(const Vector<double>&) const
{
   return(synaptic_weights);
}


/// Returns the second partial derivatives of the combination of a layer with respect to the inputs of that layer. 
/// All that partial derivatives are getd in the so called Hessian form, represented as a vector of matrices,
/// of the layer combination function.

Vector< Matrix<double> > PerceptronLayer::calculate_combinations_Hessian(const Vector<double>&) const
{
   const size_t inputs_number = get_inputs_number();
   const size_t perceptrons_number = get_perceptrons_number();

   Vector< Matrix<double> > combination_Hessian(perceptrons_number);

   for(size_t i = 0; i < perceptrons_number; i++)
   {
      combination_Hessian[i].set(inputs_number, inputs_number, 0.0);
   }

   return(combination_Hessian);
}


Matrix<double> PerceptronLayer::calculate_combinations(const Matrix<double>& inputs) const
{
    return inputs.calculate_linear_combinations(synaptic_weights, biases);
}


Vector<double> PerceptronLayer::calculate_combinations(const Vector<double>& inputs, const Vector<double>& parameters) const
{
    const Matrix<double> new_synaptic_weights = get_synaptic_weights(parameters);
    const Vector<double> new_biases = get_biases(parameters);

    return calculate_combinations(inputs, new_biases, new_synaptic_weights);
}


Matrix<double> PerceptronLayer::calculate_combinations(const Matrix<double>& inputs, const Vector<double>& parameters) const
{
    const Matrix<double> new_synaptic_weights = get_synaptic_weights(parameters);
    const Vector<double> new_biases = get_biases(parameters);

    return calculate_combinations(inputs, new_biases, new_synaptic_weights);
}


/// Returns which would be the combination of a layer as a function of the inputs and for a set of parameters. 
/// @param inputs Vector of inputs to that layer. 
/// @param parameters Vector of parameters in the layer. 

Vector<double> PerceptronLayer::calculate_combinations(const Vector<double>& inputs,
                                                       const Vector<double>& new_biases, const Matrix<double>& new_synaptic_weights) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   const size_t inputs_size = inputs.size();
   const size_t inputs_number = get_inputs_number();

   if(inputs_size != inputs_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: PerceptronLayer class.\n"
             << "Vector<double> calculate_combination_parameters(const Vector<double>&, const Vector<double>&) const method.\n"
             << "Size of layer inputs (" << inputs_size << ") must be equal to number of layer inputs (" << inputs_number << ").\n";

	  throw logic_error(buffer.str());
   }

   #endif

    return new_biases + inputs.dot(new_synaptic_weights);
}


/// Returns the partial derivatives of the combination of a layer with respect to the parameters in that layer, for a given set of inputs. 
/// All that partial derivatives are getd in the so called Jacobian matrix of the layer combination function.
/// @param inputs Vector of inputs to that layer. 

Matrix<double> PerceptronLayer::calculate_combinations_parameters_Jacobian(const Vector<double>& inputs) const
{
   const size_t perceptrons_number = get_perceptrons_number();
   const size_t parameters_number = get_parameters_number();
   const size_t inputs_number = get_inputs_number();

   Matrix<double> combinations_Jacobian(perceptrons_number, parameters_number, 0.0);

   size_t column_index = 0;

   // Synaptic weights

   for(size_t i = 0; i < perceptrons_number; i++)
   {
       for(size_t j = 0; j < inputs_number; j++)
       {
           combinations_Jacobian(i,column_index) = inputs[j];

           column_index++;
       }
   }

   // Biases

   for(size_t i = 0; i < perceptrons_number; i++)
   {
       combinations_Jacobian(i,column_index) = 1.0;

       column_index++;
   }

   return(combinations_Jacobian);
}


Vector< Matrix<double> > PerceptronLayer::calculate_combinations_parameters_Jacobian(const Matrix<double>& input_data) const
{
   const size_t rows_number = input_data.get_rows_number();

   const size_t perceptrons_number = get_perceptrons_number();
   const size_t parameters_number = get_parameters_number();
   const size_t inputs_number = get_inputs_number();

   Vector< Matrix<double> > combinations_parameters_Jacobian(rows_number);

   for(size_t row = 0; row < rows_number; row++)
   {
       const Vector<double> inputs = input_data.get_row(row);

       combinations_parameters_Jacobian[row].set(perceptrons_number, parameters_number, 0.0);

       size_t column_index = 0;

       // Synaptic weights

       for(size_t i = 0; i < perceptrons_number; i++)
       {
           for(size_t j = 0; j < inputs_number; j++)
           {
               combinations_parameters_Jacobian[row](i,column_index) = inputs[j];

               column_index++;
           }
       }

       // Biases

       for(size_t i = 0; i < perceptrons_number; i++)
       {
           combinations_parameters_Jacobian[row](i,column_index) = 1.0;

           column_index++;
       }
   }

   return combinations_parameters_Jacobian;
}



/// Returns the second partial derivatives of the combination of a layer
/// with respect to the parameters in that layer, for a given set of inputs.
/// All that partial derivatives are getd in the so called Hessian form,
/// represented as a vector of matrices, of the layer combination function.

Vector< Matrix<double> > PerceptronLayer::calculate_combinations_parameters_Hessian() const
{
   const size_t perceptrons_number = get_perceptrons_number();

   Vector< Matrix<double> > combination_parameters_Hessian(perceptrons_number);

   const size_t parameters_number = get_parameters_number();

   for(size_t i = 0; i < perceptrons_number; i++)
   {
      combination_parameters_Hessian[i].set(parameters_number, parameters_number, 0.0);
   }

   return(combination_parameters_Hessian);
}


Matrix<double> PerceptronLayer::calculate_combinations(const Matrix<double>& inputs, const Vector<double>& new_biases, const Matrix<double>& new_synaptic_weights) const
{
    return inputs.calculate_linear_combinations(new_synaptic_weights, new_biases);
}


/// Returns the activations from every perceptron in a layer as a function of their combination.
/// @param combinations Combination from every neuron in the layer.

Vector<double> PerceptronLayer::calculate_activations(const Vector<double>& combinations) const
{
   const size_t perceptrons_number = get_perceptrons_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   const size_t combination_size = combinations.size();

   if(combination_size != perceptrons_number) 
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: PerceptronLayer class.\n"
             << "Vector<double> calculate_activation(const Vector<double>&) const method.\n"
             << "Combination size must be equal to number of neurons.\n";

	  throw logic_error(buffer.str());
   }   

   if(combination_size == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: PerceptronLayer class.\n"
             << "Vector<double> calculate_activation(const Vector<double>&) const method.\n"
             << "Combination size cannot be empty.\n";

      throw logic_error(buffer.str());
   }

   #endif

   switch(activation_function)
   {
      case PerceptronLayer::Logistic:
      {
       return logistic(combinations);
      }

      case PerceptronLayer::HyperbolicTangent:
      {
           return hyperbolic_tangent(combinations);
      }

      case PerceptronLayer::Threshold:
      {
            return threshold(combinations);
      }

      case PerceptronLayer::SymmetricThreshold:
      {
           return symmetric_threshold(combinations);
      }

      case PerceptronLayer::Linear:
      {
           return combinations;
      }
   }

   return Vector<double>();
}  


/// Returns the activation derivative from every perceptron in the layer as a function of its combination.
/// @param combination Combination to every neuron in the layer.

Vector<double> PerceptronLayer::calculate_activations_derivatives(const Vector<double>& combinations) const
{
   const size_t perceptrons_number = get_perceptrons_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   const size_t combinations_size = combinations.size();

   if(combinations_size != perceptrons_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: PerceptronLayer class.\n"
             << "Vector<double> calculate_activations_derivatives(const Vector<double>&) const method.\n"
             << "Size of combination must be equal to number of neurons.\n";

	  throw logic_error(buffer.str());
   }   

   #endif

   switch(activation_function)
   {
      case PerceptronLayer::Logistic:
      {
       return logistic_derivative(combinations);
      }

      case PerceptronLayer::HyperbolicTangent:
      {
           return hyperbolic_tangent_derivative(combinations);
      }

      case PerceptronLayer::Threshold:
      {
            return threshold_derivative(combinations);
      }

      case PerceptronLayer::SymmetricThreshold:
      {
           return symmetric_threshold_derivative(combinations);
      }

      case PerceptronLayer::Linear:
      {
           return Vector<double>(perceptrons_number, 1.0);
      }
   }

   return Vector<double>();
}


/// Returns the activation second derivative from every perceptron as a function of their combination. 
/// @param combination Combination to every perceptron in the layer. 

Vector<double> PerceptronLayer::calculate_activations_second_derivatives(const Vector<double>& combinations) const
{
   const size_t perceptrons_number = get_perceptrons_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   const size_t combinations_size = combinations.size();

   if(combinations_size != perceptrons_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: PerceptronLayer class.\n"
             << "Vector<double> calculate_activations_second_derivatives(const Vector<double>&) const method.\n"
             << "Size of combinations must be equal to number of neurons.\n";

	  throw logic_error(buffer.str());
   }   

   #endif

   switch(activation_function)
   {
      case PerceptronLayer::Logistic:
      {
       return logistic_second_derivative(combinations);
      }

      case PerceptronLayer::HyperbolicTangent:
      {
           return hyperbolic_tangent_second_derivative(combinations);
      }

      case PerceptronLayer::Threshold:
      {
            return threshold_second_derivative(combinations);
      }

      case PerceptronLayer::SymmetricThreshold:
      {
           return symmetric_threshold_second_derivative(combinations);
      }

      case PerceptronLayer::Linear:
      {
           return Vector<double>(perceptrons_number, 0.0);
      }
   }

   return Vector<double>();
}


Matrix<double> PerceptronLayer::calculate_activations(const Matrix<double>& combinations) const
{

    #ifdef __OPENNN_DEBUG__

    const size_t perceptrons_number = get_perceptrons_number();

    const size_t combinations_columns_number = combinations.get_columns_number();

    if(combinations_columns_number != perceptrons_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: PerceptronLayer class.\n"
              << "Matrix<double> calculate_activations(const Matrix<double>&) const method.\n"
              << "Number of columns of combinations must be equal to number of neurons.\n";

       throw logic_error(buffer.str());
    }

    #endif

    switch(activation_function)
    {
       case PerceptronLayer::Logistic:
       {
        return logistic(combinations);
       }

       case PerceptronLayer::HyperbolicTangent:
       {
            return hyperbolic_tangent(combinations);
       }

       case PerceptronLayer::Threshold:
       {
             return threshold(combinations);
       }

       case PerceptronLayer::SymmetricThreshold:
       {
            return symmetric_threshold(combinations);
       }

       case PerceptronLayer::Linear:
       {
            return linear(combinations);
       }
    }

    return Matrix<double>();
}


Matrix<double> PerceptronLayer::calculate_activations_derivatives(const Matrix<double>& combinations) const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t perceptrons_number = get_perceptrons_number();

    const size_t combinations_columns_number = combinations.get_columns_number();

    if(combinations_columns_number != perceptrons_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: PerceptronLayer class.\n"
              << "Matrix<double> calculate_activations_derivatives(const Matrix<double>&) const method.\n"
              << "Number of columns of combination must be equal to number of neurons.\n";

       throw logic_error(buffer.str());
    }

    #endif

    switch(activation_function)
    {
       case PerceptronLayer::Logistic:
       {
        return logistic_derivative(combinations);
       }

       case PerceptronLayer::HyperbolicTangent:
       {
            return hyperbolic_tangent_derivative(combinations);
       }

       case PerceptronLayer::Threshold:
       {
             return threshold_derivative(combinations);
       }

       case PerceptronLayer::SymmetricThreshold:
       {
            return symmetric_threshold_derivative(combinations);
       }

       case PerceptronLayer::Linear:
       {
            return linear_derivative(combinations);
       }
    }

    return Matrix<double>();
}


/// Arranges a "Jacobian" matrix from a vector of derivatives. 
/// @param activation_derivative Vector of activation function derivatives. 

Matrix<double> PerceptronLayer::get_activations_Jacobian(const Vector<double>& activation_derivative) const
{
   const size_t perceptrons_number = get_perceptrons_number();

   Matrix<double> activation_Jacobian(perceptrons_number, perceptrons_number, 0.0);

   activation_Jacobian.set_diagonal(activation_derivative);

   return(activation_Jacobian);
}


/// Arranges a "Hessian form" vector of matrices from a vector of second derivatives. 
/// @param activation_second_derivative Vector of activation function second derivatives. 

Vector< Matrix<double> > PerceptronLayer::get_activations_Hessian(const Vector<double>& activation_second_derivative) const
{
   const size_t perceptrons_number = get_perceptrons_number();

   Vector< Matrix<double> > activation_Hessian(perceptrons_number);

   for(size_t i = 0; i < perceptrons_number; i++)
   {
      activation_Hessian[i].set(perceptrons_number, perceptrons_number, 0.0);
      activation_Hessian[i](i,i) = activation_second_derivative[i];
   }

   return(activation_Hessian);
}


/// Returns the outputs from every perceptron in a layer as a function of their inputs. 
/// @param inputs Input vector to the layer.

Vector<double> PerceptronLayer::calculate_outputs(const Vector<double>& inputs) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   const size_t inputs_size = inputs.size();

   const size_t inputs_number = get_inputs_number();

   if(inputs_size != inputs_number) 
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: PerceptronLayer class.\n"
             << "Vector<double> calculate_outputs(const Vector<double>&) const method.\n"
             << "Size of inputs must be equal to number of inputs to layer.\n";

	  throw logic_error(buffer.str());
   }   

   #endif

   return(calculate_activations(calculate_combinations(inputs)));
}


/// Returns the Jacobian matrix of a layer for a given inputs to that layer. 
/// This is composed by the derivatives of the layer outputs with respect to their inputs. 
/// The number of rows is the number of neurons in the layer.
/// The number of columns is the number of inputs to that layer.
/// @param inputs Input to layer.

Matrix<double> PerceptronLayer::calculate_Jacobian(const Vector<double>& inputs) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   const size_t inputs_number = get_inputs_number(); 

   const size_t inputs_size = inputs.size();

   if(inputs_size != inputs_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: PerceptronLayer class.\n"
             << "Matrix<double> calculate_Jacobian(const Vector<double>&) const method.\n"
             << "Size of inputs must be equal to number of inputs to layer.\n";

	  throw logic_error(buffer.str());
   }

   #endif

   const Vector<double> combinations = calculate_combinations(inputs);

   const Vector<double> activations_derivatives = calculate_activations_derivatives(combinations);

   return(synaptic_weights.multiply_rows(activations_derivatives));
}


/// @todo Implement formula.
/// Returns the second partial derivatives of the outputs from a layer with respect to the inputs to that layer. 
/// @param inputs Vector of inputs to that layer. 

Vector< Matrix<double> > PerceptronLayer::calculate_Hessian(const Vector<double>& inputs) const
{
   const size_t perceptrons_number = get_perceptrons_number();

   const Vector<double> combination = calculate_combinations(inputs);

   const Vector<double> activations_second_derivatives = calculate_activations_second_derivatives(combination);

   Vector< Matrix<double> > activation_Hessian(perceptrons_number);

   Vector< Matrix<double> > Hessian_form(perceptrons_number);

   for(size_t i = 0; i < perceptrons_number; i++)
   {
      activation_Hessian[i].set(perceptrons_number, perceptrons_number, 0.0);
      activation_Hessian[i](i,i) = activations_second_derivatives[i];

//	  Hessian_form[i] = synaptic_weights.calculate_transpose().dot(activation_Hessian[i]).dot(synaptic_weights);
   }

   return(Hessian_form);
}

 
/// Returns which would be the outputs from a layer for a given inputs to that layer and a set of parameters in that layer. 
/// @param inputs Vector of inputs to that layer. 
/// @param parameters Vector of parameters in that layer. 

Vector<double> PerceptronLayer::calculate_outputs(const Vector<double>& inputs, const Vector<double>& new_biases, const Matrix<double>& new_synaptic_weights) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   const size_t inputs_size = inputs.size();

   const size_t inputs_number = get_inputs_number();

   if(inputs_size != inputs_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: PerceptronLayer class.\n"
             << "Vector<double> calculate_outputs(const Vector<double>&, const Vector<double>&) const method.\n"
             << "Size of layer inputs (" << inputs_size << ") must be equal to number of layer inputs (" << inputs_number << ").\n";

	  throw logic_error(buffer.str());
   }

   #endif

   return(calculate_activations(calculate_combinations(inputs, new_biases, new_synaptic_weights)));
}


/// Returns the parameters Jacobian for a given set of inputs. 
/// This is composed by the derivatives of the layer outputs with respect to the layer parameters. 
/// The number of rows is the number of neurons in the layer.
/// The number of columns is the number of parameters in that layer.
/// @param inputs Set of inputs to the layer.
/// @param parameters Set of layer parameters.

Matrix<double> PerceptronLayer::calculate_Jacobian(const Vector<double>& inputs,
                                                   const Vector<double>& new_biases, const Matrix<double>& new_synaptic_weights) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   const size_t inputs_number = get_inputs_number(); 
   const size_t inputs_size = inputs.size();

   if(inputs_size != inputs_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: PerceptronLayer class.\n"
             << "Matrix<double> calculate_parameters_Jacobian(const Vector<double>&, const Vector<double>&) const method.\n"
             << "Size of inputs must be equal to number of inputs.\n";

	  throw logic_error(buffer.str());
   }

   #endif

//   const Vector<double> combinations = calculate_combinations(inputs, parameters);

//   const Matrix<double> combinations_Jacobian = calculate_combinations_Jacobian(inputs, parameters);

//   const Vector<double> activation_derivatives = calculate_activations_derivatives(combinations);
   
//   const Matrix<double> activation_Jacobian = get_activations_Jacobian(activation_derivatives);

//   return(activation_Jacobian.dot(combinations_Jacobian));

   return Matrix<double>();
}


/// Returns the second partial derivatives of the outputs from a layer with respect a given set of potential parameters for this layer,
/// and for a given set of inputs.
/// This quantity is the Hessian form of the layer outputs function, and it is represented as a vector of matrices. 
/// @param inputs Set of layer inputs. 
/// @param parameters Set of layer parameters.

Vector< Matrix<double> > PerceptronLayer::calculate_Hessian(const Vector<double>& inputs,
                                                            const Vector<double>& new_biases, const Matrix<double>& new_synaptic_weights) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   const size_t inputs_number = get_inputs_number(); 
   const size_t inputs_size = inputs.size();

   if(inputs_size != inputs_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: PerceptronLayer class.\n"
             << "Vector< Matrix<double> > calculate_Hessian(const Vector<double>&, const Vector<double>&) const method.\n"
             << "Size must be equal to number of inputs of layer.\n";

	  throw logic_error(buffer.str());
   }

   #endif

//   const size_t perceptrons_number = get_perceptrons_number();

//   const Vector<double> combination = calculate_combinations(inputs);

//   const Matrix<double> combination_parameters_Jacobian = calculate_combinations_Jacobian(inputs, parameters);

//   const Vector<double> activation_second_derivatives = calculate_activations_second_derivatives(combination);

//   const Vector< Matrix<double> > activation_Hessian = get_activations_Hessian(activation_second_derivatives);

//   // Calculate parameters Hessian form

//   Vector< Matrix<double> > parameters_Hessian(perceptrons_number);

//   for(size_t i = 0; i < perceptrons_number; i++)
//   {
//	  parameters_Hessian[i] = combination_parameters_Jacobian.calculate_transpose().dot(activation_Hessian[i]).dot(combination_parameters_Jacobian);
//   }

//   return(parameters_Hessian);

    return Vector< Matrix<double> >();
}


Matrix<double> PerceptronLayer::calculate_outputs(const Matrix<double>& inputs) const
{
    const Matrix<double> combinations(inputs.calculate_linear_combinations(synaptic_weights, biases));

    switch(activation_function)
    {
       case PerceptronLayer::Logistic:
       {
            return logistic(combinations);
       }

       case PerceptronLayer::HyperbolicTangent:
       {
            return hyperbolic_tangent(combinations);
       }

       case PerceptronLayer::Threshold:
       {
             return threshold(combinations);
       }

       case PerceptronLayer::SymmetricThreshold:
       {
            return symmetric_threshold(combinations);
       }

       case PerceptronLayer::Linear:
       {
            return combinations;
       }        
    }

    return Matrix<double>();
}


Matrix<double> PerceptronLayer::calculate_outputs_combinations(const Matrix<double>& combinations) const
{
    switch(activation_function)
    {
       case PerceptronLayer::Logistic:
       {
        return logistic(combinations);
       }

       case PerceptronLayer::HyperbolicTangent:
       {
            return hyperbolic_tangent(combinations);
       }

       case PerceptronLayer::Threshold:
       {
             return threshold(combinations);
       }

       case PerceptronLayer::SymmetricThreshold:
       {
            return symmetric_threshold(combinations);
       }

       case PerceptronLayer::Linear:
       {
            return combinations;
       }
    }

    return Matrix<double>();
}



Matrix<double> PerceptronLayer::calculate_outputs(const Matrix<double>& inputs, const Vector<double>& new_biases, const Matrix<double>& new_synaptic_weights) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t inputs_size = inputs.get_columns_number();

   const size_t inputs_number = get_inputs_number();

   if(inputs_size != inputs_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: PerceptronLayer class.\n"
             << "Vector<double> calculate_outputs(const Vector<double>&, const Vector<double>&) const method.\n"
             << "Size of layer inputs (" << inputs_size << ") must be equal to number of layer inputs (" << inputs_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   const Matrix<double> combinations(inputs.calculate_linear_combinations(new_synaptic_weights, new_biases));

   switch(activation_function)
   {
      case PerceptronLayer::Logistic:
      {
       return logistic(combinations);
      }

      case PerceptronLayer::HyperbolicTangent:
      {
           return hyperbolic_tangent(combinations);
      }

      case PerceptronLayer::Threshold:
      {
            return threshold(combinations);
      }

      case PerceptronLayer::SymmetricThreshold:
      {
           return symmetric_threshold(combinations);
      }

      case PerceptronLayer::Linear:
      {
           return combinations;
      }
   }

   return Matrix<double>();
}


/// Returns a string with the expression of the inputs-outputs relationship of the layer.
/// @param inputs_name Vector of strings with the name of the layer inputs. 
/// @param outputs_name Vector of strings with the name of the layer outputs. 

string PerceptronLayer::write_expression(const Vector<string>& inputs_name, const Vector<string>& outputs_name) const
{
   const size_t perceptrons_number = get_perceptrons_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   const size_t inputs_number = get_inputs_number(); 
   const size_t inputs_name_size = inputs_name.size();

   if(inputs_name_size != inputs_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: PerceptronLayer class.\n"
             << "string write_expression(const Vector<string>&, const Vector<string>&) const method.\n"
             << "Size of inputs name must be equal to number of layer inputs.\n";

	  throw logic_error(buffer.str());
   }

   const size_t outputs_name_size = outputs_name.size();

   if(outputs_name_size != perceptrons_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: PerceptronLayer class.\n"
             << "string write_expression(const Vector<string>&, const Vector<string>&) const method.\n"
             << "Size of outputs name must be equal to number of perceptrons.\n";

	  throw logic_error(buffer.str());
   }

   #endif

   ostringstream buffer;

//   for(size_t i = 0; i < perceptrons_number; i++)
//   {
//      buffer << perceptrons[i].write_expression(inputs_name, outputs_name[i]);
//   }

   return(buffer.str());
}

string PerceptronLayer::object_to_string() const
{
    const size_t inputs_number = get_inputs_number();
    const size_t perceptrons_number = get_perceptrons_number();

    ostringstream buffer;

    buffer << "Perceptron layer" << endl;
    buffer << "Inputs number: " << inputs_number << endl;
    buffer << "Activation function: " << write_activation_function() << endl;
    buffer << "Perceptrons number: " << perceptrons_number << endl;
    buffer << "Biases:\n " << biases << endl;
    buffer << "Synaptic_weights:\n" << synaptic_weights;

    return buffer.str();
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
