/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   R A N D O M   S E A R C H   C L A S S                                                                      */
/*                                                                                                              */ 

/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "random_search.h"

namespace OpenNN
{

/// Default constructor. 
/// It creates a random search training algorithm not associated to any loss index object. 
/// It also initializes the class members to their default values.

RandomSearch::RandomSearch() 
 : TrainingAlgorithm()
{
   set_default();
}


/// Loss index constructor. 
/// It creates a random search training algorithm associated to a loss index object. 
/// It also initializes the class members to their default values.
/// @param new_loss_index_pointer Pointer to a loss index object.

RandomSearch::RandomSearch(LossIndex* new_loss_index_pointer)
: TrainingAlgorithm(new_loss_index_pointer)
{   
   set_default();
}


// XML CONSTRUCTOR 

/// XML constructor. 
/// It creates a random search training algorithm not associated to any loss index object. 
/// It also loads the rest of class members from a XML document.
/// @param document TinyXML document containing the members of a random search object. 

RandomSearch::RandomSearch(const tinyxml2::XMLDocument& document) : TrainingAlgorithm(document)
{
   from_XML(document);
}


// DESTRUCTOR

/// Destructor.
/// It does not delete any object. 

RandomSearch::~RandomSearch()
{

}


// const double& get_warning_parameters_norm() const method

/// Returns the minimum value for the norm of the parameters vector at wich a warning message is written to the screen. 

const double& RandomSearch::get_warning_parameters_norm() const
{
   return(warning_parameters_norm);       
}


// const double& get_warning_training_rate() const method

/// Returns the training rate value at wich a warning message is written to the screen during line minimization.

const double& RandomSearch::get_warning_training_rate() const
{
   return(warning_training_rate);
}


// const double& get_error_parameters_norm() const method

/// Returns the value for the norm of the parameters vector at wich an error message is 
/// written to the screen and the program exits. 

const double& RandomSearch::get_error_parameters_norm() const
{
   return(error_parameters_norm);
}


// const double& get_error_training_rate() const method

/// Returns the training rate value at wich the line minimization algorithm is assumed to fail when 
/// bracketing a minimum.

const double& RandomSearch::get_error_training_rate() const
{
   return(error_training_rate);
}


// const double& get_loss_goal() const method

/// Returns the goal value for the loss. 
/// This is used as a stopping criterion when training a multilayer perceptron

const double& RandomSearch::get_loss_goal() const
{
   return(loss_goal);
}


// const size_t& get_maximum_selection_loss_decreases() const method

/// Returns the maximum number of selection failures during the training process. 

const size_t& RandomSearch::get_maximum_selection_loss_decreases() const
{
   return(maximum_selection_loss_decreases);
}


// const size_t& get_maximum_iterations_number() const method

/// Returns the maximum number of iterations for training.

const size_t& RandomSearch::get_maximum_iterations_number() const
{
   return(maximum_iterations_number);
}


// const double& get_maximum_time() const method

/// Returns the maximum training time.  

const double& RandomSearch::get_maximum_time() const
{
   return(maximum_time);
}


// const bool& get_reserve_parameters_history() const method

/// Returns true if the parameters history matrix is to be reserved, and false otherwise.

const bool& RandomSearch::get_reserve_parameters_history() const
{
   return(reserve_parameters_history);     
}


// const bool& get_reserve_parameters_norm_history() const method 

/// Returns true if the parameters norm history vector is to be reserved, and false otherwise.

const bool& RandomSearch::get_reserve_parameters_norm_history() const
{
   return(reserve_parameters_norm_history);     
}


// const bool& get_reserve_loss_history() const method

/// Returns true if the loss history vector is to be reserved, and false otherwise.

const bool& RandomSearch::get_reserve_loss_history() const
{
   return(reserve_loss_history);     
}


// const bool& get_reserve_training_direction_history() const method

/// Returns true if the training direction history matrix is to be reserved, and false otherwise.

const bool& RandomSearch::get_reserve_training_direction_history() const
{
   return(reserve_training_direction_history);     
}


// const bool& get_reserve_training_rate_history() const method

/// Returns true if the training rate history vector is to be reserved, and false otherwise.

const bool& RandomSearch::get_reserve_training_rate_history() const
{
   return(reserve_training_rate_history);     
}


// const bool& get_reserve_elapsed_time_history() const method

/// Returns true if the elapsed time history vector is to be reserved, and false otherwise.

const bool& RandomSearch::get_reserve_elapsed_time_history() const
{
   return(reserve_elapsed_time_history);     
}


// const bool& get_reserve_selection_error_history() const method

/// Returns true if the selection loss history vector is to be reserved, and false otherwise.

const bool& RandomSearch::get_reserve_selection_error_history() const
{
   return(reserve_selection_error_history);
}


// const double& get_training_rate_reduction_factor() const method

/// Returns the reducing factor for the training rate. 

const double& RandomSearch::get_training_rate_reduction_factor() const
{
   return(training_rate_reduction_factor);
}


// const size_t& get_training_rate_reduction_period() const method

/// Returns the reducing period for the training rate. 

const size_t& RandomSearch::get_training_rate_reduction_period() const
{
   return(training_rate_reduction_period);
}


// void set_default() method

/// Sets all the random search object members to their default values:
/// <ul>
/// <li> Training rate reduction factor: 0.9
/// <li> Training rate reduction period: 10
/// <li> Warning parameters norm: 1.0e6
/// <li> Error parameters norm: 1.0e9
/// <li> Performance goal: -numeric_limits<double>::max()
/// <li> Maximum time: 1.0e6
/// <li> Maximum iterations number: 100
/// <li> Reserve potential parameters history: False
/// <li> Reserve potential parameters norm history: False
/// <li> Reserve loss history: False.
/// <li> Display: True
/// <li> Display period: 10
/// </ul>

void RandomSearch::set_default()
{   
   // TRAINING PARAMETERS

   first_training_rate = 0.01;

   training_rate_reduction_factor = 0.9;
   training_rate_reduction_period = 10;

   // STOPPING CRITERIA

   loss_goal = -numeric_limits<double>::max();

   maximum_iterations_number = 100;
   maximum_time = 1000.0;

   // TRAINING HISTORY

   reserve_parameters_history = false;
   reserve_parameters_norm_history = false;

   reserve_loss_history = true;

   reserve_training_direction_history = false;
   reserve_training_rate_history = false;
   reserve_elapsed_time_history = false;

   // UTILITIES

   warning_parameters_norm = 1.0e6;
   warning_training_rate = 1.0e6;

   error_parameters_norm = 1.0e9;
   error_training_rate = 1.0e9;

   display = true;
   display_period = 5;
}


// void set_first_training_rate(const double&)

/// Sets the initial training rate in the random search.
/// The training rate is the step given in some training direction.
/// @param new_first_training_rate Firs training rate value.

void RandomSearch::set_first_training_rate(const double& new_first_training_rate)
{
    first_training_rate = new_first_training_rate;
}


// void set_training_rate_reduction_factor(const double&) method

/// Sets a new value for the reduction factor of the training rate. 
/// @param new_training_rate_reduction_factor Reduction factor value.

void RandomSearch::set_training_rate_reduction_factor(const double& new_training_rate_reduction_factor)
{
   training_rate_reduction_factor = new_training_rate_reduction_factor;
}


// void set_training_rate_reduction_period(size_t) method

/// Sets a new period value for the reduction of the training rate. This is measured in training iterations. 
/// @param new_training_rate_reduction_period Reduction period for the training rate.

void RandomSearch::set_training_rate_reduction_period(const size_t& new_training_rate_reduction_period)
{
   training_rate_reduction_period = new_training_rate_reduction_period;
}


// void set_reserve_parameters_history(bool) method

/// Makes the potential parameters history vector of vectors to be reseved or not in memory.
/// @param new_reserve_parameters_history True if the potential parameters history is to be reserved, false otherwise.

void RandomSearch::set_reserve_parameters_history(const bool& new_reserve_parameters_history)
{
   reserve_parameters_history = new_reserve_parameters_history;
}


// void set_reserve_parameters_norm_history(bool) method

/// Makes the potential parameters norm history vector to be reseved or not in memory.
/// @param new_reserve_parameters_norm_history True if the potential parameters norm history is to be reserved, 
/// false otherwise.

void RandomSearch::set_reserve_parameters_norm_history(const bool& new_reserve_parameters_norm_history)
{
   reserve_parameters_norm_history = new_reserve_parameters_norm_history;   
}


// void set_reserve_loss_history(bool) method

/// Makes the potential loss history vector to be reseved or not in memory.
/// @param new_reserve_loss_history True if the potential loss history is to be reserved, 
/// false otherwise.

void RandomSearch::set_reserve_loss_history(const bool& new_reserve_loss_history)
{
   reserve_loss_history = new_reserve_loss_history;
}


// void set_reserve_all_training_history(bool) method

/// Makes the training history of all variables to reseved or not in memory.
/// @param new_reserve_all_training_history True if the training history of all variables is to be reserved, 
/// false otherwise.

void RandomSearch::set_reserve_all_training_history(const bool& new_reserve_all_training_history)
{
   // Neural network

   reserve_parameters_history = new_reserve_all_training_history;
   reserve_parameters_norm_history = new_reserve_all_training_history;
   
   // Loss index

   reserve_loss_history = new_reserve_all_training_history;
  
   // Training algorithm

   reserve_elapsed_time_history = new_reserve_all_training_history;
}


// void set_warning_parameters_norm(const double&) method

/// Sets a new value for the parameters vector norm at which a warning message is written to the 
/// screen. 
/// @param new_warning_parameters_norm Warning norm of parameters vector value. 

void RandomSearch::set_warning_parameters_norm(const double& new_warning_parameters_norm)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_warning_parameters_norm < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: RandomSearch class.\n"
             << "void set_warning_parameters_norm(const double&) method.\n"
             << "Warning parameters norm must be equal or greater than 0.\n";

      throw logic_error(buffer.str());	  
   }

   #endif

   // Set warning parameters norm

   warning_parameters_norm = new_warning_parameters_norm;     
}



// void set_warning_training_rate(const double&) method

/// Sets a new training rate value at wich a warning message is written to the screen during line 
/// minimization.
/// @param new_warning_training_rate Warning training rate value.

void RandomSearch::set_warning_training_rate(const double& new_warning_training_rate)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_warning_training_rate < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: RandomSearch class.\n" 
             << "void set_warning_training_rate(const double&) method.\n"
             << "Warning training rate must be equal or greater than 0.\n";

      throw logic_error(buffer.str());	  
   }

   #endif

   warning_training_rate = new_warning_training_rate;
}


// void set_error_parameters_norm(const double&) method

/// Sets a new value for the parameters vector norm at which an error message is written to the 
/// screen and the program exits. 
/// @param new_error_parameters_norm Error norm of parameters vector value. 

void RandomSearch::set_error_parameters_norm(const double& new_error_parameters_norm)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_error_parameters_norm < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: RandomSearch class.\n"
             << "void set_error_parameters_norm(const double&) method.\n"
             << "Error parameters norm must be equal or greater than 0.\n";

      throw logic_error(buffer.str());	  
   }

   #endif

   // Set error parameters norm

   error_parameters_norm = new_error_parameters_norm;
}


// void set_error_training_rate(const double&) method

/// Sets a new training rate value at wich a the line minimization algorithm is assumed to fail when 
/// bracketing a minimum.
/// @param new_error_training_rate Error training rate value.

void RandomSearch::set_error_training_rate(const double& new_error_training_rate)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_error_training_rate < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: RandomSearch class.\n"
             << "void set_error_training_rate(const double&) method.\n"
             << "Error training rate must be equal or greater than 0.\n";

      throw logic_error(buffer.str());	  
   }

   #endif

   // Set error training rate

   error_training_rate = new_error_training_rate;
}


// void set_loss_goal(const doubleT) method

/// Sets a new goal value for the loss. 
/// This is used as a stopping criterion when training a multilayer perceptron
/// @param new_loss_goal Goal value for the loss.

void RandomSearch::set_loss_goal(const double& new_loss_goal)
{
   loss_goal = new_loss_goal;
}


// void set_maximum_selection_error_increases(const size_t&) method

/// Sets a new maximum number of selection failures. 
/// @param new_maximum_selection_loss_decreases Maximum number of iterations in which the selection evalutation decreases.

void RandomSearch::set_maximum_selection_error_increases(const size_t& new_maximum_selection_loss_decreases)
{
   // Set maximum selection performace decrases

   maximum_selection_loss_decreases = new_maximum_selection_loss_decreases;
}


// void set_maximum_iterations_number(const size_t&) method

/// Sets a maximum number of iterations for training.
/// @param new_maximum_iterations_number Maximum number of iterations for training.

void RandomSearch::set_maximum_iterations_number(const size_t& new_maximum_iterations_number)
{
   maximum_iterations_number = new_maximum_iterations_number;
}


// void set_maximum_time(const double&) method

/// Sets a new maximum training time.  
/// @param new_maximum_time Maximum training time.

void RandomSearch::set_maximum_time(const double& new_maximum_time)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_maximum_time < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: RandomSearch class.\n"
             << "void set_maximum_time(const double&) method.\n"
             << "Maximum time must be equal or greater than 0.\n";

      throw logic_error(buffer.str());	  
   }
   
   #endif

   // Set maximum time

   maximum_time = new_maximum_time;
}


// void set_reserve_training_direction_history(const bool&) method

/// Makes the training direction history vector of vectors to be reseved or not in memory.
/// @param new_reserve_training_direction_history True if the training direction history matrix is to be reserved, 
/// false otherwise.

void RandomSearch::set_reserve_training_direction_history(const bool& new_reserve_training_direction_history)
{
   reserve_training_direction_history = new_reserve_training_direction_history;          
}


// void set_reserve_training_direction_norm_history(const bool&) method

/// Makes the training direction norm history vector to be reseved or not in memory.
/// @param new_reserve_training_direction_norm_history True if the history of the norm of the training direction is to be reserved,
/// false otherwise.

void RandomSearch::set_reserve_training_direction_norm_history(const bool& new_reserve_training_direction_norm_history)
{
   reserve_training_direction_norm_history = new_reserve_training_direction_norm_history;
}


// void set_reserve_training_rate_history(bool) method

/// Makes the training rate history vector to be reseved or not in memory.
/// @param new_reserve_training_rate_history True if the training rate history vector is to be reserved, false 
/// otherwise.

void RandomSearch::set_reserve_training_rate_history(const bool& new_reserve_training_rate_history)
{
   reserve_training_rate_history = new_reserve_training_rate_history;          
}


// void set_reserve_elapsed_time_history(bool) method

/// Makes the elapsed time over the iterations to be reseved or not in memory. This is a vector.
/// @param new_reserve_elapsed_time_history True if the elapsed time history vector is to be reserved, false 
/// otherwise.

void RandomSearch::set_reserve_elapsed_time_history(const bool& new_reserve_elapsed_time_history)
{
   reserve_elapsed_time_history = new_reserve_elapsed_time_history;     
}


// void set_reserve_selection_error_history(bool) method

/// Makes the selection loss history to be reserved or not in memory.
/// This is a vector. 
/// @param new_reserve_selection_error_history True if the selection loss history is to be reserved, false otherwise.

void RandomSearch::set_reserve_selection_error_history(const bool& new_reserve_selection_error_history)  
{
   reserve_selection_error_history = new_reserve_selection_error_history;
}


// void set_display_period(size_t) method

/// Sets a new number of iterations between the training showing progress. 
/// @param new_display_period
/// Number of iterations between the training showing progress. 

void RandomSearch::set_display_period(const size_t& new_display_period)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 
     
   if(new_display_period <= 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: RandomSearch class.\n"
             << "void set_display_period(const double&) method.\n"
             << "First training rate must be greater than 0.\n";

      throw logic_error(buffer.str());	  
   }

   #endif

   display_period = new_display_period;
}



// Vector<double> calculate_training_direction() const method

/// Calculates a random vector to be used as training direction.

Vector<double> RandomSearch::calculate_training_direction() const
{   
   const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

   const size_t parameters_number = neural_network_pointer->get_parameters_number();

   Vector<double> random(parameters_number);
   double random_norm;

   do
   { 
      random.randomize_uniform();   
      random_norm = random.calculate_L2_norm();
   }while(random_norm == 0.0);

   return(random/random_norm);
}


// void resize_training_history(const size_t&) method

/// Resizes all the training history variables. 
/// @param new_size Size of training history variables. 

void RandomSearch::RandomSearchResults::resize_training_history(const size_t& new_size)
{

    if(random_search_pointer->get_reserve_parameters_history())
    {
        parameters_history.resize(new_size);
    }

    if(random_search_pointer->get_reserve_parameters_norm_history())
    {
        parameters_norm_history.resize(new_size);
    }

    if(random_search_pointer->get_reserve_loss_history())
    {
        loss_history.resize(new_size);
    }

    if(random_search_pointer->get_reserve_selection_error_history())
    {
        selection_error_history.resize(new_size);
    }

    if(random_search_pointer->get_reserve_training_direction_history())
    {
        training_direction_history.resize(new_size);
    }

    if(random_search_pointer->get_reserve_training_rate_history())
    {
        training_rate_history.resize(new_size);
    }

    if(random_search_pointer->get_reserve_elapsed_time_history())
    {
        elapsed_time_history.resize(new_size);
    }
}


// string object_to_string() const method

/// Returns a string representation of the current random search results structure. 

string RandomSearch::RandomSearchResults::object_to_string() const
{
   ostringstream buffer;

   // Parameters history

   if(!parameters_history.empty())
   {
     buffer << "% Parameters history:\n"
            << parameters_history << "\n";
   }

   // Parameters norm history

   if(!parameters_norm_history.empty())
   {
       buffer << "% Parameters norm history:\n"
              << parameters_norm_history << "\n"; 
   }
   
   // loss history   

   if(!loss_history.empty())
   {
       buffer << "% loss history:\n"
              << loss_history << "\n"; 
   }

   // Selection loss history

   if(!selection_error_history.empty())
   {
       buffer << "% Selection loss history:\n"
              << selection_error_history << "\n"; 
   }

   // Training direction history

   if(!training_direction_history.empty())
   {
      if(!training_direction_history[0].empty())
      {
          buffer << "% Training direction history:\n"
                 << training_direction_history << "\n"; 
	  }
   }

   // Training rate history

   if(!training_rate_history.empty())
   {
       buffer << "% Training rate history:\n"
              << training_rate_history << "\n"; 
   }

   // Elapsed time history

   if(!elapsed_time_history.empty())
   {
       buffer << "% Elapsed time history:\n"
              << elapsed_time_history << "\n"; 
   }

   return(buffer.str());
}


Matrix<string> RandomSearch::RandomSearchResults::write_final_results(const int& precision) const
{
   ostringstream buffer;

   Vector<string> names;
   Vector<string> values;

   // Final parameters norm

   names.push_back("Final parameters norm");

   buffer.str("");
   buffer << setprecision(precision) << final_parameters_norm;

   values.push_back(buffer.str());

   // Final loss

   names.push_back("Final training error");

   buffer.str("");
   buffer << setprecision(precision) << final_loss;

   values.push_back(buffer.str());

   // Final selection loss

   const LossIndex* loss_index_pointer = random_search_pointer->get_loss_index_pointer();

/*
   if(loss_index_pointer->has_selection())
   {
       names.push_back("Final selection error");

       buffer.str("");
       buffer << setprecision(precision) << final_selection_error;

       values.push_back(buffer.str());
    }
*/
   // Final training rate

//   names.push_back("Final training rate");

//   buffer.str("");
//   buffer << setprecision(precision) << final_training_rate;

//   values.push_back(buffer.str());

   // Iterations number

   names.push_back("Iterations number");

   buffer.str("");
   buffer << iterations_number;

   values.push_back(buffer.str());

   // Elapsed time

   names.push_back("Elapsed time");

   buffer.str("");
   buffer << write_elapsed_time(elapsed_time);

   values.push_back(buffer.str());

   const size_t rows_number = names.size();
   const size_t columns_number = 2;

   Matrix<string> final_results(rows_number, columns_number);

   final_results.set_column(0, names, "name");
   final_results.set_column(1, values, "value");

   return(final_results);
}


// RandomSearchResults* perform_training() method

/// Trains a neural network with an associated loss index according to the random search training algorithm.
/// Training occurs according to the training parameters. 

RandomSearch::RandomSearchResults* RandomSearch::perform_training()
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   check();

   #endif

   RandomSearchResults* results_pointer = new RandomSearchResults(this);
   results_pointer->resize_training_history(1+maximum_iterations_number);

   // Start training 

   if(display)
   {
      cout << "Training with random search...\n";
   }
   

   // Elapsed time

   time_t beginning_time, current_time;
   time(&beginning_time);
   double elapsed_time;

   // Data set stuff

   DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

   const Instances& instances = data_set_pointer->get_instances();

   const Vector<size_t> training_indices = instances.get_training_indices();
   const Vector<size_t> selection_indices = instances.get_selection_indices();

   // Neural network stuff

   NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

   const size_t parameters_number = neural_network_pointer->get_parameters_number();

   Vector<double> parameters = neural_network_pointer->get_parameters();
   double parameters_norm;

   // Loss index stuff
    
   double training_loss = 0.0;
   double potential_training_loss = numeric_limits<double>::max();

   double selection_loss = 0.0;
   double old_selection_loss = 0.0;

   size_t selection_failures = 0;

   // Training algorithm stuff 

   Vector<double> training_direction(parameters_number);
   double training_rate = 1.0;

   Vector<double> potential_parameters(parameters);
   double potential_parameters_norm;

   Vector<double> parameters_increment(parameters_number);
//   double parameters_increment_norm;

   bool stop_training = false;

   // Main loop 

   for(size_t iteration = 0; iteration <= maximum_iterations_number; iteration++)
   {             
       // Neural network stuff

       parameters_norm = parameters.calculate_L2_norm();

       if(display && parameters_norm >= warning_parameters_norm)
       {
          cout << "OpenNN Warning: Parameters norm is " << parameters_norm << ".\n";
       }

      // Loss index stuff

       if(iteration == 0)
       {
           training_loss = loss_index_pointer->calculate_loss(training_indices);
           selection_loss = loss_index_pointer->calculate_loss(selection_indices);
       }


      if(iteration != 0 && selection_loss > old_selection_loss)
      {
         selection_failures++;
      }

      potential_training_loss = loss_index_pointer->calculate_loss(training_indices, potential_parameters);

      // Training algorithm stuff

      training_direction = calculate_training_direction();
  
      if(iteration != 0 && iteration%training_rate_reduction_period == 0)
      {
         training_rate *= training_rate_reduction_factor; 
      }       

      parameters_increment = training_direction*training_rate;
//      parameters_increment_norm = parameters_increment.calculate_norm();

      potential_parameters = parameters + parameters_increment;
      potential_parameters_norm = potential_parameters.calculate_L2_norm();

      time(&current_time);
      elapsed_time = difftime(current_time, beginning_time);

      // Training history neural network

      if(reserve_parameters_history)
      {
         results_pointer->parameters_history[iteration] = parameters;
      }

      if(reserve_parameters_norm_history)
      {
         results_pointer->parameters_norm_history[iteration] = parameters_norm;
      }       
	  
      // Training history loss

      if(reserve_loss_history)
      {
         results_pointer->loss_history[iteration] = training_loss;
      }

      if(reserve_selection_error_history)
      {
          results_pointer->selection_error_history[iteration] = selection_loss;
      }

      // Training history training algorithm

      if(reserve_training_direction_history)
      {
         results_pointer->training_direction_history[iteration] = training_direction;
      }

      if(reserve_training_rate_history)
      {
         results_pointer->training_rate_history[iteration] = training_rate;
      }

//      if(reserve_potential_parameters_history)
//      {
//         results_pointer->potential_parameters_history[iteration] = potential_parameters;
//      }

//      if(reserve_potential_parameters_norm_history)
//      {
//         results_pointer->potential_parameters_norm_history[iteration] = potential_parameters_norm;
//      }

      if(reserve_elapsed_time_history)
      {
         results_pointer->elapsed_time_history[iteration] = elapsed_time;
      }

      // Stopping Criteria

      if(training_loss <= loss_goal)
      {
         if(display)
         {
            cout << "Iteration " << iteration << ": Loss goal reached.\n";
         }

         stop_training = true;
      }

      else if(iteration == maximum_iterations_number)
      {
         if(display)
         {
            cout << "Iteration " << iteration << ": Maximum number of iterations reached.\n";
         }

         stop_training = true;
      }

      else if(elapsed_time >= maximum_time)
      {
         if(display)
         {
            cout << "Iteration " << iteration << ": Maximum training time reached.\n";
         }

         stop_training = true;
      }
      
      if(iteration != 0 && iteration % save_period == 0)
      {
            neural_network_pointer->save(neural_network_file_name);
      }

      if(stop_training)
      {
          if(display)
          {
             cout << "Parameters norm: " << parameters_norm << "\n"
                       << "Potential parameters norm: " << potential_parameters_norm << "\n"
                       << "Training loss: " << training_loss << "\n"
                       << loss_index_pointer->write_information()
                       << "Potential training loss: " << potential_training_loss << "\n"
                       << "Training rate: " << training_rate << "\n"
                       << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;

             if(selection_loss != 0)
             {
                cout << "Selection loss: " << selection_loss << endl;
             }
          }

         results_pointer->final_parameters = parameters;
         results_pointer->final_parameters_norm = parameters_norm;

         results_pointer->final_loss = training_loss;
         results_pointer->final_selection_error = selection_loss;
  
         results_pointer->final_training_direction = training_direction;
         results_pointer->final_training_rate = training_rate;
         results_pointer->elapsed_time = elapsed_time;

         results_pointer->iterations_number = iteration;

         break;
      }

      else if(display && iteration % display_period == 0)
      {
         cout << "Iteration " << iteration << ";\n"
                   << "Parameters norm: " << parameters_norm << "\n"
                   << "Potential parameters norm: " << potential_parameters_norm << "\n"
                   << "Training loss: " << training_loss << "\n"
                   << loss_index_pointer->write_information()
                   << "Potential loss: " << potential_training_loss << "\n"
                   << "Training rate: " << training_rate << "\n"
                   << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl; 

         if(selection_loss != 0)
         {
            cout << "Selection loss: " << selection_loss << endl;
         }
      }

      // Set new parameters

      if(potential_training_loss < training_loss)
      {
         parameters = potential_parameters;

         neural_network_pointer->set_parameters(parameters);

         training_loss = potential_training_loss;

         selection_loss = loss_index_pointer->calculate_loss(selection_indices);
         old_selection_loss = selection_loss;
      }
   }

   return(results_pointer);
}


// string write_training_algorithm_type() const method

string RandomSearch::write_training_algorithm_type() const
{
   return("RANDOM_SEARCH");
}


// Matrix<string> to_string_matrix() const method

/// Writes as matrix of strings the most representative atributes.

Matrix<string> RandomSearch::to_string_matrix() const
{
    ostringstream buffer;

    Vector<string> labels;
    Vector<string> values;

   // Performance goal

   labels.push_back("Performance goal");

   buffer.str("");
   buffer << loss_goal;

   values.push_back(buffer.str());

   // Maximum selection failures

   labels.push_back("Maximum selection loss increases");

   buffer.str("");
   buffer << maximum_selection_loss_decreases;

   values.push_back(buffer.str());

   // Maximum iterations number

   labels.push_back("Maximum iterations number");

   buffer.str("");
   buffer << maximum_iterations_number;

   values.push_back(buffer.str());

   // Maximum time

   labels.push_back("Maximum time");

   buffer.str("");
   buffer << maximum_time;

   values.push_back(buffer.str());

   // Reserve parameters norm history

   labels.push_back("Reserve parameters norm history");

   buffer.str("");
   buffer << reserve_parameters_norm_history;

   values.push_back(buffer.str());

   // Reserve loss history

   labels.push_back("Reserve loss history");

   buffer.str("");
   buffer << reserve_loss_history;

   values.push_back(buffer.str());

   // Reserve selection loss history

   labels.push_back("Reserve selection loss history");

   buffer.str("");
   buffer << reserve_selection_error_history;

   values.push_back(buffer.str());

   // Reserve training direction norm history

//   labels.push_back("");

//   buffer.str("");
//   buffer << reserve_training_direction_norm_history;

   // Reserve training rate history

//   labels.push_back("");

//   buffer.str("");
//   buffer << reserve_training_rate_history;

//   values.push_back(buffer.str());

   // Reserve elapsed time history

   labels.push_back("Reserve elapsed time history");

   buffer.str("");
   buffer << reserve_elapsed_time_history;

   values.push_back(buffer.str());

   const size_t rows_number = labels.size();
   const size_t columns_number = 2;

   Matrix<string> string_matrix(rows_number, columns_number);

   string_matrix.set_column(0, labels, "");
   string_matrix.set_column(1, values, "");

    return(string_matrix);
}


// tinyxml2::XMLDocument* to_XML() const method

/// Prints to the screen the training parameters, the stopping criteria
/// and other user stuff concerning the random search object.

tinyxml2::XMLDocument* RandomSearch::to_XML() const 
{
   ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Training algorithm

   tinyxml2::XMLElement* root_element = document->NewElement("RandomSearch");

   document->InsertFirstChild(root_element);

   tinyxml2::XMLElement* element = nullptr;
   tinyxml2::XMLText* text = nullptr;

   // Training rate reduction factor
   {
   element = document->NewElement("TrainingRateReductionFactor");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << training_rate_reduction_factor;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Training rate reduction period
   {
   element = document->NewElement("TrainingRateReductionPeriod");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << training_rate_reduction_period;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // First training rate
   {
   element = document->NewElement("FirstTrainingRate");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << first_training_rate;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Warning parameters norm
   {
   element = document->NewElement("WarningParametersNorm");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << warning_parameters_norm;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Warning training rate 
   {
   element = document->NewElement("WarningTrainingRate");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << warning_training_rate;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Error parameters norm
   {
   element = document->NewElement("ErrorParametersNorm");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << error_parameters_norm;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Error training rate
   {
   element = document->NewElement("ErrorTrainingRate");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << error_training_rate;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Performance goal 
   {
   element = document->NewElement("LossGoal");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << loss_goal;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Maximum selection loss decreases
   {
   element = document->NewElement("MaximumSelectionLossDecreases");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << maximum_selection_loss_decreases;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Maximum iterations number 
   {
   element = document->NewElement("MaximumIterationsNumber");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << maximum_iterations_number;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Maximum time 
   {
   element = document->NewElement("MaximumTime");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << maximum_time;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Reserve parameters history 
   {
   element = document->NewElement("ReserveParametersHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_parameters_history;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Reserve parameters norm history 
   {
   element = document->NewElement("ReserveParametersNormHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_parameters_norm_history;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Reserve loss history 
   {
   element = document->NewElement("ReservePerformanceHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_loss_history;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Reserve selection loss history
   {
   element = document->NewElement("ReserveSelectionLossHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_selection_error_history;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Reserve training direction history 
   {
   element = document->NewElement("ReserveTrainingDirectionHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_training_direction_history;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Reserve training rate history 
   {
   element = document->NewElement("ReserveTrainingRateHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_training_rate_history;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Reserve elapsed time history 
   {
   element = document->NewElement("ReserveElapsedTimeHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_elapsed_time_history;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Reserve selection loss history
   {
   element = document->NewElement("ReserveSelectionLossHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_selection_error_history;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Display period
   {
   element = document->NewElement("DisplayPeriod");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << display_period;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Display
//   {
//   element = document->NewElement("Display");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << display;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);
//   }

   return(document);
}


//void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the random search object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void RandomSearch::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("RandomSearch");

    // Training rate reduction factor

    file_stream.OpenElement("TrainingRateReductionFactor");

    buffer.str("");
    buffer << training_rate_reduction_factor;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Training rate reduction period

    file_stream.OpenElement("TrainingRateReductionPeriod");

    buffer.str("");
    buffer << training_rate_reduction_period;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // First training rate

    file_stream.OpenElement("FirstTrainingRate");

    buffer.str("");
    buffer << first_training_rate;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Warning parameters norm

    file_stream.OpenElement("WarningParametersNorm");

    buffer.str("");
    buffer << warning_parameters_norm;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Warning training rate

    file_stream.OpenElement("WarningTrainingRate");

    buffer.str("");
    buffer << warning_training_rate;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Error parameters norm

    file_stream.OpenElement("ErrorParametersNorm");

    buffer.str("");
    buffer << error_parameters_norm;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Error training rate

    file_stream.OpenElement("ErrorTrainingRate");

    buffer.str("");
    buffer << error_training_rate;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Performance goal

    file_stream.OpenElement("LossGoal");

    buffer.str("");
    buffer << loss_goal;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum selection loss decreases

    file_stream.OpenElement("MaximumSelectionLossDecreases");

    buffer.str("");
    buffer << maximum_selection_loss_decreases;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum iterations number

    file_stream.OpenElement("MaximumIterationsNumber");

    buffer.str("");
    buffer << maximum_iterations_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum time

    file_stream.OpenElement("MaximumTime");

    buffer.str("");
    buffer << maximum_time;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve parameters history

    file_stream.OpenElement("ReserveParametersHistory");

    buffer.str("");
    buffer << reserve_parameters_history;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve parameters norm history

    file_stream.OpenElement("ReserveParametersNormHistory");

    buffer.str("");
    buffer << reserve_parameters_norm_history;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve loss history

    file_stream.OpenElement("ReservePerformanceHistory");

    buffer.str("");
    buffer << reserve_loss_history;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve selection loss history

    file_stream.OpenElement("ReserveSelectionLossHistory");

    buffer.str("");
    buffer << reserve_selection_error_history;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve training direction history

    file_stream.OpenElement("ReserveTrainingDirectionHistory");

    buffer.str("");
    buffer << reserve_training_direction_history;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve training rate history

    file_stream.OpenElement("ReserveTrainingRateHistory");

    buffer.str("");
    buffer << reserve_training_rate_history;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve elapsed time history

    file_stream.OpenElement("ReserveElapsedTimeHistory");

    buffer.str("");
    buffer << reserve_elapsed_time_history;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve selection loss history

    file_stream.OpenElement("ReserveSelectionLossHistory");

    buffer.str("");
    buffer << reserve_selection_error_history;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Display period

    file_stream.OpenElement("DisplayPeriod");

    buffer.str("");
    buffer << display_period;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();


    file_stream.CloseElement();
}


// void from_XML(const tinyxml2::XMLDocument&) method

void RandomSearch::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("RandomSearch");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: RandomSearch class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Random search element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // First training rate
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("FirstTrainingRate");

        if(element)
        {
           const double new_first_training_rate = atof(element->GetText());

           try
           {
              set_first_training_rate(new_first_training_rate);
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

    // Training rate reduction factor
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("TrainingRateReductionFactor");

        if(element)
        {
           const double new_training_rate_reduction_factor = atof(element->GetText());

           try
           {
              set_training_rate_reduction_factor(new_training_rate_reduction_factor);
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

    // Training rate reduction period
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("TrainingRateReductionPeriod");

        if(element)
        {
           const size_t new_training_rate_reduction_period = atoi(element->GetText());

           try
           {
              set_training_rate_reduction_period(new_training_rate_reduction_period);
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

   // Warning parameters norm
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("WarningParametersNorm");

       if(element)
       {
          const double new_warning_parameters_norm = atof(element->GetText());

          try
          {
             set_warning_parameters_norm(new_warning_parameters_norm);
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Warning training rate
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("WarningTrainingRate");

       if(element)
       {
          const double new_warning_training_rate = atof(element->GetText());

          try
          {
             set_warning_training_rate(new_warning_training_rate);
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Error parameters norm
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ErrorParametersNorm");

       if(element)
       {
          const double new_error_parameters_norm = atof(element->GetText());

          try
          {
              set_error_parameters_norm(new_error_parameters_norm);
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Error training rate
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ErrorTrainingRate");

       if(element)
       {
          const double new_error_training_rate = atof(element->GetText());

          try
          {
             set_error_training_rate(new_error_training_rate);
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Performance goal
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("LossGoal");

       if(element)
       {
          const double new_loss_goal = atof(element->GetText());

          try
          {
             set_loss_goal(new_loss_goal);
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Maximum selection loss decreases
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumSelectionLossDecreases");

       if(element)
       {
          const size_t new_maximum_selection_loss_decreases = atoi(element->GetText());

          try
          {
             set_maximum_selection_error_increases(new_maximum_selection_loss_decreases);
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Maximum iterations number
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumIterationsNumber");

       if(element)
       {
          const size_t new_maximum_iterations_number = atoi(element->GetText());

          try
          {
             set_maximum_iterations_number(new_maximum_iterations_number);
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Maximum time
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumTime");

       if(element)
       {
          const double new_maximum_time = atof(element->GetText());

          try
          {
             set_maximum_time(new_maximum_time);
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Reserve parameters history
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveParametersHistory");

       if(element)
       {
          const string new_reserve_parameters_history = element->GetText();

          try
          {
             set_reserve_parameters_history(new_reserve_parameters_history != "0");
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Reserve parameters norm history
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveParametersNormHistory");

       if(element)
       {
          const string new_reserve_parameters_norm_history = element->GetText();

          try
          {
             set_reserve_parameters_norm_history(new_reserve_parameters_norm_history != "0");
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Reserve loss history
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReservePerformanceHistory");

       if(element)
       {
          const string new_reserve_loss_history = element->GetText();

          try
          {
             set_reserve_loss_history(new_reserve_loss_history != "0");
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

    // Reserve selection loss history
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveSelectionLossHistory");

        if(element)
        {
           const string new_reserve_selection_error_history = element->GetText();

           try
           {
              set_reserve_selection_error_history(new_reserve_selection_error_history != "0");
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

   // Reserve training direction history
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveTrainingDirectionHistory");

       if(element)
       {
          const string new_reserve_training_direction_history = element->GetText();

          try
          {
             set_reserve_training_direction_history(new_reserve_training_direction_history != "0");
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

    // Reserve training direction norm history
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveTrainingDirectionNormHistory");

        if(element)
        {
           const string new_reserve_training_direction_norm_history = element->GetText();

           try
           {
              set_reserve_training_direction_norm_history(new_reserve_training_direction_norm_history != "0");
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

   // Reserve training rate history
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveTrainingRateHistory");

       if(element)
       {
          const string new_reserve_training_rate_history = element->GetText();

          try
          {
             set_reserve_training_rate_history(new_reserve_training_rate_history != "0");
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Reserve elapsed time history
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveElapsedTimeHistory");

       if(element)
       {
          const string new_reserve_elapsed_time_history = element->GetText();

          try
          {
             set_reserve_elapsed_time_history(new_reserve_elapsed_time_history != "0");
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Display period
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("DisplayPeriod");

       if(element)
       {
          const size_t new_display_period = atoi(element->GetText());

          try
          {
             set_display_period(new_display_period);
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Display
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("Display");

       if(element)
       {
          const string new_display = element->GetText();

          try
          {
             set_display(new_display != "0");
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
