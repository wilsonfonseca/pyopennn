/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   T R A I N I N G   S T R A T E G Y   C L A S S   H E A D E R                                                */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __TRAININGSTRATEGY_H__
#define __TRAININGSTRATEGY_H__

// System includes

#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <limits>
#include <cmath>
#include <ctime>

#ifdef __OPENNN_MPI__
#include <mpi.h>
#endif
// OpenNN includes

#include "loss_index.h"

#include "sum_squared_error.h"
#include "mean_squared_error.h"
#include "root_mean_squared_error.h"
#include "normalized_squared_error.h"
#include "minkowski_error.h"
#include "cross_entropy_error.h"
#include "weighted_squared_error.h"

#include "training_algorithm.h"

#include "random_search.h"
#include "evolutionary_algorithm.h"

#include "gradient_descent.h"
#include "conjugate_gradient.h"
#include "quasi_newton_method.h"
#include "levenberg_marquardt_algorithm.h"

// TinyXml includes

#include "tinyxml2.h"

namespace OpenNN
{

/// This class represents the concept of training strategy for a neural network. 
/// A training strategy is composed of three training algorithms:
/// <ul>
/// <li> Initialization training algorithm.
/// <li> Main training algorithm.
/// <li> Refinement trainining algorithm.
/// </ul> 
   
class TrainingStrategy
{

public:

   // DEFAULT CONSTRUCTOR

    explicit TrainingStrategy();

    explicit TrainingStrategy(NeuralNetwork*, DataSet*);

   // XML CONSTRUCTOR

   explicit TrainingStrategy(const tinyxml2::XMLDocument&);

   // FILE CONSTRUCTOR

   explicit TrainingStrategy(const string&);

   // DESTRUCTOR

   virtual ~TrainingStrategy();

   // ENUMERATIONS

    /// Enumeration of available error terms in OpenNN.

    enum LossMethod
    {
       SUM_SQUARED_ERROR,
       MEAN_SQUARED_ERROR,
       ROOT_MEAN_SQUARED_ERROR,
       NORMALIZED_SQUARED_ERROR,
       MINKOWSKI_ERROR,
       WEIGHTED_SQUARED_ERROR,
       CROSS_ENTROPY_ERROR
    };

    /// Enumeration of all the available types of training algorithms.

    enum TrainingMethod
    {
       GRADIENT_DESCENT,
       CONJUGATE_GRADIENT,
       QUASI_NEWTON_METHOD,
       LEVENBERG_MARQUARDT_ALGORITHM
    };

   // STRUCTURES 

   /// This structure stores the results from the training strategy.
   /// They are composed of the initialization, refinement and training algorithms results. 

   struct Results
   {
        /// Default constructor.

        explicit Results();

        /// Destructor.

        virtual ~Results();

        void save(const string&) const;

        /// Pointer to a structure with the results from the random search training algorithm.

        RandomSearch::RandomSearchResults* random_search_results_pointer;

        /// Pointer to a structure with the results from the evolutionary training algorithm.

        EvolutionaryAlgorithm::EvolutionaryAlgorithmResults* evolutionary_algorithm_results_pointer;

        /// Pointer to a structure with the results from the gradient descent training algorithm.

        GradientDescent::GradientDescentResults* gradient_descent_results_pointer;

        /// Pointer to a structure with the results from the conjugate gradient training algorithm.

        ConjugateGradient::ConjugateGradientResults* conjugate_gradient_results_pointer;

        /// Pointer to a structure with the results from the quasi-Newton method training algorithm.

        QuasiNewtonMethod::QuasiNewtonMethodResults* quasi_Newton_method_results_pointer;

        /// Pointer to a structure with the results from the Levenberg-Marquardt training algorithm.

        LevenbergMarquardtAlgorithm::LevenbergMarquardtAlgorithmResults* Levenberg_Marquardt_algorithm_results_pointer;

  };

   // METHODS

   // Initialization methods

   void initialize_random();

   // Get methods

   NeuralNetwork* get_neural_network_pointer() const;
   LossIndex* get_loss_index_pointer() const;

   bool has_loss_index() const;

   RandomSearch* get_random_search_pointer() const;
   EvolutionaryAlgorithm* get_evolutionary_algorithm_pointer() const;

   GradientDescent* get_gradient_descent_pointer() const;
   ConjugateGradient* get_conjugate_gradient_pointer() const;
   QuasiNewtonMethod* get_quasi_Newton_method_pointer() const;
   LevenbergMarquardtAlgorithm* get_Levenberg_Marquardt_algorithm_pointer() const;

   SumSquaredError* get_sum_squared_error_pointer() const;
   MeanSquaredError* get_mean_squared_error_pointer() const;
   RootMeanSquaredError* get_root_mean_squared_error_pointer() const;
   NormalizedSquaredError* get_normalized_squared_error_pointer() const;
   MinkowskiError* get_Minkowski_error_pointer() const;
   CrossEntropyError* get_cross_entropy_error_pointer() const;
   WeightedSquaredError* get_weighted_squared_error_pointer() const;

   const LossMethod& get_loss_method() const;
   const TrainingMethod& get_training_method() const;

   string write_loss_method() const;
   string write_training_method() const;

   string write_training_method_text() const;

   const bool& get_display() const;

   // Set methods

   void set();
   void set_default();

#ifdef __OPENNN_MPI__
   void set_MPI(LossIndex*, const TrainingStrategy*);
#endif

   void set_loss_index_pointer(LossIndex*);

   void set_loss_method(const LossMethod&);
   void set_training_method(const TrainingMethod&);

   void set_loss_method(const string&);
   void set_training_method(const string&);

   void set_display(const bool&);

   // Pointer methods

   void destruct_training_algorithm();

   // Training methods

   // This method trains a neural network which has a loss index associated.

   void initialize_layers_autoencoding();

   Results perform_training() const;

   // Serialization methods

   string object_to_string() const;

   void print() const;

   tinyxml2::XMLDocument* to_XML() const;   
   void from_XML(const tinyxml2::XMLDocument&);   

   void write_XML(tinyxml2::XMLPrinter&) const;
   //void read_XML(   );

   void save(const string&) const;
   void load(const string&);

private:

   DataSet* data_set_pointer = nullptr;

   NeuralNetwork* neural_network_pointer = nullptr;

    // Loss index

    /// Pointer to the sum squared error object wich can be used as the error term.

    SumSquaredError* sum_squared_error_pointer = nullptr;

    /// Pointer to the mean squared error object wich can be used as the error term.

    MeanSquaredError* mean_squared_error_pointer = nullptr;

    /// Pointer to the root mean squared error object wich can be used as the error term.

    RootMeanSquaredError* root_mean_squared_error_pointer = nullptr;

    /// Pointer to the normalized squared error object wich can be used as the error term.

    NormalizedSquaredError* normalized_squared_error_pointer = nullptr;

    /// Pointer to the Mikowski error object wich can be used as the error term.

    MinkowskiError* Minkowski_error_pointer = nullptr;

    /// Pointer to the cross entropy error object wich can be used as the error term.

    CrossEntropyError* cross_entropy_error_pointer = nullptr;

    /// Pointer to the weighted squared error object wich can be used as the error term.

    WeightedSquaredError* weighted_squared_error_pointer = nullptr;

    /// Type of loss method.

    LossMethod loss_method;

    // Training algorithm

    /// Pointer to a random search object to be used for initialization in the training strategy.

    RandomSearch* random_search_pointer = nullptr;

    /// Pointer to a evolutionary training object to be used for initialization in the training strategy.

    EvolutionaryAlgorithm* evolutionary_algorithm_pointer = nullptr;

    /// Pointer to a gradient descent object to be used as a main training algorithm.

    GradientDescent* gradient_descent_pointer = nullptr;

    /// Pointer to a conjugate gradient object to be used as a main training algorithm.

    ConjugateGradient* conjugate_gradient_pointer = nullptr;

    /// Pointer to a quasi-Newton method object to be used as a main training algorithm.

    QuasiNewtonMethod* quasi_Newton_method_pointer = nullptr;

    /// Pointer to a Levenberg-Marquardt algorithm object to be used as a main training algorithm.

    LevenbergMarquardtAlgorithm* Levenberg_Marquardt_algorithm_pointer = nullptr;

    /// Type of main training algorithm.

    TrainingMethod training_method;

    /// Display messages to screen.

    bool display;

};

}

#endif


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

