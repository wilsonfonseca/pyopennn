/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   L O S S   I N D E X   C L A S S   H E A D E R                                                              */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __LOSSINDEX_H__
#define __LOSSINDEX_H__

// System includes

#include <string>
#include <sstream>
#include <iostream>
#include <cmath>

// OpenNN includes

#include "vector.h"
#include "matrix.h"

#include "data_set.h"

#include "neural_network.h"

// TinyXml includes

#include "tinyxml2.h"

namespace OpenNN
{
/// This class represents the concept of error term. 
/// A error term is a summand in the loss functional expression.

class LossIndex
{

public:

   // DEFAULT CONSTRUCTOR

   explicit LossIndex();

   // NEURAL NETWORK CONSTRUCTOR

   explicit LossIndex(NeuralNetwork*);

   // DATA SET CONSTRUCTOR

   explicit LossIndex(DataSet*);

   // NEURAL NETWORK AND DATA SET CONSTRUCTOR

   explicit LossIndex(NeuralNetwork*, DataSet*);

   // XML CONSTRUCTOR

   explicit LossIndex(const tinyxml2::XMLDocument&);

   // COPY CONSTRUCTOR

   LossIndex(const LossIndex&);

   // DESTRUCTOR

   virtual ~LossIndex();

   // ASSIGNMENT OPERATOR

   LossIndex& operator = (const LossIndex&);

   // EQUAL TO OPERATOR

   bool operator == (const LossIndex&) const;

   enum RegularizationMethod{L1, L2, None};

   // STRUCTURES

   /// This structure contains the zero order loss quantities of a error term. 
   /// This only includes the loss itself.

   struct ZerothOrderLoss
   {
      /// Error term evaluation.

      double loss;
   };


   /// This structure contains the first order loss quantities of a error term. 
   /// This includes the loss itself and the gradient vector.

   struct FirstOrderLoss
   {
      /// Error term loss. 

      double loss;

      /// Error term gradient vector. 

      Vector<double> gradient;
   };


   /// This structure contains the second order loss quantities of a error term. 
   /// This includes the loss itself, the gradient vector and the Hessian matrix.

   struct SecondOrderLoss
   {
      /// Peformance term loss. 

      double loss;

      /// Error term gradient vector. 

      Vector<double> gradient;

	  /// Error term Hessian matrix. 

      Matrix<double> Hessian;
   };


   ///
   /// This structure contains the zero order evaluation of the terms function.
   ///

   struct ZerothOrderTerms
   {
      /// Subterms loss vector.

      Vector<double> terms;
   };

   /// Set of subterms vector and subterms Jacobian matrix of the error term. 
   /// A method returning this structure might be more efficient than calculating the error terms and the terms Jacobian separately.

   struct FirstOrderTerms
   {
      /// Subterms loss vector. 

      Vector<double> terms;

      /// Subterms Jacobian matrix. 

      Matrix<double> Jacobian;
   };


   // METHODS

   // Get methods

   /// Returns a pointer to the neural network object associated to the error term.

   inline NeuralNetwork* get_neural_network_pointer() const 
   {
        #ifdef __OPENNN_DEBUG__

        if(!neural_network_pointer)
        {
             ostringstream buffer;

             buffer << "OpenNN Exception: LossIndex class.\n"
                    << "NeuralNetwork* get_neural_network_pointer() const method.\n"
                    << "Neural network pointer is nullptr.\n";

             throw logic_error(buffer.str());
        }

        #endif

      return(neural_network_pointer);
   }

   /// Returns a pointer to the data set object associated to the error term.

   inline DataSet* get_data_set_pointer() const 
   {
        #ifdef __OPENNN_DEBUG__

        if(!data_set_pointer)
        {
             ostringstream buffer;

             buffer << "OpenNN Exception: LossIndex class.\n"
                    << "DataSet* get_data_set_pointer() const method.\n"
                    << "DataSet pointer is nullptr.\n";

             throw logic_error(buffer.str());
        }

        #endif

      return(data_set_pointer);
   }

   const double& get_regularization_weight() const;
   const bool& get_display() const;

   bool has_neural_network() const;
   bool has_data_set() const;

   // Set methods

   void set();
   void set(NeuralNetwork*);
   void set(DataSet*);
   void set(NeuralNetwork*, DataSet*);

   void set(const LossIndex&);

   void set_neural_network_pointer(NeuralNetwork*);

   void set_data_set_pointer(DataSet*);

   void set_default();

   void set_regularization_method(const RegularizationMethod&);
   void set_regularization_weight(const double&);

   void set_display(const bool&);

   // Loss methods

   double calculate_loss() const;
   double calculate_loss(const Vector<size_t>&) const;

   double calculate_loss(const Vector<double>&) const;
   double calculate_loss(const Vector<size_t>&, const Vector<double>&) const;

   double calculate_loss(const Vector<size_t>&, const Vector<double>&, const double&) const;

   double calculate_training_loss(const Vector<double>&, const double&) const;

   double calculate_training_loss() const;

   // Loss gradient methods

   Vector<double> calculate_loss_gradient() const;
   Vector<double> calculate_loss_gradient(const Vector<size_t>&) const;

   Vector<double> calculate_training_loss_gradient() const;

   // Loss Hessian methods

   Matrix<double> calculate_loss_Hessian() const;
   Matrix<double> calculate_loss_Hessian(const Vector<size_t>&) const;

   double calculate_all_instances_error() const;

   double calculate_training_error() const;
   double calculate_selection_error() const;
   double calculate_testing_error() const;   

   double calculate_error(const Vector<size_t>&) const;

   virtual double calculate_error(const Matrix<double>&, const Matrix<double>&) const = 0;

   /// Returns the default loss of a error term for a given set of neural network parameters.

   double calculate_all_instances_error(const Vector<double>&) const;

   double calculate_training_error(const Vector<double>&) const;
   double calculate_selection_error(const Vector<double>&) const;
   double calculate_testing_error(const Vector<double>&) const;

   virtual double calculate_error(const Vector<size_t>&, const Vector<double>&) const;

   Vector<double> calculate_error_outputs(const Matrix<double>&) const;

   /// Returns an loss of the error term for selection purposes.  

   Vector<double> calculate_error_gradient() const;

   Vector<double> calculate_error_gradient(const Vector<size_t>&) const;

   Vector<double> calculate_training_error_gradient() const;

   /// Returns the error term Hessian.

   Matrix<double> calculate_error_Hessian(const Vector<size_t>&) const;

   Matrix<double> calculate_Hessian_one_layer() const;
   Matrix<double> calculate_Hessian_two_layers() const;

   // Terms function

   Vector<double> calculate_error_terms(const Vector<size_t>&) const {throw logic_error("Calculate error terms Exception");}

   Matrix<double> calculate_error_terms_Jacobian(const Vector<size_t>&) const  {throw logic_error("Calculate error terms Jacobian Exception");}

   string write_error_term_type() const;

   string write_information() const;

   // Regularization methods

   double calculate_regularization() const;

   Vector<double> calculate_regularization_gradient() const;
   Matrix<double> calculate_regularization_Hessian() const;

   double calculate_regularization(const Vector<double>&) const;

   Vector<double> calculate_regularization_gradient(const Vector<double>&) const;
   Matrix<double> calculate_regularization_Hessian(const Vector<double>&) const;

   // Serialization methods

   string object_to_string() const;

   tinyxml2::XMLDocument* to_XML() const;
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;

   size_t calculate_Kronecker_delta(const size_t&, const size_t&) const;

   // Layers delta methods

   Vector< Vector<double> > calculate_layers_delta(const Vector< Vector<double> >&, const Vector<double>&) const;
   Vector< Matrix<double> > calculate_layers_delta(const Vector< Matrix<double> >&, const Matrix<double>&) const;



   // Checking methods

   void check() const;

   /// Returns the error term gradient.

   virtual Vector<double> calculate_output_gradient(const Vector<size_t>&, const Vector<double>&, const Vector<double>&) const
   {
       throw logic_error("Calculate output gradient vector Exception");
   }

   virtual Matrix<double> calculate_output_gradient(const Matrix<double>&, const Matrix<double>&) const
   {
       throw logic_error("Calculate output gradient matrix Exception");
   }

   double calculate_point_error_output_layer_combinations(const Vector<double>&) const;
   double calculate_point_error_layer_combinations(const size_t&, const size_t&, const Vector<double>&) const;

   Vector<double> calculate_points_errors_output_layer_combinations(const Matrix<double>&) const;
   Vector<double> calculate_points_errors_layer_combinations(const size_t&, const Matrix<double>&) const;



   // Interlayers Delta methods

   Matrix<double> calculate_output_interlayers_Delta(const Vector<double>&,
                                                     const Vector<double>&,
                                                     const Vector<double>&,
                                                     const Matrix<double>&) const;

   Matrix<double> calculate_interlayers_Delta(const size_t&,
                                              const size_t&,
                                              const Vector<double>&,
                                              const Vector<double>&,
                                              const Vector<double>&,
                                              const Vector<double>&,
                                              const Vector< Vector<double> >&,
                                              const Vector<double>&,
                                              const Matrix<double>&,
                                              const Matrix<double>&,
                                              const Vector< Vector<double> >&) const;

   Matrix< Matrix <double> > calculate_interlayers_Delta(const Vector< Vector<double> >&,
                                                         const Vector< Vector<double> >&,
                                                         const Matrix< Matrix<double> >&,
                                                         const Vector<double>&,
                                                         const Matrix<double>&,
                                                         const Vector< Vector<double> >&) const;

   // Point gradient methods

   Vector<double> calculate_error_gradient(const Matrix<double>&, const Vector< Matrix<double> >&, const Vector< Matrix<double> >&) const;

   Vector<double> calculate_point_gradient(const Vector<double>&, const Vector< Vector<double> >&, const Vector< Vector<double> >&) const;

   Vector<double> calculate_point_gradient(const Vector< Matrix<double> >&, const Vector< Vector<double> >&) const;

   Matrix<double> calculate_point_Hessian(const Vector< Vector<double> >&,
                                          const Vector< Vector< Vector<double> > >&,
                                          const Matrix< Matrix<double> >&,
                                          const Vector< Vector<double> >&,
                                          const Matrix< Matrix<double> >&) const;

   Matrix<double> calculate_single_hidden_layer_point_Hessian(const Vector< Vector<double> >&,
                                                              const Vector< Vector<double> >&,
                                                              const Vector< Vector< Vector<double> > >&,
                                                              const Vector< Vector<double> >&,
                                                              const Matrix<double>&) const;

   virtual Matrix<double> calculate_output_Hessian(const Vector<size_t>&, const Vector<double>&, const Vector<double>&) const {return Matrix<double>();}

protected:
   // MEMBERS

   /// Pointer to a multilayer perceptron object.

   NeuralNetwork* neural_network_pointer;

   /// Pointer to a data set object.

   DataSet* data_set_pointer;

   RegularizationMethod regularization_method;

   double regularization_weight = 0.0;

   size_t batch_size = 16;

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
