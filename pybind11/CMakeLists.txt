# CMakeLists.txt -- Build system for the pybind11 modules
#
# Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

cmake_minimum_required(VERSION 2.8.12)

if (POLICY CMP0048)
  # cmake warns if loaded from a min-3.0-required parent dir, so silence the warning:
  cmake_policy(SET CMP0048 NEW)
endif()

# CMake versions < 3.4.0 do not support try_compile/pthread checks without C as active language.
if(CMAKE_VERSION VERSION_LESS 3.4.0)
  project(pybind11)
else()
  project(pybind11 CXX)
endif()

# Check if pybind11 is being used directly or via add_subdirectory
set(PYBIND11_MASTER_PROJECT OFF)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(PYBIND11_MASTER_PROJECT ON)
endif()

option(PYBIND11_INSTALL "Install pybind11 header files?" ${PYBIND11_MASTER_PROJECT})
option(PYBIND11_TEST    "Build pybind11 test suite?"     ${PYBIND11_MASTER_PROJECT})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/tools")

include(pybind11Tools)

# Cache variables so pybind11_add_module can be used in parent projects
set(PYBIND11_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include" CACHE INTERNAL "")
set(PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS} CACHE INTERNAL "")
set(PYTHON_LIBRARIES ${PYTHON_LIBRARIES} CACHE INTERNAL "")
set(PYTHON_MODULE_PREFIX ${PYTHON_MODULE_PREFIX} CACHE INTERNAL "")
set(PYTHON_MODULE_EXTENSION ${PYTHON_MODULE_EXTENSION} CACHE INTERNAL "")

# NB: when adding a header don't forget to also add it to setup.py
set(PYBIND11_HEADERS
  include/pybind11/detail/class.h
  include/pybind11/detail/common.h
  include/pybind11/detail/descr.h
  include/pybind11/detail/init.h
  include/pybind11/detail/internals.h
  include/pybind11/detail/typeid.h
  include/pybind11/attr.h
  include/pybind11/buffer_info.h
  include/pybind11/cast.h
  include/pybind11/chrono.h
  include/pybind11/complex.h
  include/pybind11/options.h
  include/pybind11/eigen.h
  include/pybind11/embed.h
  include/pybind11/eval.h
  include/pybind11/functional.h
  include/pybind11/numpy.h
  include/pybind11/operators.h
  include/pybind11/pybind11.h
  include/pybind11/pytypes.h
  include/pybind11/stl.h
  include/pybind11/stl_bind.h
  include/pybind11/ant_colony_optimization.h
  include/pybind11/association_rules.h
  include/pybind11/bounding_layer.h
  include/pybind11/character_string.h
  include/pybind11/conditions_layer.h
  include/pybind11/conjugate_gradient.h
  include/pybind11/correlation_analysis.h
  include/pybind11/cross_entropy_error.h
  include/pybind11/data_set.h
  include/pybind11/error_term.h
  include/pybind11/evolutionary_algorithm.h
  include/pybind11/f1_score_optimization_threshold.h
  include/pybind11/file_utilities.h
  include/pybind11/final_solutions_error.h
  include/pybind11/genetic_algorithm.h
  include/pybind11/golden_section_order.h
  include/pybind11/gradient_descent.h
  include/pybind11/growing_inputs.h
  include/pybind11/incremental_order.h
  include/pybind11/independent_parameters.h
  include/pybind11/independent_parameters_error.h
  include/pybind11/inputs.h
  include/pybind11/inputs_selection_algorithm.h
  include/pybind11/inputs_trending_layer.h
  include/pybind11/instances.h
  include/pybind11/inverse_sum_squared_error.h
  include/pybind11/k_nearest_neighbors.h
  include/pybind11/kappa_coefficient_optimization_threshold.h
  include/pybind11/levenberg_marquardt_algorithm.h
  include/pybind11/loss_index.h
  include/pybind11/mathematical_model.h
  include/pybind11/matrix.h
  include/pybind11/matthew_correlation_optimization_threshold.h
  include/pybind11/mean_squared_error.h
  include/pybind11/minkowski_error.h
  include/pybind11/missing_values.h
  include/pybind11/model_selection.h
  include/pybind11/multilayer_perceptron.h
  include/pybind11/neural_network.h
  include/pybind11/neural_parameters_norm.h
  include/pybind11/newton_method.h
  include/pybind11/normalized_squared_error.h
  include/pybind11/numerical_differentiation.h
  include/pybind11/numerical_integration.h
  include/pybind11/opennn.h
  include/pybind11/order_selection_algorithm.h
  include/pybind11/ordinary_differential_equations.h
  include/pybind11/outputs.h
  include/pybind11/outputs_integrals.h
  include/pybind11/outputs_trending_layer.h
  include/pybind11/perceptron.h
  include/pybind11/perceptron_layer.h
  include/pybind11/plug_in.h
  include/pybind11/principal_components_layer.h
  include/pybind11/probabilistic_layer.h
  include/pybind11/pruning_inputs.h
  include/pybind11/quasi_newton_method.h
  include/pybind11/random_search.h
  include/pybind11/regularization_term.h
  include/pybind11/roc_area_error.h
  include/pybind11/roc_curve_optimization_threshold.h
  include/pybind11/root_mean_squared_error.h
  include/pybind11/scaling_layer.h
  include/pybind11/selective_pruning.h
  include/pybind11/simulated_annealing_order.h
  include/pybind11/solutions_error.h
  include/pybind11/sparse_matrix.h
  include/pybind11/sum_squared_error.h
  include/pybind11/testing_analysis.h
  include/pybind11/text_analytics.h
  include/pybind11/threshold_selection_algorithm.h
  include/pybind11/tinyxml2.h
  include/pybind11/training_algorithm.h
  include/pybind11/training_rate_algorithm.h
  include/pybind11/training_strategy.h
  include/pybind11/trending_layer.h
  include/pybind11/unscaling_layer.h
  include/pybind11/variables.h
  include/pybind11/vector.h
  include/pybind11/weighted_squared_error.h
  include/pybind11/weighted_squared_regression_error.h
  include/pybind11/youden_index_optimization_threshold.h

)
string(REPLACE "include/" "${CMAKE_CURRENT_SOURCE_DIR}/include/"
       PYBIND11_HEADERS "${PYBIND11_HEADERS}")

if (PYBIND11_TEST)
  add_subdirectory(tests)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# extract project version from source
file(STRINGS "${PYBIND11_INCLUDE_DIR}/pybind11/detail/common.h" pybind11_version_defines
     REGEX "#define PYBIND11_VERSION_(MAJOR|MINOR|PATCH) ")
foreach(ver ${pybind11_version_defines})
  if (ver MATCHES "#define PYBIND11_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
    set(PYBIND11_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
  endif()
endforeach()
set(${PROJECT_NAME}_VERSION ${PYBIND11_VERSION_MAJOR}.${PYBIND11_VERSION_MINOR}.${PYBIND11_VERSION_PATCH})
message(STATUS "pybind11 v${${PROJECT_NAME}_VERSION}")

option (USE_PYTHON_INCLUDE_DIR "Install pybind11 headers in Python include directory instead of default installation prefix" OFF)
if (USE_PYTHON_INCLUDE_DIR)
    file(RELATIVE_PATH CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_PREFIX} ${PYTHON_INCLUDE_DIRS})
endif()

if(NOT (CMAKE_VERSION VERSION_LESS 3.0))  # CMake >= 3.0
  # Build an interface library target:
  add_library(pybind11 INTERFACE)
  add_library(pybind11::pybind11 ALIAS pybind11)  # to match exported target
  target_include_directories(pybind11 INTERFACE $<BUILD_INTERFACE:${PYBIND11_INCLUDE_DIR}>
                                                $<BUILD_INTERFACE:${PYTHON_INCLUDE_DIRS}>
                                                $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_compile_options(pybind11 INTERFACE $<BUILD_INTERFACE:${PYBIND11_CPP_STANDARD}>)

  add_library(module INTERFACE)
  add_library(pybind11::module ALIAS module)
  if(NOT MSVC)
    target_compile_options(module INTERFACE -fvisibility=hidden)
  endif()
  target_link_libraries(module INTERFACE pybind11::pybind11)
  if(WIN32 OR CYGWIN)
    target_link_libraries(module INTERFACE $<BUILD_INTERFACE:${PYTHON_LIBRARIES}>)
  elseif(APPLE)
    target_link_libraries(module INTERFACE "-undefined dynamic_lookup")
  endif()

  add_library(embed INTERFACE)
  add_library(pybind11::embed ALIAS embed)
  target_link_libraries(embed INTERFACE pybind11::pybind11 $<BUILD_INTERFACE:${PYTHON_LIBRARIES}>)
endif()

if (PYBIND11_INSTALL)
  install(DIRECTORY ${PYBIND11_INCLUDE_DIR}/pybind11 DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  # GNUInstallDirs "DATADIR" wrong here; CMake search path wants "share".
  set(PYBIND11_CMAKECONFIG_INSTALL_DIR "share/cmake/${PROJECT_NAME}" CACHE STRING "install path for pybind11Config.cmake")

  configure_package_config_file(tools/${PROJECT_NAME}Config.cmake.in
                                "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                                INSTALL_DESTINATION ${PYBIND11_CMAKECONFIG_INSTALL_DIR})
  # Remove CMAKE_SIZEOF_VOID_P from ConfigVersion.cmake since the library does
  # not depend on architecture specific settings or libraries.
  set(_PYBIND11_CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
  unset(CMAKE_SIZEOF_VOID_P)
  write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
                                   VERSION ${${PROJECT_NAME}_VERSION}
                                   COMPATIBILITY AnyNewerVersion)
  set(CMAKE_SIZEOF_VOID_P ${_PYBIND11_CMAKE_SIZEOF_VOID_P})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
                ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
                tools/FindPythonLibsNew.cmake
                tools/pybind11Tools.cmake
          DESTINATION ${PYBIND11_CMAKECONFIG_INSTALL_DIR})

  if(NOT (CMAKE_VERSION VERSION_LESS 3.0))
    if(NOT PYBIND11_EXPORT_NAME)
      set(PYBIND11_EXPORT_NAME "${PROJECT_NAME}Targets")
    endif()

    install(TARGETS pybind11 module embed
            EXPORT "${PYBIND11_EXPORT_NAME}")
    if(PYBIND11_MASTER_PROJECT)
      install(EXPORT "${PYBIND11_EXPORT_NAME}"
              NAMESPACE "${PROJECT_NAME}::"
              DESTINATION ${PYBIND11_CMAKECONFIG_INSTALL_DIR})
    endif()
  endif()
endif()
