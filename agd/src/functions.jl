# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

struct SmoothPlusProximalFunction
  smooth_value::Function
  smooth_gradient::Function
  proximal_value::Function
  proximal_operator::Function
  dim::Int64 # 0 indicates the problem size can change
end

function value(f::SmoothPlusProximalFunction, point::Vector{Float64})
  return f.smooth_value(point) + f.proximal_value(point)
end

"""
Create a differentiable function (with no proximal component).
"""
function differentiable_function(value::Function, gradient::Function, dim::Int64)
  function identity_proximal_operator(x::Vector, step_size::Float64)
    return x
  end
  function zero_proximal_value(x::Vector)
    return 0.0
  end
  return SmoothPlusProximalFunction(
    value,
    gradient,
    zero_proximal_value,
    identity_proximal_operator,
    dim,
  )
end


function create_linear_regression_function(
    data::RegressionData;
    regularizer::Float64=0.0,
  )
  n = size(data.coefficient_matrix, 1)
  function regression_value(x::Vector)
    residual = data.coefficient_matrix * x - data.target_values
    return 0.5 * sum(residual.^2) + 0.5 * regularizer * sum(x.^2)
  end
  function regression_gradient(x::Vector)
    residual = data.coefficient_matrix * x - data.target_values
    return data.coefficient_matrix' * residual + regularizer * x
  end
  return differentiable_function(
    regression_value,
    regression_gradient,
    size(data.coefficient_matrix, 2)
  )
end

mutable struct UnivariateFunction
  value::Function
  gradient::Function
end

mutable struct UnivariateProximalOperator
  value::Function
  prox::Function
end

function create_general_regression_function(;
    data::RegressionData,
    loss::UnivariateFunction,
    regularizer::UnivariateProximalOperator,
  )
  n = size(data.coefficient_matrix, 1)
  function regression_value(x::Vector)
    residual = data.coefficient_matrix * x - data.target_values
    return sum(loss.value.(residual))
  end
  function regression_gradient(x::Vector)
    residual = data.coefficient_matrix * x - data.target_values
    return data.coefficient_matrix' * loss.gradient.(residual)
  end
  function proximal_value(x::Vector)
    return sum(regularizer.value.(x))
  end
  function proximal_operator(x::Vector, step_size::Float64)
    return regularizer.prox.(x, step_size)
  end
  return SmoothPlusProximalFunction(
    regression_value,
    regression_gradient,
    proximal_value,
    proximal_operator,
    size(data.coefficient_matrix, 2),
  )
end

function create_squared_function(lambda::Float64)
  function squared(x::Float64)
    return 0.5 * lambda * x^2
  end
  function squared_gradient(x::Float64)
    return lambda * x
  end
  return UnivariateFunction(squared, squared_gradient)
end

function create_L1_proximal_operator(lambda::Float64)
  function value(x::Float64)
    return lambda * abs(x)
  end
  function prox(x::Float64, step_size::Float64)
    return sign(x) * max( abs(x) - step_size * lambda, 0.0)
  end
  return UnivariateProximalOperator(value, prox)
end

function log_logistic_loss(x::Float64)
  return log(1 + exp(x))
end

function gradient_of_log_logistic_loss(x::Float64)
  return exp(x) / (1 + exp(x))
end

function create_logistic_regression_problem(;
    data::RegressionData,
    regularizer::UnivariateProximalOperator,
  )
  positive_label = maximum(data.target_values)
  negative_label = minimum(data.target_values)

  # identify the subsets of rows that correspond to the positive and negative
  # labels
  positive_labels = positive_label .== data.target_values
  negative_labels = negative_label .== data.target_values

  if sum(positive_labels) + sum(negative_labels) != length(data.target_values)
    error("More than two labels in data. This code can only handle two labels. ")
  end
  coefficient_matrix_positive_labels = data.coefficient_matrix[positive_labels,:]
  coefficient_matrix_negative_labels = data.coefficient_matrix[negative_labels,:]
  target_values_positive_labels = data.target_values[positive_labels,:]
  target_values_negative_labels = data.target_values[negative_labels,:]

  function logistic_value(x::Vector)
    positive_residuals = coefficient_matrix_positive_labels * x
    negative_residuals = -coefficient_matrix_negative_labels * x

    return sum(log_logistic_loss.(positive_residuals)) +
      sum(log_logistic_loss.(negative_residuals))
  end

  function logistic_gradient(x::Vector)
    positive_residuals = coefficient_matrix_positive_labels * x
    positive_gradient = coefficient_matrix_positive_labels' * gradient_of_log_logistic_loss.(positive_residuals)
    negative_residuals = -coefficient_matrix_negative_labels * x
    negative_gradient = -coefficient_matrix_negative_labels' * gradient_of_log_logistic_loss.(negative_residuals)

    return (positive_gradient + negative_gradient)[:]
  end

  function proximal_value(x::Vector)
    return sum(regularizer.value.(x))
  end
  function proximal_operator(x::Vector, step_size::Float64)
    return regularizer.prox.(x, step_size)
  end
  return SmoothPlusProximalFunction(
    logistic_value,
    logistic_gradient,
    proximal_value,
    proximal_operator,
    size(data.coefficient_matrix, 2),
  )
end
