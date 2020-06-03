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

using LinearAlgebra, Printf

@enum BacktrackingStatus SUCCESS STEP_SIZE_TOO_SMALL

mutable struct BacktrackingLineSearchResult
  accepted_point::Vector{Float64}
  fval_accepted::Float64
  step_size::Float64
  status::BacktrackingStatus
end

function do_gradient_step_with_backtracking_line_search(
    f::SmoothPlusProximalFunction,
    current_point::Vector{Float64},
    current_gradient::Vector{Float64},
    initial_step_size::Float64,
    min_step_size::Float64,
    backtracking_factor::Float64,
  )
  current_smooth_value = f.smooth_value(current_point)
  step_size = initial_step_size
  max_backtracking_iter =
    ceil(log(step_size / min_step_size) / log(backtracking_factor))
  for j = 1:max_backtracking_iter
    trial_point = f.proximal_operator(
      current_point - step_size * current_gradient,
      step_size
    )
    f_trial = value(f, trial_point)
    f_upper_bound = current_smooth_value + dot(current_gradient,
      trial_point - current_point) + (0.5 / step_size) *
      sum((trial_point - current_point).^2) +
      f.proximal_value(trial_point)
    if f_trial > f_upper_bound
      step_size /= backtracking_factor
      if step_size < min_step_size
         @warn "Backtracking gradient step size too small"
         return BacktrackingLineSearchResult(trial_point, f_trial, step_size, STEP_SIZE_TOO_SMALL)
      end
    else
      return BacktrackingLineSearchResult(trial_point, f_trial, step_size, SUCCESS)
    end
  end
  error("Backtracking code hit maximum iterations $max_backtracking_iter " *
    "with step size = $step_size")
end

mutable struct AGD_history
  fval_history::Vector{Float64}
  distance_history::Vector{Float64}
  restart_intervals::Vector{Int64}
end

function create_AGD_history(starting_point::Vector{Float64}, fval::Float64)
  return AGD_history([fval], [0.0], Vector{Int64}())
end

mutable struct AGD_state
  v_t::Vector{Float64}
  u_t::Vector{Float64}
  u_old::Vector{Float64} # u_{t-1}
  lambda::Float64
  step_size::Float64
end

mutable struct Basic_agd_parameters
  # Amount we reduce the step size during the backtracking line search.
  backtracking_factor::Float64
  # Allows us to increase the step size when steps are successful, not part of
  # the original FISTA algorithm.
  forwardtracking_factor::Float64
  # Algorithm will terminate with warning if step sizes falls below this
  # threshold.
  min_step_size::Float64
  initial_step_size::Float64
  verbose::Bool
  max_iter::Int64
  # algorithm will terminate if function value falls below this
  function_value_optimality_threshold::Float64
end

function default_basic_agd_parameters(;
    backtracking_factor::Float64=1.25,
    forwardtracking_factor::Float64=1.0,
    min_step_size::Float64=1e-8,
    initial_step_size::Float64=1.0,
    verbose::Bool=false,
    max_iter::Int64=100,
    function_value_optimality_threshold::Float64=-Inf,
  )
  parameters = Basic_agd_parameters(
    backtracking_factor,
    forwardtracking_factor,
    min_step_size,
    initial_step_size,
    verbose,
    max_iter,
    function_value_optimality_threshold,
  )
  validate(parameters)
  return parameters
end

function initialize_agd(starting_point::Vector{Float64},
    parameters::Basic_agd_parameters)
  return AGD_state(starting_point,
    starting_point,
    starting_point,
    1.0,
    parameters.initial_step_size)
end

function validate(parameters::Basic_agd_parameters)
  @assert parameters.backtracking_factor > 1.0
  @assert parameters.forwardtracking_factor >= 1.0
  @assert parameters.min_step_size >= 0.0
  @assert parameters.max_iter > 0
end

function agd_output_header(fval::Float64, initial_step_size::Float64)
  println("t | f(u_t) | step_size | momentum")
  Printf.@printf("%i \t %3.e \t %3.e \t %3.e\n", 0, fval, initial_step_size, NaN)
end

function take_agd_step(
    agd_state::AGD_state,
    history::AGD_history,
    f::SmoothPlusProximalFunction,
    parameters::Basic_agd_parameters
  )
  current_gradient = f.smooth_gradient(agd_state.v_t)
  # Take a gradient step to get to the point u_t from v_t.
  linesearch_result =
    do_gradient_step_with_backtracking_line_search(
      f,
      agd_state.v_t,
      current_gradient,
      agd_state.step_size,
      parameters.min_step_size,
      parameters.backtracking_factor,
    )
  push!(history.fval_history, linesearch_result.fval_accepted)

  # Compute momentum parameter
  old_lambda = agd_state.lambda
  new_lambda = (1.0 + sqrt(1.0 + 4 * old_lambda^2)) / 2.0
  momentum = (old_lambda - 1.0) / new_lambda
  agd_state.lambda = new_lambda

  # Take momentum step
  u_new = linesearch_result.accepted_point
  agd_state.v_t = u_new + momentum * (u_new - agd_state.u_t)
  agd_state.u_old = agd_state.u_t
  agd_state.u_t = u_new

  # Update step size
  new_step_size = linesearch_result.step_size
  if new_step_size < agd_state.step_size
    agd_state.step_size = new_step_size
  elseif new_step_size == agd_state.step_size
    # The backtracking search succeeded on its first try so increase the step
    # size.
    agd_state.step_size *= parameters.forwardtracking_factor
  else
    error("Backtracking search increased step size from" *
          "$(agd_state.step_size) to $new_step_size")
  end

  if parameters.verbose
    Printf.@printf(" %3.e \t %3.e \t %3.e\n", new_fval, agd_state.step_size,
      momentum)
  end

  return linesearch_result.status == SUCCESS
end

"""
Implementation of the backtracking FISTA algorithm from

A Fast Iterative Shrinkage-Thresholding Algorithm
for Linear Inverse Problems Amir Beck and Marc Teboulle.
"""
function basic_AGD(f::SmoothPlusProximalFunction,
    v_0::Vector{Float64},
    parameters::Basic_agd_parameters)
  initial_fval = value(f, v_0)
  history = create_AGD_history(v_0, initial_fval)
  agd_state = initialize_agd(v_0, parameters)

  for t = 1:(parameters.max_iter)
     successful_step = take_agd_step(agd_state, history, f, parameters)
     if !successful_step || history.fval_history[end] <=
        parameters.function_value_optimality_threshold
       break
     end
     push!(history.distance_history, norm(v_0 - agd_state.v_t, 2))
  end

  return agd_state.u_t, history
end

function fixed_frequency_AGD(
    f::SmoothPlusProximalFunction,
    v_0::Vector{Float64},
    parameters::Basic_agd_parameters,
    restart_period::Int64)
  initial_fval = value(f, v_0)
  history = create_AGD_history(v_0, initial_fval)
  agd_state = initialize_agd(v_0, parameters)

  for t = 1:(parameters.max_iter)
     successful_step = take_agd_step(agd_state, history, f, parameters)
     if !successful_step || history.fval_history[end] <=
        parameters.function_value_optimality_threshold
       break
     end
     push!(history.distance_history, norm(v_0 - agd_state.v_t, 2))
     if t % restart_period == 0
       # Perform restart.
       agd_state = initialize_agd(agd_state.u_t, parameters)
       push!(history.restart_intervals, t)
     end
  end

  return agd_state.u_t, history
end

function function_value_restart_AGD(
    f::SmoothPlusProximalFunction,
    v_0::Vector{Float64},
    parameters::Basic_agd_parameters)
  initial_fval = value(f, v_0)
  history = create_AGD_history(v_0, initial_fval)
  agd_state = initialize_agd(v_0, parameters)
  for t = 1:(parameters.max_iter)
     successful_step = take_agd_step(agd_state, history, f, parameters)
     if !successful_step || history.fval_history[end] <=
        parameters.function_value_optimality_threshold
       break
     end
     push!(history.distance_history, norm(v_0 - agd_state.v_t, 2))
     if t > 1 && history.fval_history[end] > history.fval_history[end - 1]
       # Perform restart.
       agd_state = initialize_agd(agd_state.u_old, parameters)
       push!(history.restart_intervals, t)
     end
  end

  return agd_state.u_t, history
end

"""
This is our restart scheme.
"""
function our_adaptive_restart_AGD(
    f::SmoothPlusProximalFunction,
    v_0::Vector{Float64},
    basic_parameters::Basic_agd_parameters;
    beta::Float64=0.25,
    initial_restart_period::Int64=1,
  )
  initial_fval = value(f, v_0)
  history = create_AGD_history(v_0, initial_fval)
  agd_state = initialize_agd(v_0, basic_parameters)
  last_restart_point = v_0
  theta_k = Inf
  iterations_since_last_restart = 0
  for t = 1:(basic_parameters.max_iter)
     iterations_since_last_restart += 1
     successful_step = take_agd_step(agd_state, history, f, basic_parameters)
     if !successful_step || history.fval_history[end] <=
        basic_parameters.function_value_optimality_threshold
       break
     end
     push!(history.distance_history,
      norm(last_restart_point - agd_state.v_t, 2))
     # Restart scheme
     theta_new = norm(agd_state.u_t - last_restart_point, 2) / (agd_state.lambda)^2
     if theta_k == Inf
        do_restart = t >= initial_restart_period
     else
        do_restart = theta_new < theta_k * beta
     end
     if do_restart
        last_restart_point = agd_state.u_t
        agd_state = initialize_agd(last_restart_point, basic_parameters)
        theta_k = theta_new
        iterations_since_last_restart = 0
        push!(history.restart_intervals, t)
     end
  end

  return agd_state.u_t, history
end
