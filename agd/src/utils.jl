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

using Plots, LaTeXStrings

function run_algs(test_function::SmoothPlusProximalFunction, v_0::Vector{Float64},
    parameters::Basic_agd_parameters)
  println("Running basic_AGD")
  final_point, basic_history = basic_AGD(test_function,
    v_0,
    parameters)

  println("Running function_value_restart_AGD")
  final_point, fval_restart_history =
    function_value_restart_AGD(test_function, v_0, parameters
     )

  println("Running our_adaptive_scheme_AGD")
  final_point, our_adaptive_history =
    our_adaptive_restart_AGD(test_function, v_0, parameters)

  return basic_history, fval_restart_history, our_adaptive_history
end

function plot_algs(basic_history, fval_restart_history, our_adaptive_history;
  y_min::Float64=1e-8)
 plot(basic_history.fval_history, yaxis=:log, line=(2,:solid),
    label="No restarts", color=1)
 plot_history!(fval_restart_history,
    label="O'D. and C.", color=4)
 plot_history!(our_adaptive_history, label="Our adaptive", color=2)

 f_max = max(
  maximum(basic_history.fval_history),
  maximum(fval_restart_history.fval_history),
  maximum(our_adaptive_history.fval_history),
 )
 ylims!((y_min, 2.0 * f_max))
 plot!(xlabel="Iteration", ylabel=L"$f(x) - f^{*}$")
end

function plot_history!(history::AGD_history;
    label::String="",
    color="black",
    line=(2,:solid),
  )
  plot!(history.fval_history,
    yaxis=:log,
    label=label,
    color=color,
    line=line,
  )
  scatter!(history.restart_intervals,
    history.fval_history[history.restart_intervals],
    yaxis=:log,
    label="",
    color=color,
  )
end

function power_of_two_list(list::Array{Int64,1})
  power_list = Array{Int64,1}()
  for i in list
    push!(power_list, 2^i)
  end
  return power_list
end

function pick_best_fixed_restart_frequency(test_function, v_0, parameters;
    set_of_frequencies::Array{Int64,1},
    approximate_optimality_threshold::Float64)
  best_number_of_iter = Inf
  best_fval = Inf
  best_history = nothing
  best_freq = NaN
  best_point = nothing
  for freq in set_of_frequencies
    final_point, fixed_frequency_history =
      fixed_frequency_AGD(test_function, v_0, parameters, freq)
    this_frequency_is_best = false
    # find the first iteration that is below approximate_optimality_threshold
    lower_iterations = findall(x -> x < approximate_optimality_threshold, fixed_frequency_history.fval_history)
    if length(lower_iterations) > 1
      num_iter_to_approx_opt = lower_iterations[1]
    else
      num_iter_to_approx_opt = length(fixed_frequency_history.fval_history)
    end
    if num_iter_to_approx_opt <= best_number_of_iter
      final_fval = fixed_frequency_history.fval_history[num_iter_to_approx_opt]
      if num_iter_to_approx_opt < best_number_of_iter ||
        final_fval < best_fval
         best_number_of_iter = num_iter_to_approx_opt
         best_fval = final_fval
         best_history = fixed_frequency_history
         best_freq = freq
         best_point = final_point
      end
    end
  end
  if best_freq == maximum(set_of_frequencies)
    @warn "best frequency is equal to maximum"
  end
  if best_freq == minimum(set_of_frequencies)
    @warn "best frequency is equal to minimum"
  end
  return (frequency=best_freq, history=best_history, point=best_point)
end
