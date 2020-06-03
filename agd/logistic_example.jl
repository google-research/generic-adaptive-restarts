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

"""
Usage:
julia --project=[...] logistic_example.jl data_source lambda
"""

include("src/AGD_distance_restart.jl")

gr()
fntsm = Plots.font("serif", pointsize=12)
fntlg = Plots.font("serif", pointsize=18)
default(titlefont=fntlg, guidefont=fntlg, tickfont=fntsm, legendfont=fntsm)

@assert length(ARGS) == 2
data_source = joinpath(@__DIR__, "data", ARGS[1])
println("Data source: ", data_source)
lambda = parse(Float64, ARGS[2])
println("Lambda: ", lambda)
original_data = load_libsvm_file(data_source)
data = preprocess_regression_problem(original_data)

# logistic regression plus L1 regularizer
logistic_regression_function = create_logistic_regression_problem(;
    data=data,
    regularizer=create_L1_proximal_operator(lambda),
  );

v_0 = zeros(logistic_regression_function.dim);

parameters = default_basic_agd_parameters();
parameters.max_iter=2000;

basic_history, fval_restart_history, our_adaptive_history = run_algs(
  logistic_regression_function, v_0, parameters
);


optimality_threshold = 1e-8
approximate_minimum = min(
  minimum(basic_history.fval_history),
  minimum(fval_restart_history.fval_history),
  minimum(our_adaptive_history.fval_history),
) - optimality_threshold / 10.0

# hyperparameter search to find the best fixed frequency
BEST_FIXED_FREQUENCY = true
if BEST_FIXED_FREQUENCY
  best_fixed_frequency_result =
    pick_best_fixed_restart_frequency(
      logistic_regression_function,
      v_0,
      parameters;
      set_of_frequencies = [128, 256, 512, 1024, 2048],
      approximate_optimality_threshold = approximate_minimum + optimality_threshold,
    )
  best_fixed_frequency_history = best_fixed_frequency_result.history
  print("best_fixed_frequency = ", best_fixed_frequency_result.frequency)
end

basic_history.fval_history .-= approximate_minimum;
fval_restart_history.fval_history .-= approximate_minimum;
our_adaptive_history.fval_history .-= approximate_minimum;
if BEST_FIXED_FREQUENCY
  best_fixed_frequency_history.fval_history .-= approximate_minimum
end

plot_algs(
  basic_history,
  fval_restart_history,
  our_adaptive_history,
  y_min=optimality_threshold,
)

if BEST_FIXED_FREQUENCY
  plot_history!(best_fixed_frequency_history;
      label="Best fixed ($(best_fixed_frequency_result.frequency))",
      color=3,
    )
end

println("")
println("Plot name: ", ARGS[1])
savefig(joinpath(@__DIR__, "plots", ARGS[1] * ".pdf"))
