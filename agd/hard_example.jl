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
julia --project=[...] hard_example.jl n delta alpha
"""

include("src/AGD_distance_restart.jl")

gr()
fntsm = Plots.font("serif", pointsize=12)
fntlg = Plots.font("serif", pointsize=18)
default(titlefont=fntlg, guidefont=fntlg, tickfont=fntsm, legendfont=fntsm)

n = parse(Int64, ARGS[1])
delta = parse(Float64, ARGS[2])
alpha = parse(Float64, ARGS[3])

function h_val(x::Float64)
  if x >= -delta
    return 0.5 * x^2
  else
    return -delta * x - 0.5 * delta^2
  end
end
function h_grad(x::Float64)
  if x >= -delta
    return x
  else
    return -delta
  end
end

vec = collect(1:n) .+ 0.0
function f_val(x::Vector{Float64})
  return sum(h_val.(x) .* vec) + 0.5 * alpha * sum(x.^2)
end
function f_grad(x::Vector{Float64})
  return h_grad.(x) .* vec + alpha * x
end

hard_function = differentiable_function(
  f_val,
  f_grad,
  n,
)

v_0 = -ones(n)
f_min_threshold = 1e-10
parameters = default_basic_agd_parameters(
  verbose=false,
  max_iter=40000,
  function_value_optimality_threshold=f_min_threshold,
);
basic_history, fval_restart_history, our_adaptive_history = run_algs(
  hard_function, v_0, parameters
);

optimality_threshold = 1e-8

BEST_FIXED_FREQUENCY = true
if BEST_FIXED_FREQUENCY
  best_fixed_frequency_result =
    pick_best_fixed_restart_frequency(hard_function, v_0, parameters;
      set_of_frequencies = [128, 256, 512, 1024, 2048, 4096, 8192],
      approximate_optimality_threshold = optimality_threshold)
  best_fixed_frequency_history = best_fixed_frequency_result.history
  print("best_fixed_frequency = ", best_fixed_frequency_result.frequency)
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
println("Plot name: hard-example")
savefig(joinpath(@__DIR__, "plots/hard-example.pdf"))


# plot h_vals function for a larger delta value
delta = 1e-1
function h_val(x::Float64)
  if x >= -delta
    return 0.5 * x^2
  else
    return -delta * x - 0.5 * delta^2
  end
end

x_vals = collect(range(-1,stop=1,length=200))
y_vals = h_val.(x_vals)
plot(x_vals,y_vals, xlabel="x", ylabel="h(x)", legend=false)

savefig(joinpath(@__DIR__, "plots/h_func.pdf"))
