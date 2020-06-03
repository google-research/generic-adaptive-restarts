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

using Test
include("../src/AGD_distance_restart.jl")

# A quadratic of the form
#   sum_{i=1}^{n} 0.5 * i * q[i]^2
# for testing.
function simple_quadratic_value(x::Vector{Float64})
  q = (1:length(x))
  return 0.5 * sum(q[i] * x[i]^2 for i = 1:length(x))
end
function simple_quadratic_gradient(x::Vector{Float64})
  q = (1:length(x))
  return x .* q
end
testing_function = differentiable_function(
  simple_quadratic_value,
  simple_quadratic_gradient,
  0,
)

v_0 = ones(10)
grad_v_0 = testing_function.smooth_gradient(v_0)
@testset "do_gradient_step" begin
  result =
    do_gradient_step_with_backtracking_line_search(
      testing_function,
      v_0,
      grad_v_0,
      1.0,
      1e-8,
      2.0,
    )
  @test result.step_size == 0.125
end

@testset "test_agd" begin
  parameters = default_basic_agd_parameters()
  parameters.max_iter=1000
  parameters.verbose = false
  v_0 = [30.0, 10.0, 4.0, 24.0, 4.0, 5.0, 1.0, 2.0, 10.0, 30.0, 10.0, 4.0, 24.0,
    4.0, 5.0, 1.0, 2.0, 10.0, 1.0, 2.0, 10.0, 30.0, 10.0, 4.0, 24.0,
    4.0, 5.0, 1.0, 2.0, 10.0]
  basic_final_point, basic_history = basic_AGD(testing_function,
    v_0,
    parameters)
  @test norm(basic_final_point, 2) < 1e-3

  final_point_fixed_frequency, fixed_frequency_history =
    fixed_frequency_AGD(testing_function, v_0,
                                              parameters, 10)
  @test norm(final_point_fixed_frequency, 2) < 1e-8

  function_value_final_point, fval_restart_history =
    function_value_restart_AGD(testing_function, v_0, parameters
     )
  @test norm(function_value_final_point, 2) < 1e-8

  final_point_adaptive, adaptive_history =
    our_adaptive_restart_AGD(testing_function, v_0, parameters)
  @test norm(final_point_adaptive, 2) < 1e-8
end

parameters = default_basic_agd_parameters()
parameters.max_iter=2000
parameters.verbose = false
n = 5000
v_0 = ones(n)

basic_history, fval_restart_history, adaptive_history = run_algs(
  testing_function, v_0, parameters);
plot_algs(basic_history, fval_restart_history,
  adaptive_history)
savefig(joinpath(@__DIR__, "../plots/test.pdf"))
