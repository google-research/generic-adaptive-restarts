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

# run download lib svm before running this code
include("../src/read-libsvm/read-libsvm.jl")

using Test
@testset "load_libsvm_file" begin
  # Checks that E2006.train is loaded and there are no errors in the data.
  file_name = joinpath(@__DIR__, "../data/E2006.train")
  result = load_libsvm_file(file_name)
  @test length(result.target_values) == 16087
  @test size(result.coefficient_matrix, 1) == 16087
  @test size(result.coefficient_matrix, 2) == 150360
  @test result.target_values[1] == -3.58943963121738
  @test result.target_values[2] == -3.52663816315417
  @test result.coefficient_matrix[1,376] == 0.000187741243486602
  @test result.coefficient_matrix[1,435] == 0.00012679437327207
  @test result.coefficient_matrix[1,435] == 0.00012679437327207
  @test result.coefficient_matrix[2,667] == 0.000139063750019391
  preprocessed_data = preprocess_regression_problem(result)
  @test size(preprocessed_data.coefficient_matrix, 2) == 150349
end
