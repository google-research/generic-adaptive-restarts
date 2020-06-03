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
Reads files from
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
"""

using SparseArrays, LinearAlgebra

mutable struct RegressionData
  coefficient_matrix::SparseMatrixCSC{Float64,Int64}
  target_values::Vector{Float64}
end

function load_libsvm_file(file_name::String)
  open(file_name, "r") do io
    target = Array{Float64,1}()
    row_indicies = Array{Int64,1}()
    col_indicies = Array{Int64,1}()
    matrix_values = Array{Float64,1}()

    row_index = 0
    for line in eachline(io)
      row_index += 1
      split_line = split(line)
      push!(target, parse(Float64,split_line[1]))
      for i = 2:length(split_line)
        push!(row_indicies, row_index)
        matrix_coef = split(split_line[i],":")
        push!(col_indicies, parse(Int64,matrix_coef[1]))
        push!(matrix_values, parse(Float64,matrix_coef[2]))
      end
    end
    coefficient_matrix = sparse(row_indicies, col_indicies, matrix_values)
    return RegressionData(coefficient_matrix, target)
  end
end

function normalize_columns(coefficient_matrix::SparseMatrixCSC{Float64,Int64})
  m = size(coefficient_matrix,2)
  normalize_columns_by = ones(m)
  for j = 1:m
    col_vals = coefficient_matrix[:,j].nzval
    if length(col_vals) > 0
      normalize_columns_by[j] = 1.0 / norm(col_vals, 2)
    end
  end
  return coefficient_matrix * sparse(1:m,1:m,normalize_columns_by)
end

function remove_empty_columns(coefficient_matrix::SparseMatrixCSC{Float64,Int64})
  keep_cols = Array{Int64,1}()
  for j = 1:size(coefficient_matrix,2)
    if length(coefficient_matrix[:,j].nzind) > 0
      push!(keep_cols, j)
    end
  end
  return coefficient_matrix[:,keep_cols]
end

function add_intercept(coefficient_matrix::SparseMatrixCSC{Float64,Int64})
  return [sparse(ones(size(coefficient_matrix,1))) coefficient_matrix]
end

function preprocess_regression_problem(result::RegressionData)
  result.coefficient_matrix = remove_empty_columns(result.coefficient_matrix)
  result.coefficient_matrix = add_intercept(result.coefficient_matrix)
  result.coefficient_matrix = normalize_columns(result.coefficient_matrix)
  return result
end
