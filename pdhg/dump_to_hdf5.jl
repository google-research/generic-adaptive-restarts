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

# usage: dump_to_hdf5.jl [mps file source] [hdf5 file dest]

import HDF5
import MathProgBase
import GLPK  # Used for reading MPS files.
import GLPKMathProgInterface
using SparseArrays

"""
```
minimize objective_vector' * x + objective_constant

s.t. constraint_matrix[1:num_equalities, :] * x =
     right_hand_side[1:num_equalities]

     constraint_matrix[(num_equalities + 1):end, :] * x >=
     right_hand_side[(num_equalities + 1):end]

     variable_lower_bound <= x <= variable_upper_bound
```
"""
mutable struct LinearProgrammingProblem
  variable_lower_bound::Vector{Float64}
  variable_upper_bound::Vector{Float64}
  objective_vector::Vector{Float64}
  objective_constant::Float64
  constraint_matrix::SparseMatrixCSC{Float64,Int64}
  right_hand_side::Vector{Float64}
  num_equalities::Int64
end

mutable struct TwoSidedLpProblem
  variable_lower_bound::Vector{Float64}
  variable_upper_bound::Vector{Float64}
  constraint_lower_bound::Vector{Float64}
  constraint_upper_bound::Vector{Float64}
  constraint_matrix::SparseArrays.SparseMatrixCSC{Float64,Int64}
  objective_vector::Vector{Float64}
end

function transform_to_standard_form(lp::TwoSidedLpProblem)
  is_equality_row = lp.constraint_lower_bound .== lp.constraint_upper_bound
  is_geq_row = .!is_equality_row .& isfinite.(lp.constraint_lower_bound)
  is_leq_row = .!is_equality_row .& isfinite.(lp.constraint_upper_bound)

  @assert !any(is_geq_row .& is_leq_row)

  num_equalities = sum(is_equality_row)

  if num_equalities +
     sum(is_geq_row) +
     sum(is_leq_row) != length(lp.constraint_lower_bound)
    error("Not all constraints have finite bounds on at least one side.")
  end

  # Flip the signs of the leq rows in place.
  for idx in 1:SparseArrays.nnz(lp.constraint_matrix)
    if is_leq_row[lp.constraint_matrix.rowval[idx]]
      lp.constraint_matrix.nzval[idx] *= -1
    end
  end

  new_row_to_old = [findall(is_equality_row); findall(.!is_equality_row)]
  if new_row_to_old != 1:size(lp.constraint_matrix, 1)
    row_permute_in_place(lp.constraint_matrix, invperm(new_row_to_old))
  end

  right_hand_side = copy(lp.constraint_lower_bound)
  right_hand_side[is_leq_row] .= .-lp.constraint_upper_bound[is_leq_row]
  permute!(right_hand_side, new_row_to_old)

  return LinearProgrammingProblem(
    lp.variable_lower_bound,
    lp.variable_upper_bound,
    lp.objective_vector,
    0.0,
    lp.constraint_matrix,
    right_hand_side,
    num_equalities,
  )
end

function read_mps_to_standard_form(filename::String)
  if !endswith(filename, ".mps.gz") && !endswith(filename, ".mps")
    error("Invalid name: `filename` needs to end with `.mps.gz` or `.mps`.")
  end

  model = MathProgBase.LinearQuadraticModel(GLPKMathProgInterface.GLPKSolverMIP(
    msg_lev = GLPK.MSG_OFF,
  ))
  MathProgBase.loadproblem!(model, filename)

  return transform_to_standard_form(TwoSidedLpProblem(
    MathProgBase.getvarLB(model),
    MathProgBase.getvarUB(model),
    MathProgBase.getconstrLB(model),
    MathProgBase.getconstrUB(model),
    MathProgBase.getconstrmatrix(model),
    MathProgBase.getobj(model),
  ))
end

lp = read_mps_to_standard_form(ARGS[1])

HDF5.h5open(ARGS[2], "w") do file
  write(file, "variable_lower_bound", lp.variable_lower_bound)
  write(file, "variable_upper_bound", lp.variable_upper_bound)
  write(file, "right_hand_side", lp.right_hand_side)
  write(file, "objective_vector", lp.objective_vector)
  write(file, "objective_constant", lp.objective_constant)
  write(file, "num_equalities", lp.num_equalities)
  # The matrix in zero-based CSC format.
  write(file, "constraint_matrix_data", lp.constraint_matrix.nzval)
  write(file, "constraint_matrix_indices", lp.constraint_matrix.rowval .- 1)
  write(file, "constraint_matrix_indptr", lp.constraint_matrix.colptr .- 1)
end
