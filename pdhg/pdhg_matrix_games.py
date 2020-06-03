# coding=utf-8
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

"""Matrix games with PDHG."""

# Usage: pdhg_matrix_games.py [output directory]

import collections
import os
import sys
from . import restarted_pdhg
import cvxpy as cp
import numpy as np
import odl
import pandas as pd


np.random.seed(12)

OUTPUT_DIR = sys.argv[1]




def uniform_game(num_rows, num_cols):
  return 0.5 * np.random.random_sample((num_rows, num_cols)) - 1


def normal_game(num_rows, num_cols):
  return np.random.normal(size=(num_rows, num_cols))


class IndicatorSimplexConjugate(odl.solvers.functional.Functional):
  """Implements the convex conjugate of the indicator of the simplex."""

  def __init__(self, space, diameter=1, sum_rtol=None):
    super(IndicatorSimplexConjugate, self).__init__(
        space=space, linear=False, grad_lipschitz=np.nan)
    # Confuses the primal and the dual space. Luckily they're the same.
    self.diameter = diameter
    self.sum_rtol = sum_rtol

  @property
  def convex_conj(self):
    """The convex conjugate."""
    return odl.solvers.IndicatorSimplex(self.domain, self.diameter,
                                        self.sum_rtol)


class CallbackStore(odl.solvers.Callback):

  def __init__(self, payoff_matrix):
    self.payoff_matrix = payoff_matrix
    self.residuals_at_current = []
    self.residuals_at_avg = []

  def __call__(self, x, y, x_avg, y_avg, did_restart):
    self.residuals_at_current.append(residual(self.payoff_matrix, x, y))
    self.residuals_at_avg.append(residual(self.payoff_matrix, x_avg, y_avg))


def residual(payoff_matrix, primal, dual):
  return np.amax(payoff_matrix @ primal) - np.amin(payoff_matrix.T @ dual)


def solve_lp(payoff_matrix):
  # This is a helper function to compute the ground truth solution of the matrix
  # game. It's not used in the results presented.
  x = cp.Variable(payoff_matrix.shape[1])
  # The activity variables are created explicitly so we can access the duals.
  activity = cp.Variable(payoff_matrix.shape[0])
  prob = cp.Problem(
      cp.Minimize(cp.max(activity)),
      [activity == payoff_matrix @ x, cp.sum(x) == 1, x >= 0])
  prob.solve(solver=cp.CVXOPT)
  return prob.value, x.value, -prob.constraints[0].dual_value


def solve_game(payoff_matrix,
               num_iters,
               tau,
               sigma,
               restart,
               fixed_restart_frequency=None):
  linear_operator = odl.MatrixOperator(payoff_matrix)
  primal_space = linear_operator.domain
  dual_space = linear_operator.range
  indicator_primal_simplex = odl.solvers.IndicatorSimplex(primal_space)
  conjugate_of_indicator_dual_simplex = IndicatorSimplexConjugate(dual_space)
  x = primal_space.zero()
  y = dual_space.zero()
  callback = CallbackStore(payoff_matrix)
  restarted_pdhg.restarted_pdhg(
      x,
      f=indicator_primal_simplex,
      g=conjugate_of_indicator_dual_simplex,
      L=linear_operator,
      niter=num_iters,
      tau=tau,
      sigma=sigma,
      y=y,
      callback=callback,
      restart=restart,
      fixed_restart_frequency=fixed_restart_frequency)
  return callback.residuals_at_current, callback.residuals_at_avg


NUM_ITERS = 20000
NUM_REPS = 50


def generate_results(game_generator):
  iteration_log = collections.defaultdict(list)
  for i in range(NUM_REPS):
    print(i, 'of', NUM_REPS)
    payoff_matrix = game_generator()

    # odl's operator norm estimates are significantly off for normal_game. The
    # matrices are small enough that we can compute the exact value instead.
    tau = sigma = np.sqrt(0.9) / np.linalg.norm(payoff_matrix, ord=2)

    iteration_log['pdhg'].append(
        solve_game(payoff_matrix, NUM_ITERS, tau, sigma, restart='none'))

    iteration_log['pdhg adaptive'].append(
        solve_game(payoff_matrix, NUM_ITERS, tau, sigma, restart='adaptive'))

    for restart_frequency in [8, 32, 128, 512, 2048]:
      residuals_at_current, residuals_at_avg = solve_game(
          payoff_matrix,
          NUM_ITERS,
          tau,
          sigma,
          restart='fixed',
          fixed_restart_frequency=restart_frequency)
      iteration_log['pdhg restart {}'.format(restart_frequency)].append(
          (residuals_at_current, residuals_at_avg))
  return iteration_log


PERCENTILE = 90


def write_to_csv(iteration_log, file_name):
  data = {
      'iteration_num': [],
      'method': [],
      'median_residual_at_current': [],
      'lower_range_at_current': [],
      'upper_range_at_current': [],
      'median_residual_at_avg': [],
      'lower_range_at_avg': [],
      'upper_range_at_avg': []
  }
  for method in iteration_log:
    for it in range(NUM_ITERS):
      data['iteration_num'].append(it)
      data['method'].append(method)
      residuals_this_iter_at_current = [
          log[it] for log in iteration_log[method][0]
      ]
      data['median_residual_at_current'].append(
          np.percentile(residuals_this_iter_at_current, 50))
      data['lower_range_at_current'].append(
          np.percentile(residuals_this_iter_at_current, 100 - PERCENTILE))
      data['upper_range_at_current'].append(
          np.percentile(residuals_this_iter_at_current, PERCENTILE))
      residuals_this_iter_at_avg = [log[it] for log in iteration_log[method][1]]
      data['median_residual_at_avg'].append(
          np.percentile(residuals_this_iter_at_avg, 50))
      data['lower_range_at_avg'].append(
          np.percentile(residuals_this_iter_at_avg, 100 - PERCENTILE))
      data['upper_range_at_avg'].append(
          np.percentile(residuals_this_iter_at_avg, PERCENTILE))
  df = pd.DataFrame.from_dict(data)
  df.to_csv(file_name)


os.makedirs(OUTPUT_DIR, exist_ok=True)
print('Solving uniform game')
write_to_csv(
    generate_results(lambda: uniform_game(100, 100)),
    os.path.join(OUTPUT_DIR, 'uniform_game_100_100.csv'))
print('Solving normal game')
write_to_csv(
    generate_results(lambda: normal_game(100, 100)),
    os.path.join(OUTPUT_DIR, 'normal_game_100_100.csv'))
