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

"""Linear programming with PDHG."""

import collections
from . import restarted_pdhg
import h5py
import numpy as np
import odl
import pandas as pd
import scipy.sparse





class LinearOnBox(odl.solvers.functional.Functional):
  """A linear function within a box, and infinity outside the box.

  Another way to say this is that this function is the sum of a linear function
  and the indicator function of the box.
  """

  def __init__(self, space, linear_coefficients, lower, upper):
    super(LinearOnBox, self).__init__(space, linear=False)
    self.lower = lower
    self.upper = upper
    self.linear_coefficients = space.element(linear_coefficients)

  # _call not implemented.

  @property
  def proximal(self):
    lower = self.lower
    upper = self.upper
    linear_coefficients = self.linear_coefficients
    space = self.domain

    class ProxLinearOnBox(odl.operator.operator.Operator):

      def __init__(self, sigma):
        super(ProxLinearOnBox, self).__init__(
            domain=space, range=space, linear=False)
        self.sigma = sigma

      def _call(self, x, out):
        """Apply the operator to ``x`` and store the result in ``out``."""
        # out = x - sigma * linear_coefficients
        out.lincomb(1.0, x, -self.sigma, linear_coefficients)
        # Now project to the box.
        out.ufuncs.maximum(lower, out=out)
        out.ufuncs.minimum(upper, out=out)

    return ProxLinearOnBox


class LinearOnBoxConjugate(odl.solvers.functional.Functional):
  """Implements the convex conjugate of LinearOnBox."""

  def __init__(self, space, linear_coefficients, lower, upper):
    super(LinearOnBoxConjugate, self).__init__(space=space, linear=False)
    # Confuses the primal and the dual space. Luckily they're the same.
    self.lower = lower
    self.upper = upper
    self.linear_coefficients = linear_coefficients

  @property
  def convex_conj(self):
    """The convex conjugate."""
    return LinearOnBox(self.domain, self.linear_coefficients, self.lower,
                       self.upper)


class LpData(object):
  """Specifies a linear programming problem.

  In the format:
  minimize objective_vector' * x + objective_constant

  s.t. constraint_matrix[:num_equalities, :] * x =
     right_hand_side[:num_equalities]

     constraint_matrix[num_equalities:, :] * x >=
     right_hand_side[num_equalities:, :]

     variable_lower_bound <= x <= variable_upper_bound

  The variable_lower_bound may contain `-inf` elements and variable_upper_bound
  may contain `inf` elements when the corresponding variable bound is not
  present.

  Fields: variable_lower_bound, variable_upper_bound, objective_vector,
  objective_constant, constraint_matrix, right_hand_side, num_equalities.
  """

  def __init__(self, variable_lower_bound, variable_upper_bound,
               objective_vector, objective_constant, constraint_matrix,
               right_hand_side, num_equalities):
    self.variable_lower_bound = variable_lower_bound
    self.variable_upper_bound = variable_upper_bound
    self.objective_vector = objective_vector
    self.objective_constant = objective_constant
    self.constraint_matrix = constraint_matrix
    self.right_hand_side = right_hand_side
    self.num_equalities = num_equalities


def lp_from_hdf5(filename):
  h5 = h5py.File(filename, 'r')
  variable_lower_bound = np.array(h5['variable_lower_bound'])
  variable_upper_bound = np.array(h5['variable_upper_bound'])
  right_hand_side = np.array(h5['right_hand_side'])
  objective_vector = np.array(h5['objective_vector'])
  constraint_matrix = scipy.sparse.csc_matrix(
      (np.array(h5['constraint_matrix_data']),
       np.array(h5['constraint_matrix_indices']),
       np.array(h5['constraint_matrix_indptr'])),
      shape=(right_hand_side.size, variable_lower_bound.size))
  print('constraint matrix dimensions:', constraint_matrix.shape)
  return LpData(variable_lower_bound, variable_upper_bound, objective_vector,
                h5['objective_constant'][()], constraint_matrix,
                right_hand_side, h5['num_equalities'][()])


def solution_stats(lp, primal, dual):
  primal_obj = np.dot(primal, lp.objective_vector) + lp.objective_constant

  # Assumes that bounds on primal and dual variables are always satisfied.

  activity = lp.constraint_matrix @ primal
  eq_error = lp.right_hand_side[:lp.num_equalities] - activity[:lp
                                                               .num_equalities]
  ineq_error = np.maximum(
      lp.right_hand_side[lp.num_equalities:] - activity[lp.num_equalities:],
      0.0)

  reduced_cost = lp.objective_vector - lp.constraint_matrix.T @ dual

  # Whenever there's no lower bound, the positive part of the reduced cost is
  # an infeasibility. Likewise when there's no upper bound, the negative part is
  # an infeasibility.
  reduced_cost_pos = np.maximum(reduced_cost, 0.0)
  reduced_cost_neg = np.maximum(-reduced_cost, 0.0)
  reduced_cost_infeas = np.isinf(
      lp.variable_lower_bound) * reduced_cost_pos + np.isinf(
          lp.variable_upper_bound) * reduced_cost_neg

  finite_lower_bounds = lp.variable_lower_bound.copy()
  finite_lower_bounds[np.isinf(finite_lower_bounds)] = 0.0

  finite_upper_bounds = lp.variable_upper_bound.copy()
  finite_upper_bounds[np.isinf(finite_upper_bounds)] = 0.0

  dual_obj = np.dot(dual, lp.right_hand_side) + np.dot(
      finite_lower_bounds, reduced_cost_pos) + np.dot(
          finite_upper_bounds, reduced_cost_neg) + lp.objective_constant

  kkt_residual = np.concatenate((eq_error, ineq_error, reduced_cost_infeas))
  kkt_residual = np.append(kkt_residual, primal_obj - dual_obj)

  stats = dict()
  stats['primal_obj'] = primal_obj
  stats['dual_obj'] = dual_obj
  stats['kkt_err_l2'] = np.linalg.norm(kkt_residual)
  stats['kkt_err_l1'] = np.linalg.norm(kkt_residual, ord=1)
  stats['kkt_err_linf'] = np.linalg.norm(kkt_residual, ord=np.inf)
  return stats


def num_active_bounds_changed(lp, x_new, y_new, x_prev, y_prev):

  x_prev_at_lower = x_prev == lp.variable_lower_bound
  x_prev_at_upper = x_prev == lp.variable_upper_bound

  x_new_at_lower = x_new == lp.variable_lower_bound
  x_new_at_upper = x_new == lp.variable_upper_bound

  x_active_bounds_changed = np.sum((x_prev_at_lower != x_new_at_lower)
                                   | (x_prev_at_upper != x_new_at_upper))

  y_prev_at_lower = y_prev[lp.num_equalities:] == 0.0
  y_new_at_lower = y_new[lp.num_equalities:] == 0.0

  y_active_bounds_changed = np.sum(y_prev_at_lower != y_new_at_lower)

  return x_active_bounds_changed + y_active_bounds_changed


class CallbackStore(odl.solvers.Callback):

  def __init__(self, lp):
    self.lp = lp
    self.stats = collections.OrderedDict()
    self.iteration_count = 0
    fields = [
        'iteration_num', 'current_primal_obj', 'current_dual_obj',
        'current_kkt_err_l2', 'current_kkt_err_l1', 'current_kkt_err_linf',
        'avg_primal_obj', 'avg_dual_obj', 'avg_kkt_err_l2', 'avg_kkt_err_l1',
        'avg_kkt_err_linf', 'num_active_bounds_changed', 'did_restart'
    ]
    for f in fields:
      self.stats[f] = []

  def __call__(self, x, y, x_avg, y_avg, did_restart):
    self.stats['iteration_num'].append(self.iteration_count)
    self.iteration_count += 1

    stats = solution_stats(self.lp, x, y)
    for stat_name in stats:
      self.stats['current_{}'.format(stat_name)].append(stats[stat_name])

    stats = solution_stats(self.lp, x_avg, y_avg)
    for stat_name in stats:
      self.stats['avg_{}'.format(stat_name)].append(stats[stat_name])

    self.stats['did_restart'].append(did_restart)

    if self.iteration_count == 1:
      self.stats['num_active_bounds_changed'].append(0)
    else:
      self.stats['num_active_bounds_changed'].append(
          num_active_bounds_changed(self.lp, x.asarray(), y.asarray(),
                                    self.x_prev, self.y_prev))
    self.x_prev = x.asarray().copy()
    self.y_prev = y.asarray().copy()

  def dataframe(self):
    return pd.DataFrame.from_dict(self.stats)


def solve_lp(lp, num_iters, tau, sigma, restart, fixed_restart_frequency=None):
  # Using the notation of ODL's primal_dual_hybrid_gradient.py, the LP is
  # formulated as
  # min_x max_y f(x) + y'Lx - g^*(y)
  # where:
  #  f(x) = objective_vector'x +
  #    Indicator([variable_lower_bound, variable_upper_bound])
  #  L = -constraint_matrix
  #  g^*(x) = -right_hand_side'y +
  #    Indicator(R_+^{num_equalities} x R^{num_variables - num_equalities}
  # The objective constant is ignored in the formulation.

  linear_operator = odl.MatrixOperator(-lp.constraint_matrix)

  primal_space = linear_operator.domain
  dual_space = linear_operator.range

  f = LinearOnBox(primal_space, lp.objective_vector, lp.variable_lower_bound,
                  lp.variable_upper_bound)
  num_constraints = lp.constraint_matrix.shape[0]
  g = LinearOnBoxConjugate(
      dual_space, -lp.right_hand_side,
      np.concatenate(
          (np.full(lp.num_equalities,
                   -np.inf), np.zeros(num_constraints - lp.num_equalities))),
      np.full(num_constraints, np.inf))

  x = primal_space.zero()
  y = dual_space.zero()
  callback = CallbackStore(lp)
  restarted_pdhg.restarted_pdhg(
      x,
      f=f,
      g=g,
      L=linear_operator,
      niter=num_iters,
      y=y,
      tau=tau,
      sigma=sigma,
      callback=callback,
      restart=restart,
      fixed_restart_frequency=fixed_restart_frequency)
  return callback.dataframe()


def step_sizes(lp, tau_sigma_ratio):
  estimated_norm = odl.MatrixOperator(-lp.constraint_matrix).norm(estimate=True)
  # tau * sigma = 0.9 / estimated_norm**2
  # and tau/sigma = scale
  sigma = np.sqrt(0.9 / tau_sigma_ratio) / estimated_norm
  tau = sigma * tau_sigma_ratio
  return tau, sigma


def solve_lps(lp,
              num_iters,
              tau_sigma_ratio,
              restart_frequencies,
              solve_adaptive=True):

  tau, sigma = step_sizes(lp, tau_sigma_ratio)

  iteration_data = collections.OrderedDict()

  name = 'pdhg'
  iteration_data[name] = solve_lp(lp, num_iters, tau, sigma, restart='none')
  print(name, '\n', iteration_data[name].tail(2))

  if solve_adaptive:
    name = 'pdhg_adaptive'
    iteration_data[name] = solve_lp(
        lp, num_iters, tau, sigma, restart='adaptive')
    print(name, '\n', iteration_data[name].tail(2))

  for restart_frequency in restart_frequencies:
    name = 'pdhg_restart_{}'.format(restart_frequency)
    iteration_data[name] = solve_lp(
        lp,
        num_iters,
        tau,
        sigma,
        restart='fixed',
        fixed_restart_frequency=restart_frequency)
    print(name, '\n', iteration_data[name].tail(2))
  return iteration_data


# These example LPs are useful for testing the code.
def trivial_lp():
  # min -2x - y
  # s.t. -x - y >= -1
  # x, y >= 0.
  return LpData(
      variable_lower_bound=np.zeros(2),
      variable_upper_bound=np.full(2, np.inf),
      objective_vector=np.array([-2.0, -1.0]),
      objective_constant=0.0,
      constraint_matrix=scipy.sparse.csc_matrix([[-1.0, -1.0]]),
      right_hand_side=np.array([-1.0]),
      num_equalities=0)


def trivial_lp2():
  # min -x
  # s.t. x - y == 0.5
  # x free
  # y in [0, 1]
  return LpData(
      variable_lower_bound=np.array([-np.inf, 0.0]),
      variable_upper_bound=np.array([np.inf, 1.0]),
      objective_vector=np.array([-1.0, 0.0]),
      objective_constant=0.0,
      constraint_matrix=scipy.sparse.csc_matrix([[1.0, -1.0]]),
      right_hand_side=np.array([0.5]),
      num_equalities=1)
