# This file is derived from ODL's primal_dual_hybrid_gradient.py, i.e.,
# https://github.com/odlgroup/odl/blob/2320e398bcbb96cdf548d0f08b177a54d8ab7e7e/odl/solvers/nonsmooth/primal_dual_hybrid_gradient.py
# Copyright 2020 Google LLC
# Copyright 2014-2019 The ODL contributors
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.
"""An implementation of PDHG with adaptive and fixed restarts."""

import numpy as np
import odl
from odl.operator import Operator


# pylint: disable=invalid-name
# pylint: disable=invalid-unary-operand-type
def restarted_pdhg(x, f, g, L, niter, tau=None, sigma=None, **kwargs):
  """Primal-dual hybrid gradient algorithm for convex optimization.

  Restarted version.

  Args:
    x: ``L.domain`` element. Starting point of the iteration, updated in-place.
    f: `Functional`. The function ``f`` in the problem definition. Needs to have
      ``f.proximal``.
    g: `Functional`. The function ``g`` in the problem definition. Needs to have
      ``g.convex_conj.proximal``.
    L: linear `Operator`. The linear operator that should be applied before
      ``g``. Its range must match the domain of ``g`` and its domain must match
      the domain of ``f``.
    niter: non-negative int. Number of iterations.
    tau: float, optional. Step size parameter for ``g``. Default is Sufficient
      for convergence, see `pdhg_stepsize`.
    sigma: sequence of floats, optional. Step size parameters for ``f``. Default
      is Sufficient for convergence, see `pdhg_stepsize`.
    **kwargs: callback - callable, optional Function called on each iteration
      with the current primal/dual iterates, the current average primal/dual
      iterates, and whether a restart was performed on this iteration. theta -
      float, optional Relaxation parameter, required to fulfill ``0 <= theta <=
      1``. Default is 1 y - ``op.range`` element, optional Required to resume
      iteration. For ``None``, ``op.range.zero()`` is used. Default is ``None``
      restart - "fixed" or "adaptive" fixed_restart_frequency - The fixed
      restart frequency, if applicable.
  """
  # Forward operator
  if not isinstance(L, Operator):
    raise TypeError('`op` {!r} is not an `Operator` instance' ''.format(L))

  # Starting point
  if x not in L.domain:
    raise TypeError('`x` {!r} is not in the domain of `op` {!r}'
                    ''.format(x, L.domain))

  # Spaces
  if f.domain != L.domain:
    raise TypeError('`f.domain` {!r} must equal `op.domain` {!r}'
                    ''.format(f.domain, L.domain))

  # Step size parameters
  tau, sigma = odl.solvers.pdhg_stepsize(L, tau, sigma)

  # Number of iterations
  if not isinstance(niter, int) or niter < 0:
    raise ValueError('`niter` {} not understood' ''.format(niter))

  # Relaxation parameter
  theta = kwargs.pop('theta', 1)
  theta, theta_in = float(theta), theta
  if not 0 <= theta <= 1:
    raise ValueError('`theta` {} not in [0, 1]' ''.format(theta_in))

  # Acceleration parameters
  gamma_primal = kwargs.pop('gamma_primal', None)
  if gamma_primal is not None:
    raise ValueError('gamma_primal not supported')

  gamma_dual = kwargs.pop('gamma_dual', None)
  if gamma_dual is not None:
    raise ValueError('gamma_dual not supported')

  # Callback object
  callback = kwargs.pop('callback', None)
  if callback is not None and not callable(callback):
    raise TypeError('`callback` {} is not callable' ''.format(callback))

  # Initialize the relaxation variable
  if kwargs.pop('x_relax', None) is not None:
    raise ValueError('x_relax argument not supported')
  x_relax = x.copy()

  # Initialize the dual variable
  y = kwargs.pop('y', None)
  if y is None:
    y = L.range.zero()
  elif y not in L.range:
    raise TypeError('`y` {} is not in the range of `L` '
                    '{}'.format(y.space, L.range))

  restart = kwargs.pop('restart', '')
  if restart != 'fixed' and restart != 'adaptive' and restart != 'none':
    raise ValueError('Invalid setting for restart')

  fixed_restart_frequency = kwargs.pop('fixed_restart_frequency', None)
  if restart == 'fixed' and fixed_restart_frequency is None:
    raise ValueError('Missing restart_frequency')

  # Get the proximals
  proximal_primal = f.proximal
  proximal_dual = g.convex_conj.proximal
  # Pre-compute proximals for efficiency
  proximal_dual_sigma = proximal_dual(sigma)
  proximal_primal_tau = proximal_primal(tau)

  # Temporary copy to store previous iterate
  x_old = x.space.element()

  # Temporaries
  dual_tmp = L.range.element()
  primal_tmp = L.domain.element()

  x_sum = L.domain.zero()
  y_sum = L.range.zero()
  iterations_since_last_restart = 0

  # This initial value forces a restart after the first iteration.
  last_potential_value = np.inf
  last_restart_x = x.copy()
  last_restart_y = y.copy()

  for it in range(niter):
    do_restart = False

    if iterations_since_last_restart > 0:  # Consider a restart
      if restart == 'fixed' and it % fixed_restart_frequency == 0:
        do_restart = True
      elif restart == 'adaptive':
        # This is the adaptive scheme in the paper, i.e., equation (1).
        candidate_new_x = x_sum / iterations_since_last_restart
        candidate_new_y = y_sum / iterations_since_last_restart
        distance_since_last_restart = np.sqrt(
            (1 / tau) * np.linalg.norm(candidate_new_x.asarray() -
                                       last_restart_x.asarray())**2 +
            (1 / sigma) * np.linalg.norm(candidate_new_y.asarray() -
                                         last_restart_y.asarray())**2)
        if distance_since_last_restart / iterations_since_last_restart < 0.5 * last_potential_value:
          do_restart = True
          last_restart_x = candidate_new_x
          last_restart_y = candidate_new_y
          last_potential_value = distance_since_last_restart / iterations_since_last_restart

    if do_restart:
      # Reset to average
      x.assign(x_sum / iterations_since_last_restart)
      y.assign(y_sum / iterations_since_last_restart)
      x_relax = x.copy()
      x_sum = L.domain.zero()
      y_sum = L.range.zero()
      iterations_since_last_restart = 0

    # Copy required for relaxation
    x_old.assign(x)

    # Gradient ascent in the dual variable y
    # Compute dual_tmp = y + sigma * L(x_relax)
    L(x_relax, out=dual_tmp)
    dual_tmp.lincomb(1, y, sigma, dual_tmp)

    # Apply the dual proximal
    proximal_dual_sigma(dual_tmp, out=y)

    # Gradient descent in the primal variable x
    # Compute primal_tmp = x + (- tau) * L.derivative(x).adjoint(y)
    L.derivative(x).adjoint(y, out=primal_tmp)
    primal_tmp.lincomb(1, x, -tau, primal_tmp)

    # Apply the primal proximal
    proximal_primal_tau(primal_tmp, out=x)

    # Over-relaxation in the primal variable x
    x_relax.lincomb(1 + theta, x, -theta, x_old)

    x_sum += x
    y_sum += y
    iterations_since_last_restart += 1

    if callback is not None:
      callback(x, y, x_sum / iterations_since_last_restart,
               y_sum / iterations_since_last_restart, do_restart)
