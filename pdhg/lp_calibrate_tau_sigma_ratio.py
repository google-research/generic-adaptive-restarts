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

"""Script to tune the tau/sigma ratio."""

import sys
from . import pdhg_linear_programming
import pandas as pd


NUM_ITERS = 1000
INSTANCE = sys.argv[1]
lp = pdhg_linear_programming.lp_from_hdf5(INSTANCE)
for ratio in [
    0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0,
    100000.0
]:
  tau, sigma = pdhg_linear_programming.step_sizes(lp, ratio)
  data = pdhg_linear_programming.solve_lp(
      lp, NUM_ITERS, tau, sigma, restart='none')
  print('ratio:', ratio)
  with pd.option_context('display.max_columns', None):
    print(data.tail(1)[['current_kkt_err_l2']])

# best for qap15: 0.0001
# best for nug08-3rd: 0.01
