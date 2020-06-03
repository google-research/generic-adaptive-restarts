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

"""Script to solve an LP under different algorithmic configurations."""

# Usage: lp_experiment.py [hdf file] [output dir] [tau sigma ratio] [num iters]

import os
import sys
from . import pdhg_linear_programming

OUTPUT_DIR = sys.argv[2]
TAU_SIGMA_RATIO = float(sys.argv[3])  # see lp_calibrate_tau_sigma_ratio.py
NUM_ITERS = int(sys.argv[4])
os.makedirs(OUTPUT_DIR, exist_ok=True)

iteration_data = pdhg_linear_programming.solve_lps(
    pdhg_linear_programming.lp_from_hdf5(sys.argv[1]),
    num_iters=NUM_ITERS,
    tau_sigma_ratio=TAU_SIGMA_RATIO,
    restart_frequencies=[64, 256, 1024, 4096, 16384, 65536])


for name in iteration_data:
  file_name = os.path.join(OUTPUT_DIR, '{}.csv'.format(name))
  iteration_data[name].to_csv(file_name)

