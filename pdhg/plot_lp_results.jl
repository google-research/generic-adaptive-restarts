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

# usage: plot_lp_results.py [results directory]

using CSV
using DataFrames
using Plots

# If set to false, all restart frequencies are plotted.
PLOT_ONLY_BEST_FIXED_RESTART = true

gr()

fntsm = Plots.font("serif", pointsize=12)
fntlg = Plots.font("serif", pointsize=18)
default(titlefont=fntlg, guidefont=fntlg, tickfont=fntsm, legendfont=fntsm)

pdhg = DataFrame(CSV.File(joinpath(ARGS[1],"pdhg.csv")))
pdhg_adaptive = DataFrame(CSV.File(joinpath(ARGS[1],"pdhg_adaptive.csv")))
pdhg_restart = Dict()

last_active_set_update = findall(pdhg_adaptive[!, "num_active_bounds_changed"] .> 0)[end]
@show last_active_set_update

# This list should match up with lp_experiment.py.
restart_frequencies = [64, 256, 1024, 4096, 16384, 65536]
for restart_frequency in restart_frequencies
  filename = "pdhg_restart_$(restart_frequency).csv"
  pdhg_restart[restart_frequency] = DataFrame(CSV.File(joinpath(ARGS[1], filename)))
end

iter_num = pdhg[!, "iteration_num"]
feature_names = [("current_kkt_err_l2", "L2 norm of KKT residual (current iterate)"),
                 ("current_kkt_err_linf", "LInf norm of KKT residual (current iterate)"),
                 ("avg_kkt_err_l2", "L2 norm of KKT residual (averaged iterate)"),
                 ("avg_kkt_err_linf", "LInf norm of KKT residual (averaged iterate)")]

# Find the fixed restart frequency for which current_kkt_err_l2 feature first
# drops below 1e-6.
best_restart_frequency = nothing
earliest_time = typemax(Int32)
for restart_frequency in restart_frequencies
  indices = findall(pdhg_restart[restart_frequency][!, "current_kkt_err_l2"] .<= 1e-6)
  if length(indices) > 0 && indices[1] < earliest_time
    global earliest_time = indices[1]
    global best_restart_frequency = restart_frequency
  end
end

for (feature, name) in feature_names
  plot(iter_num, pdhg[!, feature], label="pdhg", line=(1,:solid))
  plot!(iter_num, pdhg_adaptive[!, feature], label="pdhg adaptive",
    line = (1,:solid))

  if PLOT_ONLY_BEST_FIXED_RESTART
    @assert best_restart_frequency !== nothing "No best fixed restart frequency"
    plot!(iter_num, pdhg_restart[best_restart_frequency][!, feature],
          label="best fixed restart ($(best_restart_frequency))",
          line=(1,:solid))
  else
    for restart_frequency in restart_frequencies
      plot!(iter_num, pdhg_restart[restart_frequency][!, feature],
            label="pdhg restart $(restart_frequency)",
            line=(1,:solid))
    end
  end

  scatter!([last_active_set_update], [pdhg_adaptive[last_active_set_update, feature]], label="last active set update", markershape=:star6, markersize=6)

  yaxis!(name, :log10)
  xaxis!("Iteration")
  y_pow_10_range = -9:2
  ylims!((10.0^y_pow_10_range[1], 10.0^y_pow_10_range[end]))
  yticks!(10.0.^y_pow_10_range[1:2:end], ["10^$i" for i in y_pow_10_range[1:2:end]])
  savefig(joinpath(ARGS[1], "$(feature).pdf"))
end

# A custom summary plot with choice of average/current for different methods.
plot(iter_num, pdhg[!, "current_kkt_err_l2"], label="No restarts", line=(1,:solid))
plot!(iter_num, pdhg_adaptive[!, "avg_kkt_err_l2"], label="Our adaptive",
    line = (1,:solid))
plot!(iter_num, pdhg_restart[best_restart_frequency][!, "avg_kkt_err_l2"],
          label="Best fixed ($(best_restart_frequency))",
          line=(1,:solid))
did_restart_indices = findall(pdhg_adaptive[!, "did_restart"] .== 1)
scatter!(did_restart_indices, pdhg_adaptive[did_restart_indices, "avg_kkt_err_l2"], label="", color=2)
did_restart_indices = findall(pdhg_restart[best_restart_frequency][!, "did_restart"] .== 1)
scatter!(did_restart_indices, pdhg_restart[best_restart_frequency][did_restart_indices, "avg_kkt_err_l2"], label="", color=3)
scatter!([last_active_set_update], [pdhg_adaptive[last_active_set_update, "avg_kkt_err_l2"]], label="Last active set change", markershape=:star6, markersize=6)
yaxis!("Residual", :log10)
xaxis!("Iteration")
y_pow_10_range = -9:2
ylims!((10.0^y_pow_10_range[1], 10.0^y_pow_10_range[end]))
yticks!(10.0.^y_pow_10_range[1:2:end], ["10^$i" for i in y_pow_10_range[1:2:end]])
savefig(joinpath(ARGS[1], "summary.pdf"))
