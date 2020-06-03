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

# usage: plot_matrix_games_results.py [results directory]

using CSV
using DataFrames
using Plots
using Plots.PlotMeasures

gr()

fntsm = Plots.font("serif", pointsize=12)
fntlg = Plots.font("serif", pointsize=18)
default(titlefont=fntlg, guidefont=fntlg, tickfont=fntsm, legendfont=fntsm)

function do_plot(csv_path, threshold_for_best_fixed, y_axis_range)

  results = DataFrame(CSV.File(csv_path))
  restart_frequencies = [8, 32, 128, 512, 2048]  # Must match up with pdhg_matrix_games.py.

  # Find the fixed restart frequency for which median_residual_at_current first
  # drops below threshold_for_best_fixed.
  best_restart_frequency = nothing
  earliest_time = typemax(Int32)
  for restart_frequency in restart_frequencies
    filtered = filter(row -> row["method"] == "pdhg restart $(restart_frequency)", results)
    indices = findall(filtered[!, "median_residual_at_current"] .<= threshold_for_best_fixed)
    if length(indices) > 0 && indices[1] < earliest_time
      earliest_time = indices[1]
      best_restart_frequency = restart_frequency
    end
  end
  @show best_restart_frequency

  filtered = filter(row -> row["method"] == "pdhg", results)
  plot(filtered[!, "iteration_num"], filtered[!, "median_residual_at_current"],
       ribbon=(filtered[!, "median_residual_at_current"] .- filtered[!, "lower_range_at_current"], filtered[!, "upper_range_at_current"] .- filtered[!, "median_residual_at_current"]),
       label="No restarts", line=(1,:solid), right_margin = 4mm)

  filtered = filter(row -> row["method"] == "pdhg adaptive", results)
  plot!(filtered[!, "iteration_num"], filtered[!, "median_residual_at_avg"],
       ribbon=(filtered[!, "median_residual_at_avg"] .- filtered[!, "lower_range_at_avg"], filtered[!, "upper_range_at_avg"] .- filtered[!, "median_residual_at_avg"]),
       label="Our adaptive", line=(1,:solid))

  filtered = filter(row -> row["method"] == "pdhg restart $(best_restart_frequency)", results)
  plot!(filtered[!, "iteration_num"], filtered[!, "median_residual_at_avg"],
       ribbon=(filtered[!, "median_residual_at_avg"] .- filtered[!, "lower_range_at_avg"], filtered[!, "upper_range_at_avg"] .- filtered[!, "median_residual_at_avg"]),
       label="Best fixed ($(best_restart_frequency))", line=(1,:solid))

  # This plots all the fixed restarts.
  # for restart in restart_frequencies
  #   filtered = filter(row -> row["method"] == "pdhg restart $(restart)", results)
  #   plot!(filtered[!, "iteration_num"], filtered[!, "median_residual_at_avg"],
  #      ribbon=(filtered[!, "median_residual_at_avg"] .- filtered[!, "lower_range_at_avg"], filtered[!, "upper_range_at_avg"] .- filtered[!, "median_residual_at_avg"]),
  #      label="Fixed Restart ($(restart))", line=(1,:solid))
  # end

  yaxis!("Residual", :log10)
  if y_axis_range !== nothing
    ylims!(y_axis_range)
  end
  xaxis!("Iteration")
  savefig(replace(csv_path, ".csv" => ".pdf"))
end

do_plot(joinpath(ARGS[1], "uniform_game_100_100.csv"), 1e-4, nothing)
do_plot(joinpath(ARGS[1], "normal_game_100_100.csv"), 1e-6, (1e-10, 1.0))
