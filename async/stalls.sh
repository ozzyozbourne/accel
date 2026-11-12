#!/usr/bin/env bash
# Prints why warps stall (Nsight Compute) for one kernel launch of a binary.
# Usage: stalls.sh <binary> <launch-skip>
M=smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
for r in short_scoreboard mio_throttle lg_throttle wait math_pipe_throttle not_selected selected barrier; do
  M=$M,smsp__average_warps_issue_stalled_${r}_per_issue_active.ratio
done
M=$M,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_sector_hit_rate.pct,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
ncu --metrics "$M" --launch-skip "$2" -c 1 "$1" 2>&1 | grep -E "^ +(void|smsp__|sm__|l1tex)" |
  sed 's/smsp__average_warps_issue_stalled_//; s/_per_issue_active.ratio//' | awk '{print "  " $1, $NF}'
