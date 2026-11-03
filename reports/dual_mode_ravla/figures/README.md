# Dual-mode RA-VLA result figures

Use PNG for PowerPoint; SVG/PDF retain vector text and graphics (video frames remain raster).
All figures are generated from saved artifacts, with source hashes in figure_manifest.json.

| File stem | Content |
|---|---|
| [01_main_results](01_main_results.png) | 100-rollout success and goal-mode usage |
| [02_checkpoint_sweep](02_checkpoint_sweep.png) | 50-rollout checkpoint comparison; separate cohort |
| [03_goal_burst_outcomes](03_goal_burst_outcomes.png) | Support exits, cap exits and episode endings; all-burst denominator |
| [04_goal_sensitivity](04_goal_sensitivity.png) | 200-state median/IQR; common Gaussian source |
| [05_switching_example](05_switching_example.png) | Selected task 0/init 0: capped success vs until-q80 timeout |
| [06_per_task_success](06_per_task_success.png) | Per-task success counts; task labels in task_key.csv |
| [07_rollout_frames](07_rollout_frames.png) | Existing videos at matched control-step timestamps |

Cohorts must not be pooled across the main evaluation and checkpoint sweep.
Support threshold exits are not verified semantic recovery events.
The selected successful capped example is not representative of its aggregate success rate.
