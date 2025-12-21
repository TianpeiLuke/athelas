"""
Callback to update TrainingState during LightGBM training.

Replicates legacy state tracking behavior from customLossNoKD/customLossKDswap.
"""

from typing import Any
import numpy as np


class TrainingStateCallback:
    """
    LightGBM callback that updates TrainingState to match legacy behavior.

    Legacy mapping:
    - curr_obj_round -> current_iteration
    - curr_eval_round -> (not tracked separately, same as obj)
    - eval_mat -> per_task_metrics (converted to dict format)
    - w_trn_mat -> weight_evolution

    Design Pattern: Observer pattern - observes training and updates state
    """

    def __init__(self, training_state, loss_function):
        """
        Initialize callback.

        Parameters
        ----------
        training_state : TrainingState
            Training state object to update
        loss_function : BaseLossFunction
            Loss function containing weight history
        """
        self.training_state = training_state
        self.loss_function = loss_function
        self.order = 0  # Execute first among callbacks

        # Track last seen iteration to avoid duplicates
        self.last_iteration = -1

    def __call__(self, env):
        """
        Update training state at each iteration.

        Called AFTER each training iteration (after objective + evaluation).
        Matches legacy update timing in customLossNoKD.self_obj() and self_eval().

        Parameters
        ----------
        env : CallbackEnv
            LightGBM callback environment with:
            - model: Booster
            - params: dict
            - iteration: int (current iteration, 0-indexed)
            - begin_iteration: int
            - end_iteration: int
            - evaluation_result_list: list of tuples
        """
        # Skip if already processed this iteration
        if env.iteration == self.last_iteration:
            return

        self.last_iteration = env.iteration

        # 1. Update iteration counter (matches curr_obj_round/curr_eval_round)
        # Legacy increments counters in both objective and eval
        # We track once per iteration since they happen together
        self.training_state.current_iteration = env.iteration

        # 2. Store weight history (matches w_trn_mat)
        # Legacy: self.w_trn_mat.append(w) in self_obj()
        # Pull from loss function's weight_history
        if hasattr(self.loss_function, "weight_history"):
            # weight_history is updated in loss.objective()
            # Should have env.iteration + 1 entries (0-indexed)
            if len(self.loss_function.weight_history) > len(
                self.training_state.weight_evolution
            ):
                # Add newest weight vector
                latest_weights = self.loss_function.weight_history[-1]
                self.training_state.weight_evolution.append(latest_weights.copy())

        # 3. Store per-task evaluation metrics (matches eval_mat)
        # Legacy: self.eval_mat.append(weighted_score_vec) in self_eval()
        if env.evaluation_result_list:
            # Legacy eval_mat format: [weighted_score_task0, weighted_score_task1, ...]
            # We store unweighted scores + metadata for more flexibility

            metrics_dict = {"iteration": env.iteration, "scores": {}}

            # Parse evaluation results
            # Format: [(dataset_name, metric_name, score, is_higher_better), ...]
            for (
                dataset_name,
                metric_name,
                score,
                is_higher_better,
            ) in env.evaluation_result_list:
                key = f"{dataset_name}_{metric_name}"
                metrics_dict["scores"][key] = {
                    "value": float(score),
                    "is_higher_better": is_higher_better,
                }

            self.training_state.per_task_metrics.append(metrics_dict)

            # NEW: Store raw per-task AUC scores (matches legacy eval_mat)
            # Loss function stores raw scores in last_raw_scores attribute
            if (
                hasattr(self.loss_function, "last_raw_scores")
                and self.loss_function.last_raw_scores is not None
            ):
                raw_scores = self.loss_function.last_raw_scores
                self.training_state.raw_task_auc.append(raw_scores)

                # Log for debugging (matches legacy print statement)
                import logging

                logger = logging.getLogger(__name__)
                logger.info(
                    f"--- raw task AUC scores: {[round(s, 4) for s in raw_scores]}"
                )

            # 4. Track best performance (legacy didn't do this explicitly)
            # Use first metric as primary (typically mean_auc)
            if len(env.evaluation_result_list) > 0:
                _, _, score, is_higher_better = env.evaluation_result_list[0]
                improved = self.training_state.update_best(
                    metric=float(score), epoch=env.iteration, iteration=env.iteration
                )

                if not improved:
                    self.training_state.epochs_without_improvement += 1

        # 5. Update KD state (if applicable)
        # Legacy customLossKDswap tracks: self.replaced, self.kd_active
        if hasattr(self.loss_function, "kd_active"):
            was_active = self.training_state.kd_active
            self.training_state.kd_active = self.loss_function.kd_active

            # Track when KD was first triggered
            if not was_active and self.loss_function.kd_active:
                self.training_state.kd_trigger_epoch = env.iteration

        # 6. Update KD replacement tracking (for KD loss only)
        if hasattr(self.loss_function, "replaced"):
            # Handle both dict and numpy array formats for replaced attribute
            # Legacy KD uses numpy array, but we support both for flexibility
            if isinstance(self.loss_function.replaced, dict):
                # Dictionary format: {task_id: bool}
                replaced_tasks = [
                    task_id
                    for task_id, is_replaced in self.loss_function.replaced.items()
                    if is_replaced
                ]
            else:
                # Numpy array format (legacy KD): [bool, bool, ...]
                replaced_tasks = [
                    task_id
                    for task_id, is_replaced in enumerate(self.loss_function.replaced)
                    if is_replaced
                ]

            if replaced_tasks:
                # Store in training_state for persistence
                # Note: This is an enhancement over legacy (which didn't persist this)
                if not hasattr(self.training_state, "_kd_replaced_tasks"):
                    self.training_state._kd_replaced_tasks = replaced_tasks
                else:
                    self.training_state._kd_replaced_tasks = replaced_tasks
