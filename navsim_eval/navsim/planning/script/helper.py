# helper.py
import numpy as np
import os

from navsim.evaluate.pdm_score import pdm_score
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory, Cameras, Camera, EgoStatus
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.enums import SceneFrameType
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy
from navsim.planning.simulation.planner.pdm_planner.scoring.scene_aggregator import SceneAggregator

from hydra.utils import instantiate
from typing import List, Dict, Tuple

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def infer_start_adjacent_mapping(score_df: pd.DataFrame, time_gap_threshold: float = 0.55) -> Dict[str, str]:
    """
    Infers an adjacent mapping from the score_df DataFrame by start time.
    Each current-token is mapped to its previous-token if they are adjacent.
    Used to create the two-frame extended comfort score (reversed direction).

    :param score_df: DataFrame containing at least 'token', 'log_name', 'start_time'.
    :param time_gap_threshold: Maximum allowed gap (in seconds) between two frames to
                               consider them "adjacent".
    :return: Dictionary mapping each current-token to one previous-token.
    """
    adjacent_mapping: Dict[str, str] = {}

    for log_name, group_df in score_df[score_df["frame_type"] == SceneFrameType.ORIGINAL].groupby("log_name"):
        group_df = group_df.sort_values(by="start_time").reset_index(drop=True)
        logger.info(group_df)

        for i in range(1, len(group_df)):
            prev_row = group_df.iloc[i - 1]
            current_row = group_df.iloc[i]

            prev_token = prev_row["token"]
            current_token = current_row["token"]
            time_diff = current_row["start_time"] - prev_row["start_time"]

            if abs(time_diff) <= time_gap_threshold:
                adjacent_mapping[current_token] = prev_token

    return adjacent_mapping


def compute_final_scores(pdm_score_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute final scores for each row in pdm_score_df after updating
    the weighted metrics with two-frame extended comfort.

    If 'two_frame_extended_comfort' is NaN for a row, the corresponding
    metric and its weight are set to zero, effectively ignoring it
    during normalization.

    :param pdm_score_df: DataFrame containing PDM scores and metrics.
    :return: A new DataFrame with the computed final scores.
    """
    df = pdm_score_df.copy()

    two_frame_scores = df["two_frame_extended_comfort"].to_numpy()  # shape: (N, )
    weighted_metrics = np.stack(df["weighted_metrics"].to_numpy())  # shape: (N, M)
    weighted_metrics_array = np.stack(df["weighted_metrics_array"].to_numpy())  # shape: (N, M)

    mask = np.isnan(two_frame_scores)
    two_frame_idx = WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT

    weighted_metrics[mask, two_frame_idx] = 0.0
    weighted_metrics_array[mask, two_frame_idx] = 0.0

    non_mask = ~mask
    weighted_metrics[non_mask, two_frame_idx] = two_frame_scores[non_mask]

    weighted_sum = (weighted_metrics * weighted_metrics_array).sum(axis=1)
    total_weight = weighted_metrics_array.sum(axis=1)
    total_weight[total_weight == 0.0] = np.nan
    weighted_metric_scores = weighted_sum / total_weight

    df["score"] = df["multiplicative_metrics_prod"].to_numpy() * weighted_metric_scores
    df.drop(
        columns=["weighted_metrics", "weighted_metrics_array", "multiplicative_metrics_prod"],
        inplace=True,
    )

    return df


def create_scene_aggregators(
    all_mappings: Dict[str, str],
    full_score_df: pd.DataFrame,
    proposal_sampling: TrajectorySampling,
) -> pd.DataFrame:

    full_score_df["two_frame_extended_comfort"] = np.nan
    full_score_df = full_score_df.set_index("token")

    all_updates = []

    for now_frame, previous_frame in all_mappings.items():
        aggregator = SceneAggregator(
            now_frame=now_frame,
            previous_frame=previous_frame,
            score_df=full_score_df,
            proposal_sampling=proposal_sampling,
        )
        updated_rows = aggregator.aggregate_scores(one_stage_only=True)
        logger.info(now_frame)
        logger.info(previous_frame)
        logger.info(updated_rows)
        all_updates.append(updated_rows)

    all_updates_df = pd.concat(all_updates, ignore_index=True).set_index("token")
    full_score_df.update(all_updates_df)
    full_score_df.reset_index(inplace=True)
    full_score_df = full_score_df.drop(columns=["ego_simulated_states"])

    return full_score_df

def score_single_trajectory_epdms(cfg: DictConfig,
                                metric_cache_loader: MetricCacheLoader,
                                token: str,
                                poses: List[List[float]],
                                verbose: bool):
    """
    EPDMS cannot score a single trajectory in isolation because
    infer_start_adjacent_mapping requires the previous token.
    """
    logger.info(f"[PID {os.getpid()}] scoring token={token}")
    print(f"[PID {os.getpid()}] scoring token={token}")
    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    # metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    if cfg.traffic_agents == "non_reactive":
        traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
            cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling
        )
    elif cfg.traffic_agents == "reactive":
        traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
            cfg.traffic_agents_policy.reactive, simulator.proposal_sampling
        )

    metric_cache = metric_cache_loader.get_from_token(token)

    trajectory = Trajectory(poses=np.array(poses))

    score_row, ego_simulated_states = pdm_score(
        metric_cache=metric_cache,
        model_trajectory=trajectory,
        future_sampling=simulator.proposal_sampling,
        simulator=simulator,
        scorer=scorer,
        traffic_agents_policy=traffic_agents_policy,
    )
    score_row["valid"] = True
    score_row["log_name"] = metric_cache.log_name
    score_row["frame_type"] = metric_cache.scene_type
    score_row["start_time"] = metric_cache.timepoint.time_s
    end_pose = StateSE2(
        x=trajectory.poses[-1, 0],
        y=trajectory.poses[-1, 1],
        heading=trajectory.poses[-1, 2],
    )
    absolute_endpoint = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [end_pose])[0]
    score_row["endpoint_x"] = absolute_endpoint.x
    score_row["endpoint_y"] = absolute_endpoint.y
    score_row["start_point_x"] = metric_cache.ego_state.rear_axle.x
    score_row["start_point_y"] = metric_cache.ego_state.rear_axle.y
    score_row["ego_simulated_states"] = [ego_simulated_states]  # used for two-frames extended comfort
    score_row["token"] = token

    pdm_score_df = pd.concat([score_row])
    logger.info(pdm_score_df)
    logger.info(pdm_score_df["frame_type"])
    logger.info(pdm_score_df["log_name"])
    logger.info(pdm_score_df["start_time"])

    start_adjacent_mapping = infer_start_adjacent_mapping(pdm_score_df)
    logger.info(start_adjacent_mapping)
    print(start_adjacent_mapping)
    pdm_score_df = create_scene_aggregators(
        start_adjacent_mapping, pdm_score_df, instantiate(cfg.simulator.proposal_sampling)
    )
    pdm_score_df = compute_final_scores(pdm_score_df)


    # ----------------------
    score_dict = pdm_score_df.to_dict(orient="records")[0]
    score_dict["epdms"] = score_dict["pdm_score"]
    score_dict["pdms"] = score_dict["no_at_fault_collisions"] * score_dict["drivable_area_compliance"] * \
        (5 * score_dict["ego_progress"] + 5 * score_dict["time_to_collision_within_bound"] +
         2 * score_dict["history_comfort"]) / 12
    # logger.info(score_dict)
    if verbose:
        score_dict["token"] = token
        score_dict["visual"] = ""
    return score_dict


def score_single_trajectory(cfg: DictConfig,
                                metric_cache_loader: MetricCacheLoader,
                                token: str,
                                poses: List[List[float]],
                                verbose: bool):
    """
    Score a single trajectory.
    :param cfg: hydra cfg
    :param token: scene token
    :param poses: N x 3 array (x, y, heading)
    :return: (metrics_dict)
    """
    logger.info(f"[PID {os.getpid()}] scoring token={token}")
    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    # metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    if cfg.traffic_agents == "non_reactive":
        traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
            cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling
        )
    elif cfg.traffic_agents == "reactive":
        traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
            cfg.traffic_agents_policy.reactive, simulator.proposal_sampling
        )

    metric_cache = metric_cache_loader.get_from_token(token)

    trajectory = Trajectory(poses=np.array(poses))

    score_row, ego_simulated_states = pdm_score(
        metric_cache=metric_cache,
        model_trajectory=trajectory,
        future_sampling=simulator.proposal_sampling,
        simulator=simulator,
        scorer=scorer,
        traffic_agents_policy=traffic_agents_policy,
    )

    # ----------------------
    score_dict = score_row.to_dict(orient="records")[0]
    score_dict["pdms"] = calculate_pdms(score_dict)
    # score_dict["pdms_scaled"] = calculate_sharp_scaled_pdms(score_dict)
    # score_dict["pdms"] = score_dict["pdm_score"]
    score_dict["pdms_scaled"] = calculate_scaled_pdms(score_dict)
    # logger.info(score_dict) 
    if verbose:
        score_dict["token"] = token
        score_dict["visual"] = ""
    return score_dict
    
def score_same_token_trajectory(cfg: DictConfig,
                                metric_cache_loader: MetricCacheLoader,
                                token: str,
                                poses: List[List[List[float]]],
                                verbose: bool):
    """
    Score trajectories sharing the same token.
    :param cfg: hydra cfg
    :param token: scene token
    :param poses: B x N x 3 array (x, y, heading)
    :return: List[metrics_dict]
    """
    logger.info(f"[PID {os.getpid()}] scoring start token={token}")
    metric_cache = metric_cache_loader.get_from_token(token)
    
    # Iterate over poses in the batch; simulator and scorer cannot be reused (thread-safety)
    results = []
    for pose in poses:
        simulator: PDMSimulator = instantiate(cfg.simulator)
        scorer: PDMScorer = instantiate(cfg.scorer)
        # metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
        if cfg.traffic_agents == "non_reactive":
            traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
            cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling
        )
        elif cfg.traffic_agents == "reactive":
            traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
                cfg.traffic_agents_policy.reactive, simulator.proposal_sampling
            )

        trajectory = Trajectory(poses=np.array(pose))

        score_row, ego_simulated_states = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=trajectory,
            future_sampling=simulator.proposal_sampling,
            simulator=simulator,
            scorer=scorer,
            traffic_agents_policy=traffic_agents_policy,
        )

        # ----------------------
        score_dict = score_row.to_dict(orient="records")[0]
        score_dict["pdms"] = calculate_pdms(score_dict)
        # score_dict["pdms_scaled"] = calculate_sharp_scaled_pdms(score_dict)
        # score_dict["pdms"] = score_dict["pdm_score"]
        score_dict["pdms_scaled"] = calculate_scaled_pdms(score_dict)
        # logger.info(score_dict) 
        if verbose:
            score_dict["token"] = token
            score_dict["visual"] = ""
        results.append(score_dict)
    logger.info(f"[PID {os.getpid()}] scoring done token={token}")
    return results

def calculate_pdms(score_dict):
    pdms = score_dict["no_at_fault_collisions"] * score_dict["drivable_area_compliance"] * \
        (5 * score_dict["ego_progress"] + 5 * score_dict["time_to_collision_within_bound"] +
         2 * score_dict["history_comfort"]) / 12
    return pdms

def calculate_scaled_pdms(score_dict, gamma=[0.6, 0.6, 1.0], weights=[5, 5, 2]):
    """
    Additive Focal-PDMS:
    - Uses additive structure for robustness (a single zero sub-score won't zero out the total).
    - Applies Focal Scaling to sub-scores, stretching differences in the high-score range.
    """
    # 1. Hard Safety Gates - collision or leaving drivable area yields 0
    no_coll = score_dict.get("no_at_fault_collisions", 1.0)
    drive = score_dict.get("drivable_area_compliance", 1.0)
    if no_coll == 0 or drive == 0:
        return 0.0

    # 2. Raw continuous sub-scores (default 0.0 if missing)
    ego = np.clip(score_dict.get("ego_progress", 0.0), 0.0, 1.0)
    ttc = np.clip(score_dict.get("time_to_collision_within_bound", 0.0), 0.0, 1.0)
    comfort = np.clip(score_dict.get("history_comfort", 0.0), 0.0, 1.0)

    # 3. Focal mapping (core sparsification step)
    # When gamma < 1.0, stretches the high-score range (e.g., 0.9~1.0) to a wider interval (e.g., 0.6~1.0)
    def focal_map(x, g):
        return 1.0 - (1.0 - x) ** g

    ego_f = focal_map(ego, gamma[0])
    ttc_f = focal_map(ttc, gamma[1])
    comfort_f = focal_map(comfort, gamma[2])

    # 4. Additive Weighted Sum (using original PDMS weights [5, 5, 2])
    weighted_sum = weights[0] * ego_f + weights[1] * ttc_f + weights[2] * comfort_f
    total_weight = sum(weights)

    pdms = weighted_sum / total_weight

    return float(np.clip(pdms, 0.0, 1.0))


def focal_map(x, gamma):
    return 1.0 - (1.0 - x) ** gamma

def scaled_pdms(score_dict, gamma=(0.4,0.4,1.0), eps=1e-6):
    no_coll = score_dict.get("no_at_fault_collisions", 1.0)
    drive = score_dict.get("drivable_area_compliance", 1.0)
    if no_coll == 0 or drive == 0:
        return 0.0
    E = np.clip(score_dict.get("ego_progress", 1.0), 0.0, 1.0)
    T = np.clip(score_dict.get("time_to_collision_within_bound", 1.0), 0.0, 1.0)
    H = np.clip(score_dict.get("history_comfort", 1.0), 0.0, 1.0)
    return float(focal_map(E, gamma[0]) * focal_map(T, gamma[1]) * focal_map(H, gamma[2]) + eps)

def calculate_smooth_scaled_pdms(score_dict, pdms, alpha=0.3, k=0.75, gamma=(0.4,0.4,1.0), eps=1e-6):
    spdms = scaled_pdms(score_dict, gamma=gamma, eps=eps)
    if spdms < alpha:
        reward = 0.5 * pdms**2 / alpha
    else:
        # High-score range: retain scaled-pdms scaling, ensuring > alpha/2
        reward = alpha/2 + k * (spdms - alpha)
    return float(np.clip(reward, 0.0, 1.0))
