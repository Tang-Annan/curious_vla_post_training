import os
import httpx
import re
import json
import threading
import numpy as np
import random
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from verl.utils.reward_score.navsim.helper import (
    denormalize,
    parse_trajectory_string_after_tag,
    get_trajectory_parser,
)
from verl.utils.reward_score.navsim.pdms_logger import BatchJsonlLogger

import logging
from datetime import datetime
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

REWARD_NAME = "navsim_span_grpo"
REWARD_TYPE = "batch"

time_str = datetime.now().strftime("%m%d%H%M")
exp_name = os.environ.get("EXP_NAME", "default_exp")
_log_dir = os.path.join("checkpoints", "debug", exp_name)
os.makedirs(_log_dir, exist_ok=True)
log_file_path = os.path.join(_log_dir, f"generations_{time_str}.jsonl")
log_lock = threading.Lock()

def log_to_jsonl(data: dict, file_path: str):
    with log_lock:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
# batch_logger = BatchJsonlLogger(
#     file_path=log_file_path,
#     batch_size=100,
#     flush_interval=5
# )


'''
"answer": 
    {'gt': [[4.88, 0.0, 0.0], [9.51, 0.0, 0.0], [13.92, 0.0, 0.0], [18.12, 0.05, 0.0], 
            [22.12, 0.08, 0.0], [25.81, 0.07, 0.0], [29.05, 0.05, 0.0], [31.76, -0.08, -0.08]], 
    'token': 'bd2a4d57c04d50f1'}
'''

'''
PAYLOAD = {
    "token": "ffef12d9476e557b",
    "poses": [
        [2.6424, 0.2735, 0.2282], [5.3800, 1.0834, 0.3573],
        [8.1056, 2.2591, 0.4646], [10.9658, 3.7943, 0.5213],
        [13.8787, 5.5712, 0.5706], [16.8150, 7.4979, 0.5823],
        [19.8750, 9.5652, 0.5965], [23.1119, 11.7544, 0.5954]
    ],
    "verbose": False
}
'''

url_pool = ["http://0.0.0.0:8901/score"]
url_pool_group = ["http://0.0.0.0:8901/score_group"]
headers = {"Content-Type": "application/json"}
retries = 3
timeout = 120

SAVED_METRICS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "history_comfort",
    "pdms",
    "pdms_scaled",
)

EXPECTED_FIELDS = {
    "critical_objects": dict,
    "explanation": str,
    "meta_behaviour": dict,
    "future_trajectory": str,
}


def simulator_reward(token: str, poses: list[list[float]], verbose: bool):
    if len(poses) != 8:
        return {field: 0.0 for field in SAVED_METRICS}
    
    payload = {
        "token": token,
        "poses": poses,
        "verbose": verbose
    }
    # print(payload)

    for attempt in range(1, retries + 1):
        try:
            with httpx.Client(trust_env=False, timeout=timeout) as client:
                resp = client.post(random.choice(url_pool), content=json.dumps(payload), headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                return {field: float(data[field]) for field in SAVED_METRICS}
            else:
                logger.warning(f"[WARN] server error code: {resp.status_code}, try again {attempt}/{retries}")
        except httpx.HTTPError as e:
            print(f"[ERROR] request error {e}, after tries {attempt}/{retries}")
            if attempt == retries:
                return {field: 0.0 for field in SAVED_METRICS}
    return {field: 0.0 for field in SAVED_METRICS}

def step_length_reward(indices, ground_truth):
    return int(len(indices) == len(ground_truth))


def format_reward(parsed):
    """
    计算格式奖励：
    - 成功解析JSON +0.4
    - 包含所有必需字段 +0.4
    - 字段类型正确 +0.1
    - 内容非空（至少 explanation 与 trajectory 有内容） +0.1
    """
    score = 0.0
    data = parsed.get("data")

    if not parsed["parsed_ok"] or not isinstance(data, dict):
        return score

    # 1. parse ok
    score += 0.4

    # 2. all present
    all_present = all(k in data for k in EXPECTED_FIELDS)
    if all_present:
        score += 0.4

    # 3. type ok
    type_ok = all(isinstance(data.get(k), t) for k, t in EXPECTED_FIELDS.items())
    if type_ok:
        score += 0.1

    # 4. non empty
    non_empty = (
        bool(data.get("explanation")) and
        bool(data.get("future_trajectory")) and
        len(data.get("critical_objects", {})) > 0
    )
    if non_empty:
        score += 0.1

    return score


def compute_score_fast(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for pdms reward function.")

    parse_fn = get_trajectory_parser()
    
    # 定义单个请求的处理函数
    def process_single_input(reward_input: Dict[str, Any]) -> Dict[str, float]:
        response = reward_input["response"]
        ground_truth = reward_input["ground_truth"]
        token = ground_truth["token"]
        
        # poses = parse_text_waypoint(response)
        normalized_poses = parse_fn(response)
        parsed_ok = normalized_poses is not None and len(normalized_poses) == 8
        poses = denormalize(normalized_poses) if parsed_ok else []
        # print(f"Denormalized poses for token {token}: {poses}")
        
        metrics = simulator_reward(token, poses, False)
        pdms = metrics["pdms"]
        scaled_pdms = metrics["pdms_scaled"]
        # format_score = format_reward(parsed_dict)
        format_score = 1.0 # 格式奖励已经舍弃，不会参与计算
        if pdms is None:
            pdms = 0.0
        if scaled_pdms is None:
            scaled_pdms = 0.0
            
        save_dict = {}
        save_dict['poses'] = poses
        save_dict["token"] = token
        save_dict["pdms"] = pdms
        save_dict["pdms_scaled"] = scaled_pdms
        save_dict["format_score"] = format_score
        save_dict["parsed_ok"] = parsed_ok
        save_dict["raw_response"] = response
        save_dict["response_chars"] = len(response)
        save_dict["response_length"] = int(reward_input["response_length"])
        save_dict.update(metrics)
        # save_dict["overall_score"] =(1 - format_weight) * pdms + format_weight * format_score
        save_dict["overall_score"] = scaled_pdms
        log_to_jsonl(save_dict, log_file_path)
        # batch_logger.write(save_dict)
        
        return {
            # "overall": (1 - format_weight) * pdms + format_weight * format_score,
            "overall": scaled_pdms,
            "format": format_score,
            "accuracy": scaled_pdms,
            "pdms": pdms
        }
    
    # 多线程并发处理，保持结果顺序与输入一致
    with ThreadPoolExecutor(max_workers=96) as executor:
        # 提交所有任务并记录顺序
        future_to_index = {
            executor.submit(process_single_input, req): i 
            for i, req in enumerate(reward_inputs)
        }
        
        # 初始化结果列表，按原顺序填充
        scores = [None] * len(reward_inputs)
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                result = future.result()
                scores[idx] = result
            except Exception as e:
                scores[idx] = {"overall": 0.0, "format": 0.0, "accuracy": 0.0, "pdms": 0.0}
                print(f"Error processing input {idx}: {str(e)}")
    
    return scores
