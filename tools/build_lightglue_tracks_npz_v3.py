#!/usr/bin/env python3
"""
build_lightglue_tracks_npz_v3.py

Stage-oriented multi-view track builder:
1. Pair-graph construction from LightGlue matches
2. Quality-prioritized union / merge
3. Conflict handling with soft resolution
4. Lightweight transitive completion
5. Geometry-aware filtering before export
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

import _bootstrap  # noqa: F401
from calib_utils import (
    format_kitti_layout_help,
    load_intrinsics,
    read_custom_Ks,
    resolve_kitti_scene_inputs,
)
from degeneracy_utils import dump_json, safe_ratio, summarize_array
from frontend_cache import (
    build_lightglue_frontend,
    load_or_extract_feature,
    read_gray_u8,
    resolve_device,
)
from quality_utils import (
    compute_qpair,
    infer_qpair_config,
    load_quality_config,
    make_quality_config_record,
    resolve_quality_section,
    save_quality_config_record,
)


def parse_deltas(text: str) -> List[int]:
    vals = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(int(item))
    vals = sorted(set(v for v in vals if v > 0))
    if not vals:
        raise ValueError("Empty --deltas")
    return vals


def save_frame_npz(out_dir: str, fidx: int, track_ids: np.ndarray, xy: np.ndarray) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, f"{fidx:06d}.npz"),
        track_ids=np.asarray(track_ids, np.int64),
        xy=np.asarray(xy, np.float32),
    )


def undistort_to_pixel(pts_px, K, dist, model):
    import cv2

    pts = np.asarray(pts_px, np.float32).reshape(-1, 1, 2)
    if model in ["radial-tangential", "plumb_bob", "radtan"]:
        out = cv2.undistortPoints(pts, K, dist, P=K)
        return out.reshape(-1, 2).astype(np.float64)
    if "fisheye" in str(model).lower():
        out = cv2.fisheye.undistortPoints(pts, K, dist, P=K)
        return out.reshape(-1, 2).astype(np.float64)
    out = cv2.undistortPoints(pts, K, dist, P=K)
    return out.reshape(-1, 2).astype(np.float64)


def pixel_to_normalized(pts_px, K):
    pts_px = np.asarray(pts_px, np.float64).reshape(-1, 2)
    pts_h = np.concatenate([pts_px, np.ones((pts_px.shape[0], 1), np.float64)], axis=1)
    Kinv = np.linalg.inv(np.asarray(K, np.float64))
    pts_n = (Kinv @ pts_h.T).T
    pts_n /= np.maximum(pts_n[:, 2:3], 1e-12)
    return pts_n[:, :2]


def estimate_essential_inliers_normalized(pts0_n, pts1_n, min_inliers):
    import cv2

    if len(pts0_n) < 8:
        return np.zeros((0,), dtype=bool)
    K_I = np.eye(3, dtype=np.float64)
    E, mask = cv2.findEssentialMat(
        np.asarray(pts0_n, np.float64),
        np.asarray(pts1_n, np.float64),
        K_I,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1e-3,
    )
    if E is None or mask is None:
        return np.zeros((len(pts0_n),), dtype=bool)
    inliers = mask.reshape(-1).astype(bool)
    if int(inliers.sum()) < min_inliers:
        return np.zeros((len(pts0_n),), dtype=bool)
    return inliers


@dataclass
class PairRecord:
    pair_id: int
    i: int
    j: int
    delta: int
    nmatches_raw: int
    ninliers: int
    inlier_ratio: float
    score_mean: float
    score_med: float
    q_pair: float = 0.0
    pair_quality: float = 0.0
    kp0: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int32))
    kp1: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int32))
    match_score: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.float32))
    inlier_mask: np.ndarray = field(default_factory=lambda: np.zeros((0,), bool))
    candidate_source: str = "base"
    edge_level: str = "drop"


@dataclass
class TrackObservation:
    frame: int
    kp_idx: int
    gid: int
    xy: np.ndarray
    obs_score: float


@dataclass
class TrackState:
    root: int
    obs: Dict[int, TrackObservation] = field(default_factory=dict)
    support_sum: float = 0.0
    edge_support_count: int = 0
    pair_q_values: List[float] = field(default_factory=list)
    deltas: List[int] = field(default_factory=list)
    pair_supports: set = field(default_factory=set)
    dropped_observations: int = 0
    conflict_resolutions: int = 0
    rejected_merges: int = 0

    def avg_support(self) -> float:
        if self.edge_support_count <= 0:
            return 0.0
        return float(self.support_sum / self.edge_support_count)

    def track_len(self) -> int:
        return len(self.obs)

    def frame_span(self) -> int:
        if not self.obs:
            return 0
        frames = sorted(self.obs.keys())
        return int(frames[-1] - frames[0])


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        p = int(self.parent[x])
        while p != int(self.parent[p]):
            self.parent[p] = self.parent[int(self.parent[p])]
            p = int(self.parent[p])
        while x != p:
            nxt = int(self.parent[x])
            self.parent[x] = p
            x = nxt
        return p

    def union_roots(self, ra: int, rb: int) -> int:
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra


class MultiViewTrackBuilder:
    def __init__(
        self,
        *,
        offsets: np.ndarray,
        kpts_xy_by_frame: Sequence[np.ndarray],
        pair_records: Sequence[PairRecord],
        edge_quality_alpha: float,
        conflict_merge_px: float,
        conflict_keep_ratio: float,
        completion_px: float,
        completion_rounds: int,
        min_completion_quality: float,
    ):
        self.offsets = np.asarray(offsets, np.int64)
        self.kpts_xy_by_frame = list(kpts_xy_by_frame)
        self.pair_records = list(pair_records)
        self.edge_quality_alpha = float(edge_quality_alpha)
        self.conflict_merge_px = float(conflict_merge_px)
        self.conflict_keep_ratio = float(conflict_keep_ratio)
        self.completion_px = float(completion_px)
        self.completion_rounds = int(max(completion_rounds, 0))
        self.min_completion_quality = float(min_completion_quality)

        total_nodes = int(self.offsets[-1])
        self.uf = UnionFind(total_nodes)
        self.track_states: Dict[int, TrackState] = {}
        self.edge_support_attempts = 0
        self.edge_merges = 0
        self.edge_same_root = 0
        self.edge_conflict_rejects = 0
        self.edge_soft_resolutions = 0
        self.completion_merges = 0

        global_frames = np.concatenate(
            [np.full(self.offsets[i + 1] - self.offsets[i], i, dtype=np.int32) for i in range(len(self.offsets) - 1)],
            axis=0,
        ) if total_nodes > 0 else np.zeros((0,), dtype=np.int32)
        self.global_frames = global_frames

    def _make_initial_track(self, gid: int) -> TrackState:
        frame = int(self.global_frames[gid])
        kp_idx = int(gid - self.offsets[frame])
        obs = TrackObservation(
            frame=frame,
            kp_idx=kp_idx,
            gid=gid,
            xy=self.kpts_xy_by_frame[frame][kp_idx].astype(np.float64),
            obs_score=0.0,
        )
        return TrackState(root=gid, obs={frame: obs})

    def _get_track(self, root: int) -> TrackState:
        root = int(self.uf.find(root))
        state = self.track_states.get(root)
        if state is None:
            state = self._make_initial_track(root)
            self.track_states[root] = state
        return state

    def _promote_observation_score(self, state: TrackState, frame: int, obs_score: float) -> None:
        obs = state.obs.get(frame)
        if obs is not None and obs_score > obs.obs_score:
            obs.obs_score = float(obs_score)

    def _add_edge_support(self, root: int, rec: PairRecord, local_score: float) -> None:
        state = self._get_track(root)
        support_value = float(rec.q_pair * (0.5 + 0.5 * np.clip(local_score, 0.0, 1.0)))
        state.support_sum += support_value
        state.edge_support_count += 1
        state.pair_q_values.append(float(rec.q_pair))
        state.deltas.append(int(rec.delta))
        state.pair_supports.add((int(rec.i), int(rec.j)))

    def _merge_states(
        self,
        state_a: TrackState,
        state_b: TrackState,
        *,
        local_score: float,
    ) -> Tuple[bool, TrackState, int]:
        merged_obs: Dict[int, TrackObservation] = {}
        soft_resolutions = 0

        frames = sorted(set(state_a.obs.keys()) | set(state_b.obs.keys()))
        for frame in frames:
            obs_a = state_a.obs.get(frame)
            obs_b = state_b.obs.get(frame)
            if obs_a is None:
                merged_obs[frame] = obs_b
                continue
            if obs_b is None:
                merged_obs[frame] = obs_a
                continue
            if obs_a.kp_idx == obs_b.kp_idx:
                merged_obs[frame] = obs_a if obs_a.obs_score >= obs_b.obs_score else obs_b
                continue

            dist_px = float(np.linalg.norm(obs_a.xy - obs_b.xy))
            score_a = max(state_a.avg_support(), 1e-6) * (1.0 + obs_a.obs_score)
            score_b = max(state_b.avg_support(), 1e-6) * (1.0 + obs_b.obs_score)
            stronger = obs_a if score_a >= score_b else obs_b
            strong_score = max(score_a, score_b)
            weak_score = min(score_a, score_b)

            if dist_px <= self.conflict_merge_px:
                merged_obs[frame] = stronger
                soft_resolutions += 1
                continue

            if strong_score >= self.conflict_keep_ratio * max(weak_score, 1e-6):
                merged_obs[frame] = stronger
                soft_resolutions += 1
                continue

            if local_score >= self.min_completion_quality and dist_px <= max(self.conflict_merge_px * 2.0, self.completion_px):
                merged_obs[frame] = stronger
                soft_resolutions += 1
                continue

            return False, state_a, soft_resolutions

        merged = TrackState(
            root=state_a.root,
            obs=merged_obs,
            support_sum=state_a.support_sum + state_b.support_sum,
            edge_support_count=state_a.edge_support_count + state_b.edge_support_count,
            pair_q_values=state_a.pair_q_values + state_b.pair_q_values,
            deltas=state_a.deltas + state_b.deltas,
            pair_supports=set(state_a.pair_supports) | set(state_b.pair_supports),
            dropped_observations=state_a.dropped_observations + state_b.dropped_observations,
            conflict_resolutions=state_a.conflict_resolutions + state_b.conflict_resolutions + soft_resolutions,
            rejected_merges=state_a.rejected_merges + state_b.rejected_merges,
        )
        if soft_resolutions > 0:
            merged.dropped_observations += soft_resolutions
        return True, merged, soft_resolutions

    def process_match_edges(self) -> None:
        scored_edges = []
        for rec in self.pair_records:
            if rec.edge_level != "strong":
                continue
            inlier_idx = np.where(rec.inlier_mask)[0]
            if inlier_idx.size == 0:
                continue
            for local_idx in inlier_idx.tolist():
                local_score = float(rec.match_score[local_idx]) if rec.match_score.size else 0.5
                edge_quality = self.edge_quality_alpha * float(rec.q_pair) + (1.0 - self.edge_quality_alpha) * float(
                    np.clip(local_score, 0.0, 1.0)
                )
                ga = int(self.offsets[rec.i] + int(rec.kp0[local_idx]))
                gb = int(self.offsets[rec.j] + int(rec.kp1[local_idx]))
                scored_edges.append((edge_quality, ga, gb, rec.pair_id, local_score))

        scored_edges.sort(key=lambda x: x[0], reverse=True)
        for _edge_quality, ga, gb, pair_id, local_score in scored_edges:
            self.edge_support_attempts += 1
            rec = self.pair_records[pair_id]
            ra = self.uf.find(ga)
            rb = self.uf.find(gb)
            state_a = self._get_track(ra)
            state_b = self._get_track(rb)
            self._promote_observation_score(state_a, int(self.global_frames[ga]), local_score)
            self._promote_observation_score(state_b, int(self.global_frames[gb]), local_score)

            if ra == rb:
                self.edge_same_root += 1
                self._add_edge_support(ra, rec, local_score)
                continue

            ok, merged_state, soft_resolutions = self._merge_states(state_a, state_b, local_score=local_score)
            if not ok:
                self.edge_conflict_rejects += 1
                state_a.rejected_merges += 1
                state_b.rejected_merges += 1
                continue

            new_root = self.uf.union_roots(ra, rb)
            merged_state.root = new_root
            self.track_states.pop(ra, None)
            self.track_states.pop(rb, None)
            self.track_states[new_root] = merged_state
            self._add_edge_support(new_root, rec, local_score)
            self.edge_merges += 1
            self.edge_soft_resolutions += int(soft_resolutions)

    def _collect_completion_candidates(self) -> List[Tuple[float, int, int, Dict[str, float]]]:
        candidates: Dict[Tuple[int, int], Dict[str, float]] = {}

        frame_index: Dict[int, List[Tuple[int, TrackObservation]]] = defaultdict(list)
        for root in self.active_roots():
            state = self._get_track(root)
            if state.track_len() < 2:
                continue
            for frame, obs in state.obs.items():
                frame_index[int(frame)].append((int(root), obs))

        for frame, items in frame_index.items():
            if len(items) < 2:
                continue
            for idx_a in range(len(items)):
                root_a, obs_a = items[idx_a]
                for idx_b in range(idx_a + 1, len(items)):
                    root_b, obs_b = items[idx_b]
                    if root_a == root_b:
                        continue
                    dist_px = float(np.linalg.norm(obs_a.xy - obs_b.xy))
                    if dist_px > self.completion_px:
                        continue
                    key = (min(root_a, root_b), max(root_a, root_b))
                    item = candidates.setdefault(
                        key,
                        {
                            "shared_frames": 0.0,
                            "bridge_edges": 0.0,
                            "bridge_score_sum": 0.0,
                            "frame_pair_count": 0.0,
                            "shared_score_sum": 0.0,
                            "min_shared_dist": float("inf"),
                        },
                    )
                    item["shared_frames"] += 1.0
                    item["shared_score_sum"] += 1.0 / (1.0 + dist_px)
                    item["min_shared_dist"] = min(float(item["min_shared_dist"]), dist_px)

        for rec in self.pair_records:
            if rec.edge_level != "weak":
                continue
            if rec.q_pair < self.min_completion_quality:
                continue
            inlier_idx = np.where(rec.inlier_mask)[0]
            if inlier_idx.size == 0:
                continue
            seen_local_pairs = set()
            for local_idx in inlier_idx.tolist():
                ga = int(self.offsets[rec.i] + int(rec.kp0[local_idx]))
                gb = int(self.offsets[rec.j] + int(rec.kp1[local_idx]))
                ra = int(self.uf.find(ga))
                rb = int(self.uf.find(gb))
                if ra == rb:
                    continue
                key = (min(ra, rb), max(ra, rb))
                pair_key = (int(rec.i), int(rec.j))
                item = candidates.setdefault(
                    key,
                    {
                        "shared_frames": 0.0,
                        "bridge_edges": 0.0,
                        "bridge_score_sum": 0.0,
                        "frame_pair_count": 0.0,
                        "shared_score_sum": 0.0,
                        "min_shared_dist": float("inf"),
                    },
                )
                local_score = float(rec.match_score[local_idx]) if rec.match_score.size else 0.5
                item["bridge_edges"] += 1.0
                item["bridge_score_sum"] += float(rec.q_pair) * (0.35 + 0.65 * np.clip(local_score, 0.0, 1.0))
                if pair_key not in seen_local_pairs:
                    item["frame_pair_count"] += 1.0
                    seen_local_pairs.add(pair_key)

        scored = []
        for (ra, rb), item in candidates.items():
            score = (
                1.40 * float(item["shared_score_sum"])
                + 0.25 * float(item["bridge_edges"])
                + 1.00 * float(item["bridge_score_sum"])
                + 0.35 * float(item["frame_pair_count"])
            )
            scored.append((score, int(ra), int(rb), item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def transitive_completion(self) -> None:
        if self.completion_rounds <= 0:
            return

        for _round in range(self.completion_rounds):
            merged_this_round = 0
            for score, root_a, root_b, item in self._collect_completion_candidates():
                ra = int(self.uf.find(root_a))
                rb = int(self.uf.find(root_b))
                if ra == rb:
                    continue
                state_a = self._get_track(ra)
                state_b = self._get_track(rb)
                if min(state_a.avg_support(), state_b.avg_support()) < self.min_completion_quality:
                    continue
                if item["shared_frames"] <= 0.0 and item["bridge_edges"] < 2.0 and item["frame_pair_count"] < 2.0:
                    continue

                local_score = float(
                    np.clip(
                        0.5 * min(1.0, item["bridge_score_sum"] / max(item["bridge_edges"], 1.0))
                        + 0.5 * min(1.0, item["shared_score_sum"]),
                        0.0,
                        1.0,
                    )
                )
                ok, merged_state, soft_resolutions = self._merge_states(
                    state_a,
                    state_b,
                    local_score=local_score,
                )
                if not ok:
                    continue
                new_root = self.uf.union_roots(ra, rb)
                merged_state.root = new_root
                self.track_states.pop(ra, None)
                self.track_states.pop(rb, None)
                self.track_states[new_root] = merged_state
                if soft_resolutions > 0:
                    self.edge_soft_resolutions += int(soft_resolutions)
                self.completion_merges += 1
                merged_this_round += 1
            if merged_this_round == 0:
                break

    def active_roots(self) -> List[int]:
        roots = []
        for root in list(self.track_states.keys()):
            r = self.uf.find(root)
            if r == root:
                roots.append(int(root))
        return roots

    def compute_track_stats(self, root: int) -> Dict[str, float]:
        state = self._get_track(root)
        track_len = state.track_len()
        pair_q = np.asarray(state.pair_q_values, dtype=np.float64)
        deltas = np.asarray(state.deltas, dtype=np.float64)
        num_pair_supports = int(len(state.pair_supports))
        return {
            "track_len": int(track_len),
            "support_sum": float(state.support_sum),
            "support_mean": float(state.avg_support()),
            "frame_span": int(state.frame_span()),
            "median_delta": float(np.median(deltas)) if deltas.size else 0.0,
            "num_distinct_pair_supports": num_pair_supports,
            "pair_q_mean": float(np.mean(pair_q)) if pair_q.size else 1.0,
            "pair_q_min": float(np.min(pair_q)) if pair_q.size else 1.0,
            "pair_q_count": int(pair_q.size),
            "dropped_observations": int(state.dropped_observations),
            "conflict_resolutions": int(state.conflict_resolutions),
            "rejected_merges": int(state.rejected_merges),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_glob", required=True, help="e.g. .../image_0/*.png")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--deltas", type=str, default="1")
    ap.add_argument("--dense_local_max_delta", type=int, default=0,
                    help="If >0, always include all local deltas up to this value.")
    ap.add_argument("--bridge_deltas", type=str, default="",
                    help="Additional sparse bridge deltas, e.g. 8,10,12.")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--max_kpts", type=int, default=2048)
    ap.add_argument("--filter_th", type=float, default=0.1)
    ap.add_argument("--mutual", action="store_true")
    ap.add_argument("--min_score", type=float, default=0.0)
    ap.add_argument("--use_ransac", action="store_true")
    ap.add_argument("--ransac_thresh", type=float, default=1.0)
    ap.add_argument("--min_inliers", type=int, default=50)
    ap.add_argument("--min_inliers_weak", type=int, default=20,
                    help="Weak-edge minimum inliers for layered geometric verification.")
    ap.add_argument("--dataset", choices=["kitti", "euroc", "custom", "none"], default="none")
    ap.add_argument("--K_npy", type=str, default=None)
    ap.add_argument("--Ks_npy", type=str, default=None)
    ap.add_argument("--image_K_idx_npy", type=str, default=None)
    ap.add_argument("--kitti_calib", type=str, default=None)
    ap.add_argument("--kitti_cam", type=str, default="P0")
    ap.add_argument("--euroc_yaml", type=str, default=None)
    ap.add_argument("--undistort", action="store_true")
    ap.add_argument("--seed_all", action="store_true")
    ap.add_argument("--qpair_mode", choices=["weighted_sum", "product", "log_additive"], default="weighted_sum")
    ap.add_argument("--qpair_threshold", type=float, default=0.0)
    ap.add_argument("--weak_qpair_threshold", type=float, default=0.45)
    ap.add_argument("--strong_qpair_threshold", type=float, default=0.75)
    ap.add_argument("--dump_quality_stats", action="store_true")
    ap.add_argument("--dump_pair_graph", action="store_true")
    ap.add_argument("--feature_cache_dir", type=str, default=None)
    ap.add_argument("--quality_config", type=str, default=None)
    ap.add_argument("--auto_quality_refs", action="store_true")
    ap.add_argument("--edge_quality_alpha", type=float, default=0.8)
    ap.add_argument("--conflict_merge_px", type=float, default=2.0)
    ap.add_argument("--conflict_keep_ratio", type=float, default=1.25)
    ap.add_argument("--completion_px", type=float, default=2.0)
    ap.add_argument("--completion_rounds", type=int, default=1)
    ap.add_argument("--min_completion_quality", type=float, default=0.25)
    ap.add_argument("--rescue_min_degree", type=int, default=0,
                    help="If >0, add rescue pairs for frames whose verified degree falls below this threshold.")
    ap.add_argument("--rescue_extra_deltas", type=str, default="",
                    help="Extra deltas to try only for low-degree frames, e.g. 4,6,8,10,12.")
    ap.add_argument("--min_track_len_export", type=int, default=2)
    ap.add_argument("--min_track_support_sum", type=float, default=0.0)
    ap.add_argument("--min_distinct_pair_supports", type=int, default=1)
    ap.add_argument("--min_frame_span", type=int, default=1)
    ap.add_argument("--track_quality_threshold", type=float, default=0.0)
    ap.add_argument("--topk_tracks", type=int, default=0)
    return ap


def resolve_images_and_intrinsics(args):
    if args.dataset == "kitti":
        resolved = resolve_kitti_scene_inputs(
            kitti_calib=args.kitti_calib,
            image_glob=args.image_glob,
        )
        if resolved["image_glob"] != args.image_glob or resolved["kitti_calib"] != args.kitti_calib:
            print(f"[KITTI] auto-resolved scene {resolved['scene_id']} to:")
            print(f"  calib     : {resolved['kitti_calib']}")
            print(f"  image_glob: {resolved['image_glob']}")
            args.image_glob = resolved["image_glob"]
            args.kitti_calib = resolved["kitti_calib"]

    imgs = sorted(glob.glob(args.image_glob))
    if not imgs:
        if args.dataset == "kitti":
            raise FileNotFoundError(
                f"No KITTI images matched: {args.image_glob}\n"
                f"{format_kitti_layout_help(None)}"
            )
        raise FileNotFoundError(args.image_glob)
    if args.max_frames is not None:
        imgs = imgs[:args.max_frames]

    Ks = None
    image_K_idx = None
    if args.dataset == "custom" and args.Ks_npy is not None:
        Ks, image_K_idx = read_custom_Ks(args.Ks_npy, args.image_K_idx_npy)
        if image_K_idx is None:
            raise ValueError("--Ks_npy requires --image_K_idx_npy")
        if image_K_idx.shape[0] < len(imgs):
            raise ValueError(f"image_K_idx has {image_K_idx.shape[0]} entries but needs at least {len(imgs)}")
        image_K_idx = image_K_idx[: len(imgs)]
        K = None
        dist = None
        dist_model = None
    else:
        K, dist, dist_model = load_intrinsics(
            dataset=args.dataset,
            kitti_calib=args.kitti_calib,
            kitti_cam=args.kitti_cam,
            euroc_yaml=args.euroc_yaml,
            K_npy=args.K_npy,
        )
    return imgs, K, Ks, image_K_idx, dist, dist_model


def extract_features(imgs, args):
    import torch

    dev = resolve_device(args.device)
    print("[Device]", dev)
    extractor, matcher = build_lightglue_frontend(
        max_kpts=args.max_kpts,
        filter_th=args.filter_th,
        device=dev,
    )
    feats_cache = []
    kpts_save_all = []
    n_kpts_per_frame = []
    for img_path in imgs:
        feats, kpts = load_or_extract_feature(
            img_path,
            extractor=extractor,
            device=dev,
            max_kpts=args.max_kpts,
            cache_dir=args.feature_cache_dir,
            image_loader=read_gray_u8,
        )
        feats_cache.append(feats)
        kpts_save_all.append(kpts.astype(np.float64))
        n_kpts_per_frame.append(int(kpts.shape[0]))
    return matcher, feats_cache, kpts_save_all, n_kpts_per_frame


def maybe_undistort(kpts_save_all, args, K, dist, dist_model):
    if not args.undistort:
        return kpts_save_all
    if K is None or dist is None:
        raise ValueError("--undistort requires calibration")
    out = []
    for pts in kpts_save_all:
        out.append(undistort_to_pixel(pts, K, dist, dist_model))
    return out


def build_candidate_pairs(num_frames: int, base_deltas: Sequence[int], dense_local_max_delta: int, bridge_deltas: Sequence[int]):
    pair_specs = {}
    dense_deltas = list(range(1, int(dense_local_max_delta) + 1)) if int(dense_local_max_delta) > 0 else []

    for d in list(base_deltas) + dense_deltas:
        d = int(d)
        if d <= 0:
            continue
        for i in range(num_frames):
            j = i + d
            if j >= num_frames:
                continue
            pair_specs[(i, j)] = "base" if d in set(base_deltas) else "dense"

    for d in bridge_deltas:
        d = int(d)
        if d <= 0:
            continue
        for i in range(num_frames):
            j = i + d
            if j >= num_frames:
                continue
            pair_specs.setdefault((i, j), "bridge")
    return pair_specs


def build_pair_records(
    *,
    matcher,
    feats_cache,
    kpts_save_all,
    args,
    K,
    Ks,
    image_K_idx,
    deltas,
) -> Tuple[List[PairRecord], Dict[int, int], Dict[int, int]]:
    import torch

    pair_records: List[PairRecord] = []
    gap_attempts = defaultdict(int)
    gap_success = defaultdict(int)
    pair_id = 0

    bridge_deltas = parse_deltas(args.bridge_deltas) if str(args.bridge_deltas).strip() else []
    rescue_deltas = parse_deltas(args.rescue_extra_deltas) if str(args.rescue_extra_deltas).strip() else []
    pair_specs = build_candidate_pairs(len(kpts_save_all), deltas, args.dense_local_max_delta, bridge_deltas)

    def evaluate_pair(i: int, j: int, source: str):
        nonlocal pair_id
        feats0 = feats_cache[i]
        feats1 = feats_cache[j]
        kpts0_save = kpts_save_all[i]
        kpts1_save = kpts_save_all[j]
        d = int(j - i)
        gap_attempts[d] += 1

        with torch.no_grad():
            out = matcher({"image0": feats0, "image1": feats1})

        m0 = out["matches0"][0].detach().cpu().numpy().astype(np.int64)
        valid = m0 >= 0
        nmatch_raw = int(valid.sum())

        if args.mutual and ("matches1" in out):
            m1 = out["matches1"][0].detach().cpu().numpy().astype(np.int64)
            idx0_ref = np.arange(m0.shape[0], dtype=np.int64)
            ok = valid.copy()
            ok[valid] &= (m1[m0[valid]] == idx0_ref[valid])
            valid &= ok

        s0 = None
        if "matching_scores0" in out:
            s0 = out["matching_scores0"][0].detach().cpu().numpy()
        if args.min_score > 0 and s0 is not None:
            valid &= (s0 >= args.min_score)

        idx0_all = np.where(valid)[0]
        idx1_all = m0[idx0_all]
        if idx0_all.size == 0:
            rec = PairRecord(
                pair_id=pair_id,
                i=int(i),
                j=int(j),
                delta=d,
                nmatches_raw=int(max(nmatch_raw, 0)),
                ninliers=0,
                inlier_ratio=0.0,
                score_mean=float("nan"),
                score_med=float("nan"),
                candidate_source=source,
            )
            pair_id += 1
            return rec

        pair_scores = np.asarray(s0[idx0_all], np.float32) if s0 is not None else np.full((idx0_all.size,), 0.5, np.float32)
        score_mean = float(np.mean(pair_scores))
        score_med = float(np.median(pair_scores))

        inlier_mask = np.ones((idx0_all.size,), dtype=bool)
        if args.use_ransac:
            if idx0_all.size >= 8:
                pts0 = kpts0_save[idx0_all].astype(np.float64)
                pts1 = kpts1_save[idx1_all].astype(np.float64)
                if Ks is not None:
                    K0 = Ks[int(image_K_idx[i])]
                    K1 = Ks[int(image_K_idx[j])]
                    pts0_n = pixel_to_normalized(pts0, K0)
                    pts1_n = pixel_to_normalized(pts1, K1)
                    inlier_mask = estimate_essential_inliers_normalized(pts0_n, pts1_n, args.min_inliers_weak)
                else:
                    import cv2

                    E, mask = cv2.findEssentialMat(
                        pts0,
                        pts1,
                        K,
                        method=cv2.RANSAC,
                        prob=0.999,
                        threshold=args.ransac_thresh,
                    )
                    if mask is None:
                        inlier_mask = np.zeros((idx0_all.size,), dtype=bool)
                    else:
                        inlier_mask = mask.reshape(-1).astype(bool)
                        if int(inlier_mask.sum()) < args.min_inliers_weak:
                            inlier_mask = np.zeros((idx0_all.size,), dtype=bool)
            else:
                inlier_mask = np.zeros((idx0_all.size,), dtype=bool)

        ninliers = int(inlier_mask.sum())
        inlier_ratio = float(safe_ratio(ninliers, max(nmatch_raw, 1)))
        if ninliers > 0:
            gap_success[d] += 1
        rec = PairRecord(
            pair_id=pair_id,
            i=int(i),
            j=int(j),
            delta=d,
            nmatches_raw=int(max(nmatch_raw, 0)),
            ninliers=int(ninliers),
            inlier_ratio=float(inlier_ratio),
            score_mean=float(score_mean),
            score_med=float(score_med),
            kp0=np.asarray(idx0_all, np.int32),
            kp1=np.asarray(idx1_all, np.int32),
            match_score=np.asarray(pair_scores, np.float32),
            inlier_mask=np.asarray(inlier_mask, bool),
            candidate_source=source,
        )
        pair_id += 1
        return rec

    processed = set()
    for (i, j), source in sorted(pair_specs.items()):
        pair_records.append(evaluate_pair(i, j, source))
        processed.add((i, j))

    if args.rescue_min_degree > 0 and rescue_deltas:
        degree = np.zeros((len(kpts_save_all),), dtype=np.int32)
        for rec in pair_records:
            if rec.ninliers >= args.min_inliers_weak:
                degree[rec.i] += 1
                degree[rec.j] += 1
        low_degree_frames = np.where(degree < int(args.rescue_min_degree))[0].tolist()
        for i in low_degree_frames:
            for d in rescue_deltas:
                for j in (i + int(d), i - int(d)):
                    if j < 0 or j >= len(kpts_save_all) or j == i:
                        continue
                    a, b = (i, j) if i < j else (j, i)
                    if (a, b) in processed:
                        continue
                    pair_records.append(evaluate_pair(a, b, "rescue"))
                    processed.add((a, b))

    return pair_records, dict(gap_attempts), dict(gap_success)


def resolve_qpair_config(pair_records: Sequence[PairRecord], args):
    quality_payload = load_quality_config(args.quality_config)
    qpair_config_resolved = resolve_quality_section(quality_payload, "qpair", {})
    qpair_config = dict(qpair_config_resolved)
    if args.auto_quality_refs and pair_records:
        score_values = []
        for rec in pair_records:
            vals = [rec.score_mean, rec.score_med]
            vals = [float(v) for v in vals if np.isfinite(v)]
            if vals:
                score_values.append(float(np.mean(vals)))
        qpair_config = infer_qpair_config(
            ninliers=[rec.ninliers for rec in pair_records],
            inlier_ratio=[rec.inlier_ratio for rec in pair_records],
            score_values=score_values,
            base_config=qpair_config,
        )
    quality_record = make_quality_config_record(
        stage="build_lightglue_tracks_npz_v3",
        section="qpair",
        raw_payload=quality_payload,
        resolved_section=qpair_config_resolved,
        effective_section=qpair_config,
        mode=args.qpair_mode,
        auto_quality_refs=args.auto_quality_refs,
        quality_config_path=args.quality_config,
        extra={
            "output_path": os.path.join(args.out_dir, "quality_config_used.json"),
            "decision_parameters": {
                "qpair_threshold": float(args.qpair_threshold),
                "edge_quality_alpha": float(args.edge_quality_alpha),
            },
        },
    )
    return qpair_config, quality_record


def apply_pair_quality(pair_records: Sequence[PairRecord], qpair_config, args) -> None:
    for rec in pair_records:
        rec.q_pair = float(
            compute_qpair(
                ninliers=rec.ninliers,
                nmatches=max(rec.nmatches_raw, 1),
                inlier_ratio=rec.inlier_ratio,
                score_mean=rec.score_mean,
                score_med=rec.score_med,
                delta=rec.delta,
                method=0,
                mode=args.qpair_mode,
                config=qpair_config,
            )
        )
        rec.pair_quality = rec.q_pair
        if rec.ninliers < args.min_inliers_weak or rec.q_pair < max(args.qpair_threshold, args.weak_qpair_threshold):
            rec.edge_level = "drop"
            rec.inlier_mask = np.zeros_like(rec.inlier_mask, dtype=bool)
        elif rec.ninliers >= args.min_inliers and rec.q_pair >= args.strong_qpair_threshold:
            rec.edge_level = "strong"
        else:
            rec.edge_level = "weak"


def filter_and_rank_tracks(builder: MultiViewTrackBuilder, args) -> List[Tuple[int, Dict[str, float], float]]:
    ranked = []
    for root in builder.active_roots():
        stats = builder.compute_track_stats(root)
        len_score = min(float(stats["track_len"]) / max(float(args.min_track_len_export), 1.0), 2.0) / 2.0
        span_score = min(float(stats["frame_span"]) / max(float(args.min_frame_span), 1.0), 2.0) / 2.0
        pair_div_score = min(float(stats["num_distinct_pair_supports"]) / max(float(args.min_distinct_pair_supports), 1.0), 2.0) / 2.0
        support_score = np.clip(float(stats["support_mean"]), 0.0, 1.0)
        quality = float(
            0.34 * len_score
            + 0.24 * pair_div_score
            + 0.18 * span_score
            + 0.14 * support_score
            + 0.10 * np.clip(float(stats["pair_q_mean"]), 0.0, 1.0)
        )
        keep = (
            int(stats["track_len"]) >= int(args.min_track_len_export)
            and float(stats["support_sum"]) >= float(args.min_track_support_sum)
            and int(stats["num_distinct_pair_supports"]) >= int(args.min_distinct_pair_supports)
            and int(stats["frame_span"]) >= int(args.min_frame_span)
            and quality >= float(args.track_quality_threshold)
        )
        if keep:
            ranked.append((int(root), stats, quality))
    ranked.sort(key=lambda x: x[2], reverse=True)
    if args.topk_tracks and len(ranked) > int(args.topk_tracks):
        ranked = ranked[: int(args.topk_tracks)]
    return ranked


def export_tracks(builder: MultiViewTrackBuilder, ranked_tracks, kpts_save_all, out_dir, seed_all: bool):
    N = len(kpts_save_all)
    kp_maps = [dict() for _ in range(N)]
    track_stats_export = []
    for tid, (root, stats, quality) in enumerate(ranked_tracks):
        state = builder._get_track(root)
        for frame, obs in state.obs.items():
            kp_maps[int(frame)][int(obs.kp_idx)] = int(tid)
        stats = dict(stats)
        stats["track_quality"] = float(quality)
        stats["track_id"] = int(tid)
        track_stats_export.append(stats)

    if seed_all:
        next_tid = len(track_stats_export)
        for frame in range(N):
            for kp_idx in range(kpts_save_all[frame].shape[0]):
                if kp_idx in kp_maps[frame]:
                    continue
                kp_maps[frame][int(kp_idx)] = int(next_tid)
                track_stats_export.append(
                    {
                        "track_id": int(next_tid),
                        "track_len": 1,
                        "support_sum": 0.0,
                        "support_mean": 0.0,
                        "frame_span": 0,
                        "median_delta": 0.0,
                        "num_distinct_pair_supports": 0,
                        "pair_q_mean": 1.0,
                        "pair_q_min": 1.0,
                        "pair_q_count": 0,
                        "dropped_observations": 0,
                        "conflict_resolutions": 0,
                        "rejected_merges": 0,
                        "track_quality": 0.0,
                    }
                )
                next_tid += 1

    for frame in range(N):
        kp_map = kp_maps[frame]
        if not kp_map:
            save_frame_npz(out_dir, frame, np.zeros((0,), np.int64), np.zeros((0, 2), np.float32))
            continue
        kp_idx = np.array(list(kp_map.keys()), dtype=np.int64)
        tids = np.array([kp_map[int(k)] for k in kp_idx], dtype=np.int64)
        xy = kpts_save_all[frame][kp_idx].astype(np.float32)
        order = np.argsort(tids)
        save_frame_npz(out_dir, frame, tids[order], xy[order])
    return track_stats_export


def save_pair_graph(pair_records: Sequence[PairRecord], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    pair_i = []
    pair_j = []
    pair_delta = []
    pair_nmatches = []
    pair_ninliers = []
    pair_inlier_ratio = []
    pair_score_mean = []
    pair_score_med = []
    pair_q = []
    pair_offsets = [0]
    kp0_all = []
    kp1_all = []
    score_all = []
    inlier_all = []
    for rec in pair_records:
        pair_i.append(int(rec.i))
        pair_j.append(int(rec.j))
        pair_delta.append(int(rec.delta))
        pair_nmatches.append(int(rec.nmatches_raw))
        pair_ninliers.append(int(rec.ninliers))
        pair_inlier_ratio.append(float(rec.inlier_ratio))
        pair_score_mean.append(float(rec.score_mean) if np.isfinite(rec.score_mean) else np.nan)
        pair_score_med.append(float(rec.score_med) if np.isfinite(rec.score_med) else np.nan)
        pair_q.append(float(rec.q_pair))
        kp0_all.append(np.asarray(rec.kp0, np.int32))
        kp1_all.append(np.asarray(rec.kp1, np.int32))
        score_all.append(np.asarray(rec.match_score, np.float32))
        inlier_all.append(np.asarray(rec.inlier_mask, bool))
        pair_offsets.append(pair_offsets[-1] + int(rec.kp0.size))
    np.savez_compressed(
        os.path.join(out_dir, "pair_graph_edges.npz"),
        pair_i=np.asarray(pair_i, np.int32),
        pair_j=np.asarray(pair_j, np.int32),
        pair_delta=np.asarray(pair_delta, np.int16),
        pair_nmatches=np.asarray(pair_nmatches, np.int32),
        pair_ninliers=np.asarray(pair_ninliers, np.int32),
        pair_inlier_ratio=np.asarray(pair_inlier_ratio, np.float32),
        pair_score_mean=np.asarray(pair_score_mean, np.float32),
        pair_score_med=np.asarray(pair_score_med, np.float32),
        pair_q=np.asarray(pair_q, np.float32),
        pair_offsets=np.asarray(pair_offsets, np.int64),
        kp_idx_i=np.concatenate(kp0_all, axis=0) if kp0_all else np.zeros((0,), np.int32),
        kp_idx_j=np.concatenate(kp1_all, axis=0) if kp1_all else np.zeros((0,), np.int32),
        match_score=np.concatenate(score_all, axis=0) if score_all else np.zeros((0,), np.float32),
        inlier_mask=np.concatenate(inlier_all, axis=0) if inlier_all else np.zeros((0,), bool),
    )


def save_quality_sidecars(out_dir: str, pair_records, track_stats_export, gap_attempts, gap_success, builder, quality_record) -> None:
    os.makedirs(out_dir, exist_ok=True)
    save_quality_config_record(os.path.join(out_dir, "quality_config_used.json"), quality_record)

    edge_i = np.asarray([rec.i for rec in pair_records], np.int32)
    edge_j = np.asarray([rec.j for rec in pair_records], np.int32)
    edge_delta = np.asarray([rec.delta for rec in pair_records], np.int16)
    edge_q = np.asarray([rec.q_pair for rec in pair_records], np.float32)
    edge_ninliers = np.asarray([rec.ninliers for rec in pair_records], np.int32)
    edge_inlier_ratio = np.asarray([rec.inlier_ratio for rec in pair_records], np.float32)
    edge_score_mean = np.asarray([rec.score_mean for rec in pair_records], np.float32)
    edge_score_med = np.asarray([rec.score_med for rec in pair_records], np.float32)

    track_ids = np.asarray([item["track_id"] for item in track_stats_export], np.int64)
    track_len = np.asarray([item["track_len"] for item in track_stats_export], np.int32)
    pair_q_mean = np.asarray([item["pair_q_mean"] for item in track_stats_export], np.float32)
    pair_q_min = np.asarray([item["pair_q_min"] for item in track_stats_export], np.float32)
    pair_q_count = np.asarray([item["pair_q_count"] for item in track_stats_export], np.int32)
    support_sum = np.asarray([item["support_sum"] for item in track_stats_export], np.float32)
    support_mean = np.asarray([item["support_mean"] for item in track_stats_export], np.float32)
    frame_span = np.asarray([item["frame_span"] for item in track_stats_export], np.int32)
    median_delta = np.asarray([item["median_delta"] for item in track_stats_export], np.float32)
    distinct_pair_supports = np.asarray([item["num_distinct_pair_supports"] for item in track_stats_export], np.int32)
    track_quality = np.asarray([item["track_quality"] for item in track_stats_export], np.float32)
    dropped_observations = np.asarray([item["dropped_observations"] for item in track_stats_export], np.int32)
    conflict_resolutions = np.asarray([item["conflict_resolutions"] for item in track_stats_export], np.int32)

    np.savez_compressed(
        os.path.join(out_dir, "track_quality_summary.npz"),
        track_ids=track_ids,
        track_len=track_len,
        pair_q_mean=pair_q_mean,
        pair_q_min=pair_q_min,
        pair_q_count=pair_q_count,
        support_sum=support_sum,
        support_mean=support_mean,
        frame_span=frame_span,
        median_delta=median_delta,
        distinct_pair_supports=distinct_pair_supports,
        track_quality=track_quality,
        dropped_observations=dropped_observations,
        conflict_resolutions=conflict_resolutions,
    )
    np.savez_compressed(
        os.path.join(out_dir, "pair_quality_edges.npz"),
        i=edge_i,
        j=edge_j,
        delta=edge_delta,
        q_pair=edge_q,
        ninliers=edge_ninliers,
        inlier_ratio=edge_inlier_ratio,
        score_mean=edge_score_mean,
        score_med=edge_score_med,
    )
    dump_json(
        os.path.join(out_dir, "track_build_quality_stats.json"),
        {
            "pair_stats": {
                "q_pair": summarize_array(edge_q),
                "ninliers": summarize_array(edge_ninliers),
                "inlier_ratio": summarize_array(edge_inlier_ratio),
                "edge_level_counts": {
                    "strong": int(sum(1 for rec in pair_records if rec.edge_level == "strong")),
                    "weak": int(sum(1 for rec in pair_records if rec.edge_level == "weak")),
                    "drop": int(sum(1 for rec in pair_records if rec.edge_level == "drop")),
                },
                "candidate_source_counts": {
                    "base": int(sum(1 for rec in pair_records if rec.candidate_source == "base")),
                    "dense": int(sum(1 for rec in pair_records if rec.candidate_source == "dense")),
                    "bridge": int(sum(1 for rec in pair_records if rec.candidate_source == "bridge")),
                    "rescue": int(sum(1 for rec in pair_records if rec.candidate_source == "rescue")),
                },
            },
            "track_stats": {
                "count": int(track_ids.size),
                "track_length": summarize_array(track_len),
                "pair_q_mean": summarize_array(pair_q_mean),
                "pair_q_min": summarize_array(pair_q_min),
                "support_sum": summarize_array(support_sum),
                "support_mean": summarize_array(support_mean),
                "frame_span": summarize_array(frame_span),
                "median_delta": summarize_array(median_delta),
                "distinct_pair_supports": summarize_array(distinct_pair_supports),
                "track_quality": summarize_array(track_quality),
            },
            "merge_stats": {
                "edge_support_attempts": int(builder.edge_support_attempts),
                "edge_merges": int(builder.edge_merges),
                "edge_same_root": int(builder.edge_same_root),
                "edge_conflict_rejects": int(builder.edge_conflict_rejects),
                "edge_soft_resolutions": int(builder.edge_soft_resolutions),
                "completion_merges": int(builder.completion_merges),
            },
            "gap_stats": {
                str(int(d)): {
                    "attempts": int(gap_attempts.get(int(d), 0)),
                    "successful_pairs": int(gap_success.get(int(d), 0)),
                    "success_rate": float(safe_ratio(gap_success.get(int(d), 0), max(gap_attempts.get(int(d), 1), 1))),
                }
                for d in sorted(gap_attempts.keys())
            },
        },
    )


def main():
    args = build_arg_parser().parse_args()
    imgs, K, Ks, image_K_idx, dist, dist_model = resolve_images_and_intrinsics(args)
    deltas = parse_deltas(args.deltas)
    print("[TracksV3] frames:", len(imgs))
    print("[TracksV3] deltas:", deltas)
    if K is not None:
        print("[K]\n", K)
    elif Ks is not None:
        print(f"[Ks] count={Ks.shape[0]}")
    if dist is not None:
        print("[DIST]", dist, "model=", dist_model)
    if args.use_ransac and K is None and Ks is None:
        raise ValueError("--use_ransac needs intrinsics K (set --dataset kitti/euroc/custom and calib args)")

    matcher, feats_cache, kpts_save_all, n_kpts_per_frame = extract_features(imgs, args)
    kpts_save_all = maybe_undistort(kpts_save_all, args, K, dist, dist_model)

    offsets = np.zeros(len(imgs) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(np.asarray(n_kpts_per_frame, np.int64))

    pair_records, gap_attempts, gap_success = build_pair_records(
        matcher=matcher,
        feats_cache=feats_cache,
        kpts_save_all=kpts_save_all,
        args=args,
        K=K,
        Ks=Ks,
        image_K_idx=image_K_idx,
        deltas=deltas,
    )
    qpair_config, quality_record = resolve_qpair_config(pair_records, args)
    apply_pair_quality(pair_records, qpair_config, args)

    builder = MultiViewTrackBuilder(
        offsets=offsets,
        kpts_xy_by_frame=kpts_save_all,
        pair_records=pair_records,
        edge_quality_alpha=args.edge_quality_alpha,
        conflict_merge_px=args.conflict_merge_px,
        conflict_keep_ratio=args.conflict_keep_ratio,
        completion_px=args.completion_px,
        completion_rounds=args.completion_rounds,
        min_completion_quality=args.min_completion_quality,
    )
    builder.process_match_edges()
    builder.transitive_completion()

    ranked_tracks = filter_and_rank_tracks(builder, args)
    track_stats_export = export_tracks(builder, ranked_tracks, kpts_save_all, args.out_dir, args.seed_all)

    if pair_records:
        qvals = [rec.q_pair for rec in pair_records]
        kept = [rec.ninliers for rec in pair_records]
        print(
            f"[TracksV3] pair count={len(pair_records)}, kept matches median={np.median(kept):.1f}, "
            f"q_pair median={np.median(qvals):.3f}"
        )
    print(
        f"[TracksV3] exported_tracks={len(track_stats_export)}, "
        f"edge_merges={builder.edge_merges}, conflict_rejects={builder.edge_conflict_rejects}, "
        f"completion_merges={builder.completion_merges}"
    )
    for d in deltas:
        succ = gap_success.get(int(d), 0)
        att = gap_attempts.get(int(d), 0)
        print(f"[TracksV3][gap={int(d)}] attempts={att}, successful_pairs={succ}, success_rate={safe_ratio(succ, max(att, 1)):.3f}")

    if args.dump_quality_stats:
        save_quality_sidecars(args.out_dir, pair_records, track_stats_export, gap_attempts, gap_success, builder, quality_record)
    if args.dump_pair_graph:
        save_pair_graph(pair_records, args.out_dir)


if __name__ == "__main__":
    main()
