#!/usr/bin/env python3
# Pose_Only_patched_v2.py
#
# Pose-Only (LiGT) translation estimation given global rotations.
#
# v2 changes (aimed at fixing huge translation drift while keeping paper-faithful math):
#   - Add explicit weak-parallax / degeneracy filtering (paper notes B=C=D=0 when [X_i]_x R_{xi;i} X_xi = 0).
#   - Improve base-pair selection to better approximate argmax_u over long tracks (paper Eq.(67)).
#   - Add optional IRLS reweighting on LiGT equations to down-weight outlier/contradictory equations.
#   - Keep Step-6 sign disambiguation a_{xi h}^T t_{xi;h} >= 0 (median voting).
#
# Inputs:
#   - R_abs.npy: (N,3,3) world->cam rotations.
#   - tracks_npz_dir: per-frame .npz containing (track_ids, xy) pixel coords.
#   - K: intrinsics.

import os, glob, argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as SciRotation
from scipy.sparse import coo_matrix, identity
from scipy.sparse.linalg import eigsh, lobpcg, cg, LinearOperator, lsmr
from scipy.sparse.linalg._eigen.arpack.arpack import ArpackNoConvergence


# ----------------- geometry helpers -----------------

def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v.reshape(-1).tolist()
    return np.array(
        [[0, -z, y],
         [z, 0, -x],
         [-y, x, 0]],
        dtype=np.float64,
    )


def load_kitti_K_from_calib(calib_txt: str, cam: str = "P0") -> np.ndarray:
    with open(calib_txt, "r") as f:
        for line in f:
            if line.startswith(cam + ":"):
                vals = [float(x) for x in line.split()[1:]]
                P = np.array(vals, np.float64).reshape(3, 4)
                return P[:, :3].copy()
    raise RuntimeError(f"Cannot find {cam}: in {calib_txt}")


def to_homo_norm_batch(xy_px: np.ndarray, K: np.ndarray) -> np.ndarray:
    xy_px = np.asarray(xy_px, np.float64).reshape(-1, 2)
    n = xy_px.shape[0]
    pts_h = np.concatenate([xy_px, np.ones((n, 1), np.float64)], axis=1)
    Kinv = np.linalg.inv(K)
    X = (Kinv @ pts_h.T).T
    X /= np.maximum(X[:, 2:3], 1e-12)
    return X


def compute_u(Ri, Rj, Xi, Xj) -> float:
    Rij = Rj @ Ri.T
    return float(np.linalg.norm(skew(Xj) @ (Rij @ Xi)))


def compute_aT(Ri, Rj, Xi, Xj) -> np.ndarray:
    """Return a^T_{i,j} (1x3)."""
    Rij = Rj @ Ri.T
    v = skew(Rij @ Xi) @ Xj
    return (v.reshape(1, 3) @ skew(Xj))


def choose_base_pair(
    frames: List[int],
    Xs: List[np.ndarray],
    R_abs: np.ndarray,
    max_candidates: int = 80,
    full_search_len: int = 50,
    min_gap: int = 0,
    max_gap: int = 0,
    rng: Optional[np.random.Generator] = None,
):
    """Pick (p,q) indices (within obs list) that maximize u_{i,j}.

    Paper Step-3 is argmax u over all pairs. We approximate it:
      - If track length <= full_search_len: exhaustive O(L^2)
      - Else: sample endpoints + uniform + random up to max_candidates.

    Gap constraints:
      - min_gap: require |fj-fi| >= min_gap (0 disables)
      - max_gap: require |fj-fi| <= max_gap (0 disables)
    """
    L = len(frames)
    if L < 2:
        return None, 0.0

    if rng is None:
        rng = np.random.default_rng(0)

    if L <= full_search_len:
        idxs = np.arange(L, dtype=int)
    else:
        # deterministic backbone
        keep = set()
        keep.add(0)
        keep.add(L - 1)
        # uniform samples
        uni = np.linspace(0, L - 1, min(L, max_candidates // 2), dtype=int)
        keep.update(int(i) for i in uni)
        # random fill
        remain = max_candidates - len(keep)
        if remain > 0 and L > len(keep):
            cand = np.array(sorted(set(range(L)) - keep), dtype=int)
            if cand.size > 0:
                pick = rng.choice(cand, size=min(remain, cand.size), replace=False)
                keep.update(int(i) for i in pick)
        idxs = np.array(sorted(keep), dtype=int)

    best, best_u = None, 0.0
    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            p, q = int(idxs[a]), int(idxs[b])
            fi, fj = frames[p], frames[q]
            gap = abs(fj - fi)
            if min_gap and gap < min_gap:
                continue
            if max_gap and gap > max_gap:
                continue
            u = compute_u(R_abs[fi], R_abs[fj], Xs[p], Xs[q])
            if u > best_u:
                best_u = u
                best = (p, q)
    return best, best_u


# ----------------- load tracks -----------------

def build_tracks_obs(track_npz_dir: str, K: np.ndarray, max_frames: Optional[int] = None):
    npz_files = sorted(glob.glob(os.path.join(track_npz_dir, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(track_npz_dir)
    if max_frames is not None:
        npz_files = npz_files[:max_frames]

    tracks: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    for fidx, p in tqdm(list(enumerate(npz_files)), desc="Load tracks npz"):
        d = np.load(p)
        ids = d["track_ids"].astype(np.int64)
        xy = d["xy"].astype(np.float64)
        if len(ids) == 0:
            continue
        Xn = to_homo_norm_batch(xy, K)
        for k, tid in enumerate(ids):
            tracks[int(tid)].append((fidx, Xn[k]))
    return dict(tracks), npz_files


def _stack_rows_from_skew_x(x: np.ndarray) -> np.ndarray:
    """Return 2 independent rows from [x]_x for normalized bearing x."""
    return skew(x)[:2, :]


def triangulate_track_point_ls(
    obs: List[Tuple[int, np.ndarray]],
    R_abs: np.ndarray,
    t_all: np.ndarray,
) -> Tuple[Optional[np.ndarray], float]:
    """Estimate one 3D point from fixed rotations and camera centers."""
    A_rows: List[np.ndarray] = []
    b_rows: List[np.ndarray] = []

    for fidx, Xn in obs:
        x = np.asarray(Xn, np.float64).reshape(3,)
        C = np.asarray(t_all[fidx], np.float64).reshape(3,)
        S = _stack_rows_from_skew_x(x) @ R_abs[fidx]
        A_rows.append(S)
        b_rows.append(S @ C)

    if len(A_rows) < 2:
        return None, np.inf

    A = np.concatenate(A_rows, axis=0)
    b = np.concatenate(b_rows, axis=0)
    if A.shape[0] < 4:
        return None, np.inf

    try:
        Xw, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None, np.inf

    resid = A @ Xw - b
    rms = float(np.sqrt(np.mean(resid * resid))) if resid.size else 0.0
    return Xw.astype(np.float64), rms


def estimate_track_points(
    tracks_obs: Dict[int, List[Tuple[int, np.ndarray]]],
    R_abs: np.ndarray,
    t_all: np.ndarray,
    min_track_len: int = 3,
    max_point_rms: float = 5e-3,
    min_depth: float = 1e-6,
) -> Dict[int, np.ndarray]:
    """Triangulate world points from current camera centers."""
    pts3d: Dict[int, np.ndarray] = {}

    for tid, obs in tracks_obs.items():
        if len(obs) < min_track_len:
            continue

        Xw, rms = triangulate_track_point_ls(obs, R_abs, t_all)
        if Xw is None or not np.isfinite(Xw).all() or rms > max_point_rms:
            continue

        # Require positive depth in at least two views to avoid unstable points.
        positive_depth = 0
        for fidx, _ in obs:
            z = (R_abs[fidx] @ (Xw - t_all[fidx]))[2]
            if z > min_depth:
                positive_depth += 1
        if positive_depth < 2:
            continue

        pts3d[int(tid)] = Xw

    return pts3d


def compute_track_reprojection_residuals(
    obs: List[Tuple[int, np.ndarray]],
    Xw: np.ndarray,
    R_abs: np.ndarray,
    t_all: np.ndarray,
    min_depth: float = 1e-6,
) -> Optional[np.ndarray]:
    res = []
    Xw = np.asarray(Xw, np.float64).reshape(3,)

    for fidx, Xn in obs:
        x = np.asarray(Xn, np.float64).reshape(3,)
        Xc = R_abs[fidx] @ (Xw - t_all[fidx])
        if Xc[2] <= min_depth:
            return None
        pred = Xc[:2] / Xc[2]
        res.append(pred - x[:2])

    if not res:
        return None
    return np.concatenate(res, axis=0).astype(np.float64)


def compute_poseonly_reprojection_stats(
    tracks_obs: Dict[int, List[Tuple[int, np.ndarray]]],
    R_abs: np.ndarray,
    t_all: np.ndarray,
    min_track_len: int = 3,
    point_rms: float = 5e-3,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    pts3d = estimate_track_points(
        tracks_obs,
        R_abs,
        t_all,
        min_track_len=min_track_len,
        max_point_rms=point_rms,
    )

    residual_blocks: List[np.ndarray] = []
    pts_kept: Dict[int, np.ndarray] = {}
    for tid, Xw in pts3d.items():
        obs = tracks_obs[int(tid)]
        r = compute_track_reprojection_residuals(obs, Xw, R_abs, t_all)
        if r is None or not np.isfinite(r).all():
            continue
        residual_blocks.append(r)
        pts_kept[int(tid)] = Xw

    if not residual_blocks:
        return np.zeros((0,), dtype=np.float64), {}
    return np.concatenate(residual_blocks, axis=0), pts_kept


def pack_pose_params(
    R_abs: np.ndarray,
    t_all: np.ndarray,
    ref_idx: int = 0,
    refine_rotations: bool = True,
    refine_translations: bool = True,
) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for i in range(R_abs.shape[0]):
        if i == ref_idx:
            continue
        if refine_rotations:
            chunks.append(SciRotation.from_matrix(R_abs[i]).as_rotvec().astype(np.float64))
        if refine_translations:
            chunks.append(np.asarray(t_all[i], np.float64).reshape(3,))
    if not chunks:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def unpack_pose_params(
    x: np.ndarray,
    R_ref: np.ndarray,
    t_ref: np.ndarray,
    ref_idx: int = 0,
    refine_rotations: bool = True,
    refine_translations: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    N = R_ref.shape[0]
    R_abs = np.asarray(R_ref, np.float64).copy()
    t_all = np.asarray(t_ref, np.float64).copy()

    ptr = 0
    for i in range(N):
        if i == ref_idx:
            continue
        if refine_rotations:
            rv = np.asarray(x[ptr:ptr + 3], np.float64)
            ptr += 3
            R_abs[i] = SciRotation.from_rotvec(rv).as_matrix()
        if refine_translations:
            t_all[i] = np.asarray(x[ptr:ptr + 3], np.float64)
            ptr += 3
    return R_abs, t_all


def make_fixed_points_residual_fn(
    tracks_obs: Dict[int, List[Tuple[int, np.ndarray]]],
    pts3d: Dict[int, np.ndarray],
    R_ref: np.ndarray,
    t_ref: np.ndarray,
    ref_idx: int = 0,
    refine_rotations: bool = True,
    refine_translations: bool = True,
) -> callable:
    tids = [int(tid) for tid in pts3d.keys()]

    def residual_fn(x: np.ndarray) -> np.ndarray:
        R_abs, t_all = unpack_pose_params(
            x,
            R_ref,
            t_ref,
            ref_idx=ref_idx,
            refine_rotations=refine_rotations,
            refine_translations=refine_translations,
        )
        residual_blocks: List[np.ndarray] = []
        for tid in tids:
            r = compute_track_reprojection_residuals(tracks_obs[tid], pts3d[tid], R_abs, t_all)
            if r is None:
                # Keep dimension fixed while heavily penalizing invalid poses.
                r = np.full((2 * len(tracks_obs[tid]),), 1e2, dtype=np.float64)
            residual_blocks.append(r)
        return np.concatenate(residual_blocks, axis=0) if residual_blocks else np.zeros((0,), dtype=np.float64)

    return residual_fn


def solve_translations_from_points(
    tracks_obs: Dict[int, List[Tuple[int, np.ndarray]]],
    pts3d: Dict[int, np.ndarray],
    R_abs: np.ndarray,
    ref_idx: int = 0,
    irls_iters: int = 2,
    huber_k: float = 1.5,
) -> np.ndarray:
    """Solve camera centers from fixed world points and fixed rotations."""
    N = int(R_abs.shape[0])

    def col_base(f: int):
        if f == ref_idx:
            return None
        ridx = f - 1 if f > ref_idx else f
        return 3 * ridx

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    rhs_rows: List[float] = []
    row_id = 0
    rr = np.repeat(np.arange(2, dtype=int), 3)
    cc = np.tile(np.arange(3, dtype=int), 2)

    for tid, obs in tracks_obs.items():
        Xw = pts3d.get(int(tid), None)
        if Xw is None:
            continue
        Xw = np.asarray(Xw, np.float64).reshape(3,)

        for fidx, Xn in obs:
            x = np.asarray(Xn, np.float64).reshape(3,)
            S = _stack_rows_from_skew_x(x) @ R_abs[fidx]
            b = S @ Xw

            cb = col_base(int(fidx))
            if cb is None:
                continue

            vv = S.reshape(-1)
            rows.extend((row_id + rr).tolist())
            cols.extend((cb + cc).tolist())
            data.extend(vv.tolist())
            rhs_rows.extend(b.tolist())
            row_id += 2

    if row_id == 0:
        raise RuntimeError("PA could not build any camera equations from triangulated points.")

    L = coo_matrix(
        (np.asarray(data, np.float64), (np.asarray(rows, np.int64), np.asarray(cols, np.int64))),
        shape=(row_id, 3 * (N - 1)),
    ).tocsr()
    b = np.asarray(rhs_rows, np.float64)

    w_row = np.ones(row_id, dtype=np.float64)
    x = None
    for it in range(max(0, irls_iters) + 1):
        Lw = L.multiply(w_row[:, None]) if it > 0 else L
        bw = b * w_row if it > 0 else b
        sol = lsmr(Lw, bw, atol=1e-8, btol=1e-8, maxiter=20000)
        x = np.asarray(sol[0], np.float64)

        if it == irls_iters:
            break

        r = (L @ x - b).reshape(-1, 2)
        rn = np.linalg.norm(r, axis=1)
        med = float(np.median(rn))
        delta = max(1e-12, huber_k * med)
        w_eq = np.ones_like(rn)
        big = rn > delta
        w_eq[big] = delta / np.maximum(rn[big], 1e-12)
        w_row = np.repeat(np.sqrt(w_eq), 2)
        rq = np.quantile(rn, [0.5, 0.9, 0.99])
        print(f"[PA-trans][IRLS {it+1}/{irls_iters}] residual median={rq[0]:.3e}, p90={rq[1]:.3e}, p99={rq[2]:.3e}, huber_delta={delta:.3e}")

    assert x is not None
    t_red = x.reshape(-1, 3)
    t_all = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        if i == ref_idx:
            continue
        ridx = i - 1 if i > ref_idx else i
        t_all[i] = t_red[ridx]
    return t_all


def run_pose_adjustment(
    tracks_obs: Dict[int, List[Tuple[int, np.ndarray]]],
    R_abs: np.ndarray,
    t_init: np.ndarray,
    ref_idx: int = 0,
    min_track_len: int = 3,
    pa_iters: int = 3,
    pa_point_rms: float = 5e-3,
    pa_refine_rotations: bool = True,
    pa_refine_translations: bool = True,
    pa_max_nfev: int = 50,
    pa_loss: str = "soft_l1",
    pa_f_scale: float = 1e-3,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    Conservative paper-style PA:
      1. analytically reconstruct 3D points from current poses
      2. optimize poses only with fixed reconstructed points
      3. reconstruct points again and accept the update only if the
         true pose-only reprojection error decreases
    """
    R_curr = np.asarray(R_abs, np.float64).copy()
    t_curr = np.asarray(t_init, np.float64).copy()
    residual0, pts3d = compute_poseonly_reprojection_stats(
        tracks_obs,
        R_curr,
        t_curr,
        min_track_len=min_track_len,
        point_rms=pa_point_rms,
    )
    if residual0.size == 0 or not pts3d:
        raise RuntimeError("PA initialization produced no valid reprojection residuals.")

    cost_curr = 0.5 * float(np.dot(residual0, residual0))
    rmse_curr = float(np.sqrt(np.mean(residual0 * residual0)))
    print(f"[PA] init points={len(pts3d)}, residual_rmse={rmse_curr:.3e}, cost={cost_curr:.6e}")

    for it in range(max(1, pa_iters)):
        residual_fixed, pts_fixed = compute_poseonly_reprojection_stats(
            tracks_obs,
            R_curr,
            t_curr,
            min_track_len=min_track_len,
            point_rms=pa_point_rms,
        )
        if residual_fixed.size == 0 or not pts_fixed:
            print(f"[PA {it+1}/{pa_iters}] skipped: no valid reconstructed points.")
            break

        x0 = pack_pose_params(
            R_curr,
            t_curr,
            ref_idx=ref_idx,
            refine_rotations=pa_refine_rotations,
            refine_translations=pa_refine_translations,
        )
        if x0.size == 0:
            print(f"[PA {it+1}/{pa_iters}] skipped: no free pose variables.")
            break

        residual_fn = make_fixed_points_residual_fn(
            tracks_obs,
            pts_fixed,
            R_curr,
            t_curr,
            ref_idx=ref_idx,
            refine_rotations=pa_refine_rotations,
            refine_translations=pa_refine_translations,
        )
        result = least_squares(
            residual_fn,
            x0,
            method="trf",
            loss=pa_loss,
            f_scale=pa_f_scale,
            max_nfev=pa_max_nfev,
        )

        R_next, t_next = unpack_pose_params(
            result.x,
            R_curr,
            t_curr,
            ref_idx=ref_idx,
            refine_rotations=pa_refine_rotations,
            refine_translations=pa_refine_translations,
        )
        residual_next, pts_next = compute_poseonly_reprojection_stats(
            tracks_obs,
            R_next,
            t_next,
            min_track_len=min_track_len,
            point_rms=pa_point_rms,
        )
        if residual_next.size == 0 or not pts_next:
            print(f"[PA {it+1}/{pa_iters}] rejected: candidate produced no valid reprojection residuals.")
            break

        cost_next = 0.5 * float(np.dot(residual_next, residual_next))
        rmse_next = float(np.sqrt(np.mean(residual_next * residual_next)))
        step_t = np.linalg.norm(t_next - t_curr, axis=1)
        dR = []
        for i in range(R_curr.shape[0]):
            dR_i = SciRotation.from_matrix(R_next[i] @ R_curr[i].T).magnitude()
            dR.append(dR_i)
        dR = np.asarray(dR, np.float64)
        print(
            f"[PA {it+1}/{pa_iters}] fixed_points={len(pts_fixed)}, "
            f"candidate_points={len(pts_next)}, "
            f"rmse {rmse_curr:.3e} -> {rmse_next:.3e}, "
            f"cost {cost_curr:.6e} -> {cost_next:.6e}, "
            f"step_t_max={np.max(step_t):.3e}, step_r_max_deg={np.degrees(np.max(dR)):.3e}"
        )
        if cost_next + 1e-12 < cost_curr:
            R_curr, t_curr = R_next, t_next
            pts3d = pts_next
            cost_curr = cost_next
            rmse_curr = rmse_next
            print(f"[PA {it+1}/{pa_iters}] accepted")
        else:
            print(f"[PA {it+1}/{pa_iters}] rejected (no reprojection improvement)")
            break

    return R_curr, t_curr, pts3d


def _solve_min_eigvector(A: "scipy.sparse.spmatrix", solver: str = "auto") -> np.ndarray:
    """Return eigenvector of smallest eigenvalue for SPD matrix A."""
    t_red = None

    if solver in ("auto", "eigsh"):
        try:
            w, v = eigsh(A, k=1, which="SM", tol=1e-6, maxiter=5000)
            t_red = v[:, 0].astype(np.float64)
            print("[LiGT] eigsh ok, lambda=", float(w[0]))
        except ArpackNoConvergence as e:
            print("[LiGT] eigsh failed:", str(e))

    if t_red is None and solver in ("auto", "lobpcg"):
        try:
            n = A.shape[0]
            A_reg = A + (1e-12 * identity(n, format="csr", dtype=np.float64))
            diag = A_reg.diagonal()
            M_inv = 1.0 / np.maximum(diag, 1e-12)

            def _jacobi_matvec(x: np.ndarray) -> np.ndarray:
                x = np.asarray(x)
                if x.ndim != 1:
                    x = x.reshape(-1)
                return M_inv * x

            def _jacobi_matmat(X: np.ndarray) -> np.ndarray:
                X = np.asarray(X)
                if X.ndim != 2:
                    X = X.reshape(-1, 1)
                return M_inv[:, None] * X

            M = LinearOperator(A_reg.shape, matvec=_jacobi_matvec, matmat=_jacobi_matmat, dtype=np.float64)
            X0 = np.random.default_rng(0).standard_normal((n, 1)).astype(np.float64)
            X0 /= max(float(np.linalg.norm(X0)), 1e-12)
            w, V = lobpcg(A_reg, X0, M=M, largest=False, tol=1e-6, maxiter=2000)
            t_red = V[:, 0].astype(np.float64)
            print("[LiGT] lobpcg ok, lambda=", float(w[0]))
        except Exception as e:
            print("[LiGT] lobpcg failed:", repr(e))

    if t_red is not None:
        return t_red

    # fallback: constrained solve
    print("[LiGT] fallback constrained solve")
    n = A.shape[0]
    diag = A.diagonal()
    j0 = int(np.argmax(diag))
    print(f"[LiGT] Fix t[{j0}] = 1.0 (max diag={diag[j0]:.3e})")

    mask = np.ones(n, dtype=bool)
    mask[j0] = False
    rest = np.where(mask)[0]

    A_rr = A[rest, :][:, rest].tocsr()
    rhs = (-A[rest, j0]).toarray().ravel()
    M_inv = 1.0 / np.maximum(A_rr.diagonal(), 1e-12)
    M = LinearOperator(A_rr.shape, matvec=lambda x: M_inv * x, dtype=np.float64)
    try:
        x, info = cg(A_rr, rhs, M=M, rtol=1e-6, atol=1e-10, maxiter=20000)
    except TypeError:
        x, info = cg(A_rr, rhs, M=M, tol=1e-6, maxiter=20000)

    if info == 0:
        t_red = np.zeros(n, dtype=np.float64)
        t_red[rest] = x
        t_red[j0] = 1.0
        print("[LiGT] CG converged on constrained normal equations.")
        return t_red

    print(f"[LiGT] CG did not converge (info={info}). Switch to LSMR on L in caller.")
    raise RuntimeError("CG failed")


def solve_ligt_sparse(
    tracks_obs: Dict[int, List[Tuple[int, np.ndarray]]],
    R_abs: np.ndarray,
    ref_idx: int = 0,
    min_track_len: int = 3,
    max_tracks: int = 50000,
    max_equations: int = 2_000_000,
    base_pair_candidates: int = 80,
    base_pair_full_search_len: int = 50,
    base_pair_min_gap: int = 0,
    base_pair_max_gap: int = 0,
    u_min: float = 1e-3,
    g_min: float = 1e-3,
    eq_norm: str = "fro",  # none|fro|u2
    solver: str = "auto",
    sign_vote_tracks: int = 2000,
    irls_iters: int = 0,
    irls_huber_k: float = 1.5,
    gt_t_all: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build LiGT and solve translations.

    Notes:
      - u_min: reject tracks whose chosen base pair has too small parallax (paper Step-3 maximizes u).
      - g_min: reject near-degenerate equations where ||[X_i]_x R_{xi;i} X_xi|| is tiny (paper notes B=C=D=0).
      - irls_iters: robustify by reweighting equations based on residual norms.
    """
    N = int(R_abs.shape[0])
    rng = np.random.default_rng(0)

    def col_base(f: int):
        if f == ref_idx:
            return None
        ridx = f - 1 if f > ref_idx else f
        return 3 * ridx

    items = list(tracks_obs.items())
    rng.shuffle(items)
    items = items[: min(len(items), max_tracks)]

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    n_eq = 0
    rr = np.repeat(np.arange(3, dtype=int), 3)
    cc = np.tile(np.arange(3, dtype=int), 3)

    def add_block(r0: int, c0: int, B: np.ndarray):
        vv = B.reshape(-1)
        rows.extend((r0 + rr).tolist())
        cols.extend((c0 + cc).tolist())
        data.extend(vv.tolist())

    u_kept: List[float] = []
    g_kept: List[float] = []

    for _, obs in tqdm(items, desc="Build LiGT"):
        if len(obs) < min_track_len:
            continue
        obs = sorted(obs, key=lambda x: x[0])
        frames = [int(f) for f, _ in obs]
        Xs = [np.asarray(X, np.float64).reshape(3,) for _, X in obs]

        best, best_u = choose_base_pair(
            frames,
            Xs,
            R_abs,
            max_candidates=base_pair_candidates,
            full_search_len=base_pair_full_search_len,
            min_gap=base_pair_min_gap,
            max_gap=base_pair_max_gap,
            rng=rng,
        )
        if best is None or best_u < u_min:
            continue

        p_idx, q_idx = best
        f_xi, f_h = frames[p_idx], frames[q_idx]
        X_xi, X_h = Xs[p_idx], Xs[q_idx]
        u_xh = compute_u(R_abs[f_xi], R_abs[f_h], X_xi, X_h)
        if u_xh < u_min:
            continue
        u_kept.append(float(u_xh))

        aT = compute_aT(R_abs[f_xi], R_abs[f_h], X_xi, X_h)
        aTRh = (aT @ R_abs[f_h]).reshape(1, 3)

        for k in range(len(frames)):
            f_i = frames[k]
            if f_i == f_xi or f_i == f_h:
                continue
            X_i = Xs[k]

            R_xi_i = R_abs[f_i] @ R_abs[f_xi].T
            v = R_xi_i @ X_xi

            # paper degeneracy note: if [X_i]_x R_{xi;i} X_xi == 0 => B=C=D=0 (no constraint)
            g = float(np.linalg.norm(skew(X_i) @ v))
            if g < g_min:
                continue
            g_kept.append(g)

            Btmp = v.reshape(3, 1) @ aTRh.reshape(1, 3)
            B = skew(X_i) @ Btmp
            C = (u_xh ** 2) * (skew(X_i) @ R_abs[f_i])
            D = -(B + C)  # coefficient for t_xi is -(B+C)

            if eq_norm == "fro":
                denom = float(np.linalg.norm(B, "fro") + np.linalg.norm(C, "fro"))
                if denom > 1e-12:
                    w = 1.0 / denom
                    B *= w; C *= w; D *= w
            elif eq_norm == "u2":
                denom = float(u_xh * u_xh)
                if denom > 1e-12:
                    w = 1.0 / denom
                    B *= w; C *= w; D *= w

            r0 = 3 * n_eq
            cb_h, cb_i, cb_xi = col_base(f_h), col_base(f_i), col_base(f_xi)
            if cb_h is not None:
                add_block(r0, cb_h, B)
            if cb_i is not None:
                add_block(r0, cb_i, C)
            if cb_xi is not None:
                add_block(r0, cb_xi, D)

            n_eq += 1
            if n_eq >= max_equations:
                break
        if n_eq >= max_equations:
            break

    if n_eq == 0:
        raise RuntimeError("No LiGT equations. Try lower thresholds or better tracks.")

    if u_kept:
        uq = np.quantile(u_kept, [0.1, 0.5, 0.9])
        print(f"[LiGT] u_xh kept quantiles: p10={uq[0]:.3e}, p50={uq[1]:.3e}, p90={uq[2]:.3e} (n={len(u_kept)})")
    if g_kept:
        gq = np.quantile(g_kept, [0.1, 0.5, 0.9])
        print(f"[LiGT] g kept quantiles: p10={gq[0]:.3e}, p50={gq[1]:.3e}, p90={gq[2]:.3e} (n={len(g_kept)})")

    n_rows = 3 * n_eq
    n_cols = 3 * (N - 1)
    L = coo_matrix(
        (np.array(data, np.float64), (np.array(rows, np.int64), np.array(cols, np.int64))),
        shape=(n_rows, n_cols),
    ).tocsr()

    # Debug: evaluate LiGT residual using provided GT translations (camera centers in world).
    # If GT residual is large but the solved residual is tiny, tracks/associations are inconsistent with true geometry.
    if gt_t_all is not None:
        gt_t_all = np.asarray(gt_t_all, dtype=np.float64)
        if gt_t_all.shape[0] >= N and gt_t_all.shape[1:] == (3,):
                    # align GT to the same gauge as our system: enforce t_ref = 0
            gt_t_all = gt_t_all - gt_t_all[ref_idx]
            t_red_gt = []
            for i in range(N):
                if i == ref_idx:
                    continue
                t_red_gt.append(gt_t_all[i])
            t_vec_gt = np.asarray(t_red_gt, dtype=np.float64).reshape(-1)
            r_gt = (L @ t_vec_gt).reshape(-1, 3)
            rn = np.linalg.norm(r_gt, axis=1)
            q = np.quantile(rn, [0.5, 0.9, 0.99])
            print(f"[LiGT][GT-check] residual norm median={q[0]:.3e}, p90={q[1]:.3e}, p99={q[2]:.3e} (n_eq={rn.size})")
        else:
            print('[LiGT][GT-check] skip: gt_t_all has incompatible shape', gt_t_all.shape)


    # IRLS loop: reweight equations using Huber weights on 3D residual norms.
    w_row = np.ones(n_rows, dtype=np.float64)
    t_vec = None
    for it in range(max(0, irls_iters) + 1):
        Lw = L.multiply(w_row[:, None]) if it > 0 else L
        A = (Lw.T @ Lw).tocsr()

        try:
            t_vec = _solve_min_eigvector(A, solver=solver)
        except RuntimeError:
            # stable constrained LS on Lw
            print("[LiGT] switching to constrained LSMR (LS on L) due to CG failure")
            n = A.shape[0]
            diag = A.diagonal()
            j0 = int(np.argmax(diag))
            mask = np.ones(n, dtype=bool); mask[j0] = False
            rest = np.where(mask)[0]
            L_rr = Lw[:, rest]
            b = (-Lw[:, j0]).toarray().ravel()
            sol = lsmr(L_rr, b, atol=1e-6, btol=1e-6, maxiter=20000)
            x_lsmr = sol[0]
            t_vec = np.zeros(n, dtype=np.float64)
            t_vec[rest] = x_lsmr
            t_vec[j0] = 1.0

        if it == irls_iters:
            break

        # compute residuals on unweighted L (or weighted? use Lw for consistency)
        r = Lw @ t_vec
        r = np.asarray(r).reshape(-1)
        r3 = r.reshape(-1, 3)
        rn = np.linalg.norm(r3, axis=1)
        med = float(np.median(rn))
        delta = max(1e-12, irls_huber_k * med)
        w_eq = np.ones_like(rn)
        big = rn > delta
        w_eq[big] = delta / np.maximum(rn[big], 1e-12)
        # row weights are sqrt(w_eq) repeated for the 3 rows of each equation block
        w_row = np.repeat(np.sqrt(w_eq), 3)
        rq = np.quantile(rn, [0.5, 0.9, 0.99])
        print(f"[LiGT][IRLS {it+1}/{irls_iters}] residual norm median={med:.3e}, p90={rq[1]:.3e}, p99={rq[2]:.3e}, huber_delta={delta:.3e}")

    assert t_vec is not None
    t_red = t_vec.reshape(-1, 3)

    # insert ref translation = 0
    t_all = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        if i == ref_idx:
            continue
        ridx = i - 1 if i > ref_idx else i
        t_all[i] = t_red[ridx]

    # Step 6 sign disambiguation: median voting of a^T t_{xi;h}
    scores: List[float] = []
    sample_items = items[: min(sign_vote_tracks, len(items))]
    for _, obs in sample_items:
        if len(obs) < 3:
            continue
        obs = sorted(obs, key=lambda x: x[0])
        frames = [int(f) for (f, _) in obs]
        Xs = [np.asarray(X, dtype=np.float64).reshape(3,) for (_, X) in obs]

        best, best_u = choose_base_pair(
            frames,
            Xs,
            R_abs,
            max_candidates=base_pair_candidates,
            full_search_len=base_pair_full_search_len,
            min_gap=base_pair_min_gap,
            max_gap=base_pair_max_gap,
            rng=rng,
        )
        if best is None or best_u < u_min:
            continue
        p_idx, q_idx = best
        f_xi, f_h = frames[p_idx], frames[q_idx]
        X_xi, X_h = Xs[p_idx], Xs[q_idx]
        aT = compute_aT(R_abs[f_xi], R_abs[f_h], X_xi, X_h)
        t_xh = R_abs[f_h] @ (t_all[f_xi] - t_all[f_h])
        scores.append(float((aT @ t_xh.reshape(3, 1)).item()))

    if scores:
        med = float(np.median(scores))
        print(f"[LiGT] sign vote: median(a^T t_xh) = {med:.6e} (N={len(scores)})")
        if med < 0:
            t_all = -t_all
            print("[LiGT] flipped sign of all translations (Step 6).")

    # Normalize scale for numerical stability (evaluation may still do Sim(3) anyway)
    norms = [np.linalg.norm(t_all[i]) for i in range(N) if i != ref_idx]
    s = float(np.median(norms)) if norms else 1.0
    if s > 1e-9:
        t_all = t_all / s

    return t_all


def save_poses_w2c(path: str, R_abs: np.ndarray, t_world: np.ndarray):
    with open(path, "w") as f:
        for R, t in zip(R_abs, t_world):
            P = np.hstack([R, (-R @ t.reshape(3, 1))])
            f.write(" ".join(f"{v:.9e}" for v in P.reshape(-1)) + "\n")


def save_poses_c2w(path: str, R_abs: np.ndarray, t_world: np.ndarray):
    with open(path, "w") as f:
        for R, t in zip(R_abs, t_world):
            P = np.hstack([R.T, t.reshape(3, 1)])
            f.write(" ".join(f"{v:.9e}" for v in P.reshape(-1)) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track_npz_dir", required=True)
    ap.add_argument("--r_abs_npy", required=True)
    ap.add_argument("--out_dir", default="./poseonly_out")

    ap.add_argument("--dataset", choices=["kitti", "custom"], default="kitti")
    ap.add_argument("--calib_txt", default=None)
    ap.add_argument("--kitti_cam", default="P0")
    ap.add_argument("--K_npy", default=None)

    ap.add_argument("--ref_idx", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--min_track_len", type=int, default=5)
    ap.add_argument("--max_tracks", type=int, default=50000)
    ap.add_argument("--max_equations", type=int, default=2000000)

    # Paper Step-3 base pair selection
    ap.add_argument("--base_pair_candidates", type=int, default=80)
    ap.add_argument("--base_pair_full_search_len", type=int, default=50)
    ap.add_argument("--base_pair_min_gap", type=int, default=0)
    ap.add_argument("--base_pair_max_gap", type=int, default=0)

    # Weak-constraint filters (critical for translation)
    ap.add_argument("--u_min", type=float, default=1e-3)
    ap.add_argument("--g_min", type=float, default=1e-3)

    # Robustification
    ap.add_argument("--irls_iters", type=int, default=2)
    ap.add_argument("--irls_huber_k", type=float, default=1.5)

    ap.add_argument("--sign_vote_tracks", type=int, default=2000)
    ap.add_argument("--solver", choices=["auto", "eigsh", "lobpcg"], default="auto")
    ap.add_argument("--eq_norm", choices=["fro", "u2", "none"], default="fro")
    ap.add_argument("--gt_pose_txt", type=str, default=None, help="KITTI GT poses txt (c2w) for diagnostic")
    ap.add_argument("--run_pa", action="store_true", help="Run optional PA refinement after LiGT.")
    ap.add_argument("--pa_iters", type=int, default=3, help="Number of alternating PA refinement rounds.")
    ap.add_argument("--pa_point_rms", type=float, default=5e-3, help="Max LS residual RMS for keeping a triangulated point.")
    ap.add_argument("--pa_no_refine_rot", action="store_true", help="Keep rotations fixed during PA.")
    ap.add_argument("--pa_no_refine_trans", action="store_true", help="Keep translations fixed during PA.")
    ap.add_argument("--pa_max_nfev", type=int, default=50, help="Max function evaluations per PA outer iteration.")
    ap.add_argument("--pa_loss", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"], default="soft_l1")
    ap.add_argument("--pa_f_scale", type=float, default=1e-3, help="Robust loss scale used by scipy.least_squares.")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # K
    if args.K_npy:
        K = np.load(args.K_npy).astype(np.float64)
    else:
        if args.dataset == "kitti":
            if args.calib_txt is None:
                raise ValueError("KITTI needs --calib_txt (or provide --K_npy)")
            K = load_kitti_K_from_calib(args.calib_txt, cam=args.kitti_cam)
        else:
            raise ValueError("custom dataset needs --K_npy")
    print("[K]\n", K)

    R_abs = np.load(args.r_abs_npy).astype(np.float64)
    assert R_abs.ndim == 3 and R_abs.shape[1:] == (3, 3)

    tracks_obs, npz_files = build_tracks_obs(args.track_npz_dir, K, max_frames=args.max_frames)
    T = min(len(npz_files), R_abs.shape[0])
    R_abs = R_abs[:T]

    tracks_pruned = {}
    for tid, obs in tracks_obs.items():
        obs2 = [(f, X) for (f, X) in obs if 0 <= f < T]
        if len(obs2) >= args.min_track_len:
            tracks_pruned[tid] = obs2

    gt_t_all = None
    if args.gt_pose_txt:
        Ts = []
        with open(args.gt_pose_txt, 'r') as f:
            for ln in f:
                vals = [float(x) for x in ln.split()]
                Tm = np.eye(4, dtype=np.float64)
                Tm[:3, :4] = np.array(vals, dtype=np.float64).reshape(3, 4)
                Ts.append(Tm)
        Ts = np.stack(Ts, axis=0)
        gt_t_all = Ts[:T, :3, 3].copy()  # camera centers in world (c2w)
        print('[GT] loaded', args.gt_pose_txt, 't shape=', gt_t_all.shape)

    t_ligt = solve_ligt_sparse(
        tracks_pruned,
        R_abs,
        ref_idx=args.ref_idx,
        min_track_len=args.min_track_len,
        max_tracks=args.max_tracks,
        max_equations=args.max_equations,
        base_pair_candidates=args.base_pair_candidates,
        base_pair_full_search_len=args.base_pair_full_search_len,
        base_pair_min_gap=args.base_pair_min_gap,
        base_pair_max_gap=args.base_pair_max_gap,
        u_min=args.u_min,
        g_min=args.g_min,
        eq_norm=(args.eq_norm if args.eq_norm != "none" else "none"),
        solver=args.solver,
        sign_vote_tracks=args.sign_vote_tracks,
        irls_iters=args.irls_iters,
        irls_huber_k=args.irls_huber_k,
        gt_t_all=gt_t_all,
    )

    np.save(os.path.join(args.out_dir, "t_all_ligt.npy"), t_ligt)
    np.save(os.path.join(args.out_dir, "R_abs_used.npy"), R_abs)
    save_poses_w2c(os.path.join(args.out_dir, "poses_w2c_ligt.txt"), R_abs, t_ligt)
    save_poses_c2w(os.path.join(args.out_dir, "poses_c2w_ligt.txt"), R_abs, t_ligt)

    t_all = t_ligt
    if args.run_pa:
        try:
            R_pa, t_all, pts3d = run_pose_adjustment(
                tracks_pruned,
                R_abs,
                t_ligt,
                ref_idx=args.ref_idx,
                min_track_len=args.min_track_len,
                pa_iters=args.pa_iters,
                pa_point_rms=args.pa_point_rms,
                pa_refine_rotations=(not args.pa_no_refine_rot),
                pa_refine_translations=(not args.pa_no_refine_trans),
                pa_max_nfev=args.pa_max_nfev,
                pa_loss=args.pa_loss,
                pa_f_scale=args.pa_f_scale,
            )
            R_abs = R_pa
            np.save(os.path.join(args.out_dir, "t_all_pa.npy"), t_all)
            np.save(os.path.join(args.out_dir, "pa_points.npy"), np.stack(list(pts3d.values()), axis=0).astype(np.float64))
            save_poses_w2c(os.path.join(args.out_dir, "poses_w2c_pa.txt"), R_abs, t_all)
            save_poses_c2w(os.path.join(args.out_dir, "poses_c2w_pa.txt"), R_abs, t_all)
        except RuntimeError as e:
            print(f"[PA] skipped: {e}")

    np.save(os.path.join(args.out_dir, "t_all.npy"), t_all)
    save_poses_w2c(os.path.join(args.out_dir, "poses_w2c.txt"), R_abs, t_all)
    save_poses_c2w(os.path.join(args.out_dir, "poses_c2w.txt"), R_abs, t_all)
    print("[Done] saved to", args.out_dir)


if __name__ == "__main__":
    main()

    
# python tools/Pose_Only_patched_v3_fixed.py \
#   --track_npz_dir runs/strecha/fountain-P11/tracks_npz \
#   --r_abs_npy data/prepared/strecha/fountain-P11/R_abs_gt_w2c.npy \
#   --dataset custom \
#   --K_npy data/prepared/strecha/fountain-P11/K.npy \
#   --out_dir runs/strecha/fountain-P11/poseonly_gt \
#   --min_track_len 5 \
#   --max_tracks 20000 \
#   --u_min 1e-3 \
#   --g_min 1e-3 \
#   --base_pair_candidates 80 \
#   --base_pair_full_search_len 50 \
#   --irls_iters 2 \
#   --gt_pose_txt data/prepared/strecha/fountain-P11/gt_poses_c2w.txt




# python tools/Pose_Only_patched_v3_fixed.py \
#   --track_npz_dir runs/strecha/fountain-P11/tracks_npz \
#   --r_abs_npy runs/strecha/fountain-P11/rraa_output/R_abs_fountain-P11.npy \
#   --dataset custom \
#   --K_npy data/prepared/strecha/fountain-P11/K.npy \
#   --out_dir runs/strecha/fountain-P11/poseonly_rraa \
#   --min_track_len 5 \
#   --max_tracks 20000 \
#   --u_min 1e-3 \
#   --g_min 1e-3 \
#   --base_pair_candidates 80 \
#   --base_pair_full_search_len 50 \
#   --irls_iters 2 \
#   --gt_pose_txt data/prepared/strecha/fountain-P11/gt_poses_c2w.txt
