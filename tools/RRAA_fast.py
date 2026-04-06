import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, diags, eye
from scipy.sparse.linalg import splu, spsolve, cg

from degeneracy_utils import dump_json, safe_ratio, summarize_array
from quality_utils import load_quality_config, make_quality_config_record, save_quality_config_record


def build_adjacency_lists(N: int, edges: np.ndarray):
    adj = [[] for _ in range(int(N))]
    for i, j in np.asarray(edges, dtype=np.int64):
        ii = int(i)
        jj = int(j)
        if ii < 0 or ii >= N or jj < 0 or jj >= N:
            continue
        adj[ii].append(jj)
        adj[jj].append(ii)
    return adj


def connected_components_from_edges(N: int, edges: np.ndarray):
    adj = build_adjacency_lists(N, edges)
    vis = np.zeros(int(N), dtype=bool)
    comps = []
    for start in range(int(N)):
        if vis[start]:
            continue
        stack = [start]
        vis[start] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(int(u))
            for v in adj[u]:
                if not vis[v]:
                    vis[v] = True
                    stack.append(v)
        comps.append(sorted(comp))
    comps.sort(key=len, reverse=True)
    return comps


def graph_diagnostics(N: int, edges: np.ndarray, anchor: int = 0):
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    deg = np.zeros(int(N), dtype=np.int64)
    for i, j in edges:
        deg[int(i)] += 1
        deg[int(j)] += 1
    comps = connected_components_from_edges(N, edges)
    anchor_comp_size = 0
    for comp in comps:
        if int(anchor) in comp:
            anchor_comp_size = len(comp)
            break
    top_degree_nodes = np.argsort(-deg, kind="stable")[: min(5, int(N))]
    return {
        "node_count": int(N),
        "edge_count": int(edges.shape[0]),
        "component_count": int(len(comps)),
        "largest_component_size": int(len(comps[0])) if comps else 0,
        "largest_component_ratio": safe_ratio(len(comps[0]) if comps else 0, max(int(N), 1)),
        "anchor": int(anchor),
        "anchor_degree": int(deg[int(anchor)]) if 0 <= int(anchor) < int(N) else None,
        "anchor_component_size": int(anchor_comp_size),
        "anchor_component_ratio": safe_ratio(anchor_comp_size, max(int(N), 1)),
        "is_connected": bool(len(comps) <= 1),
        "degree_stats": summarize_array(deg.astype(np.float64)),
        "isolated_nodes": [int(i) for i in np.where(deg == 0)[0].tolist()],
        "components_top": [
            {"size": int(len(comp)), "nodes": [int(v) for v in comp[:10]]}
            for comp in comps[:5]
        ],
        "top_degree_nodes": [
            {"node": int(v), "degree": int(deg[int(v)])}
            for v in top_degree_nodes.tolist()
        ],
    }

def so3_log(rot_mats: np.ndarray) -> np.ndarray:
    return R.from_matrix(rot_mats).as_rotvec()

def so3_exp(rotvecs: np.ndarray) -> np.ndarray:
    return R.from_rotvec(rotvecs).as_matrix()

def robust_weight(x: np.ndarray, loss: str = "l1_2", a: float = 5.0 * np.pi / 180.0, eps: float = 1e-12):
    x = np.asarray(x, dtype=np.float64)
    ax = np.maximum(np.abs(x), eps)

    if loss == "l2":
        return np.ones_like(ax)
    if loss == "l1":
        return 1.0 / ax
    if loss == "l1_2":
        return ax ** (-1.5)
    if loss == "geman":
        denom = (a * a + ax * ax)
        return (a * a) / (denom * denom)
    if loss == "huber":
        w = np.ones_like(ax)
        m = ax > a
        w[m] = a / ax[m]
        return w
    raise ValueError(f"Unknown loss: {loss}")

def build_incidence_B(N: int, edges: np.ndarray, anchor: int = 0) -> csr_matrix:
    """B: (M, N-1) incidence matrix with anchor column removed. Each row k (i,j): +1 at i, -1 at j."""
    edges = np.asarray(edges, dtype=np.int64)
    M = edges.shape[0]

    var_index = -np.ones(N, dtype=np.int64)
    cnt = 0
    for v in range(N):
        if v == anchor:
            continue
        var_index[v] = cnt
        cnt += 1

    rows, cols, data = [], [], []
    for k in range(M):
        i, j = int(edges[k, 0]), int(edges[k, 1])
        if i != anchor:
            rows.append(k); cols.append(int(var_index[i])); data.append(+1.0)
        if j != anchor:
            rows.append(k); cols.append(int(var_index[j])); data.append(-1.0)

    B = coo_matrix((data, (rows, cols)), shape=(M, N - 1)).tocsr()
    return B

def unstack_var_to_full(x_var: np.ndarray, N: int, anchor: int = 0) -> np.ndarray:
    """x_var: (N-1,3) -> full (N,3) with anchor row = 0."""
    x_var = np.asarray(x_var, dtype=np.float64).reshape(N-1, 3)
    out = np.zeros((N, 3), dtype=np.float64)
    idx = 0
    for i in range(N):
        if i == anchor:
            continue
        out[i] = x_var[idx]
        idx += 1
    return out

def compute_edge_residuals(R_abs: np.ndarray, edges: np.ndarray, Rij: np.ndarray) -> np.ndarray:
    """dvE_k = log( R_j^{-1} * Rij * R_i )  ; should be ~0 when consistent."""
    edges = np.asarray(edges, dtype=np.int64)
    i = edges[:, 0]
    j = edges[:, 1]
    RjT = np.transpose(R_abs[j], (0, 2, 1))
    Ri  = R_abs[i]
    DRij = RjT @ Rij @ Ri
    return so3_log(DRij)  # (M,3)

def shrink(x, kappa):
    return np.sign(x) * np.maximum(np.abs(x) - kappa, 0.0)

def solve_l1_admm(B: csr_matrix, b: np.ndarray, rho: float = 1.0, max_iter: int = 200, tol: float = 1e-4):
    """
    Solve: min_x ||B x + b||_1   via ADMM on z = Bx + b
      x-update: (B^T B) x = B^T (z - u - b)
      z-update: shrink(Bx + b + u, 1/rho)
      u-update: u += (Bx + b - z)
    """
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    M, n = B.shape

    # factorize (B^T B + eps I) once
    BtB = (B.T @ B).tocsc()
    eps = 1e-9
    lu = splu(BtB + eps * eye(n, format="csc"))

    x = np.zeros(n, dtype=np.float64)
    z = np.zeros(M, dtype=np.float64)
    u = np.zeros(M, dtype=np.float64)

    for it in range(max_iter):
        rhs = B.T @ (z - u - b)
        x = lu.solve(rhs)

        r = B @ x + b
        z_new = shrink(r + u, 1.0 / rho)
        u = u + (r - z_new)
        z = z_new

        # stopping (primal residual)
        pr = np.linalg.norm(r - z) / np.sqrt(M)
        if pr < tol:
            break

    return x


def apply_row_weights(B: csr_matrix, b: np.ndarray, weights: np.ndarray):
    sw = np.sqrt(np.asarray(weights, dtype=np.float64).reshape(-1))
    return B.multiply(sw[:, None]), sw * np.asarray(b, dtype=np.float64).reshape(-1)

def spanning_tree_init(N: int, edges: np.ndarray, Rij: np.ndarray, anchor: int = 0) -> np.ndarray:
    """BFS init; requires connected graph."""
    edges = np.asarray(edges, dtype=np.int64)
    Rij = np.asarray(Rij, dtype=np.float64)

    adj = [[] for _ in range(N)]
    for k in range(edges.shape[0]):
        i, j = int(edges[k,0]), int(edges[k,1])
        R_ij = Rij[k]
        adj[i].append((j, R_ij))
        adj[j].append((i, R_ij.T))

    R_init = np.tile(np.eye(3), (N,1,1))
    vis = np.zeros(N, dtype=bool)
    vis[anchor] = True
    stack = [anchor]
    while stack:
        u = stack.pop()
        Ru = R_init[u]
        for v, R_uv in adj[u]:
            if vis[v]:
                continue
            R_init[v] = R_uv @ Ru
            vis[v] = True
            stack.append(v)

    if not np.all(vis):
        raise RuntimeError(f"Graph not connected from anchor={anchor}: visited {vis.sum()}/{N}")
    return R_init

def initial_l1_stage(N, edges, Rij, R_init, anchor=0, iters=3,
                     rho=1.0, admm_iter=150, admm_tol=1e-4, edge_weights=None):
    B = build_incidence_B(N, edges, anchor=anchor)
    R_abs = R_init.copy()
    R_abs[anchor] = np.eye(3)
    if edge_weights is None:
        edge_weights = np.ones(edges.shape[0], dtype=np.float64)
    edge_weights = np.asarray(edge_weights, dtype=np.float64).reshape(-1)

    for _ in range(iters):
        dvE = compute_edge_residuals(R_abs, edges, Rij)  # (M,3)

        # solve 3 independent L1 problems
        Bw0, b0 = apply_row_weights(B, dvE[:, 0], edge_weights)
        Bw1, b1 = apply_row_weights(B, dvE[:, 1], edge_weights)
        Bw2, b2 = apply_row_weights(B, dvE[:, 2], edge_weights)
        x0 = solve_l1_admm(Bw0, b0, rho=rho, max_iter=admm_iter, tol=admm_tol)
        x1 = solve_l1_admm(Bw1, b1, rho=rho, max_iter=admm_iter, tol=admm_tol)
        x2 = solve_l1_admm(Bw2, b2, rho=rho, max_iter=admm_iter, tol=admm_tol)
        x_var = np.stack([x0, x1, x2], axis=1)  # (N-1,3)

        dV = unstack_var_to_full(x_var, N, anchor=anchor)
        R_abs = R_abs @ so3_exp(dV)
        R_abs[anchor] = np.eye(3)

        if np.linalg.norm(dV) < 1e-6:
            break

    return R_abs

def irls_stage(N, edges, Rij, R_init, anchor=0, loss="l1_2", a=5*np.pi/180, tol=1e-6, max_iter=50, edge_weights=None):
    B = build_incidence_B(N, edges, anchor=anchor)
    R_abs = R_init.copy()
    R_abs[anchor] = np.eye(3)
    if edge_weights is None:
        edge_weights = np.ones(edges.shape[0], dtype=np.float64)
    edge_weights = np.asarray(edge_weights, dtype=np.float64).reshape(-1)

    for _ in range(max_iter):
        dvE = compute_edge_residuals(R_abs, edges, Rij)  # (M,3)
        s = np.linalg.norm(dvE, axis=1)
        w = edge_weights * robust_weight(s, loss=loss, a=a)  # (M,)

        # build weighted normal matrix L = B^T W B
        # (use Bw = sqrt(W)*B to get SPD matrix)
        sw = np.sqrt(w).astype(np.float64)
        Bw = B.multiply(sw[:, None])
        L = (Bw.T @ Bw).tocsr()

        dV_var = np.zeros((N-1, 3), dtype=np.float64)
        for c in range(3):
            rhs = -(Bw.T @ (sw * dvE[:, c]))
            try:
                dV_var[:, c] = spsolve(L, rhs)
            except Exception:
                x, info = cg(L, rhs, maxiter=2000)
                if info != 0:
                    raise RuntimeError(f"CG failed info={info}")
                dV_var[:, c] = x

        dV = unstack_var_to_full(dV_var, N, anchor=anchor)

        if not np.isfinite(dV).all():
            raise RuntimeError("NaN/Inf in update dV (graph/weights ill-conditioned)")

        if np.linalg.norm(dV) < tol:
            break

        R_abs = R_abs @ so3_exp(dV)
        R_abs[anchor] = np.eye(3)

    return R_abs

def rraa_solve(N, edges, Rij, anchor=0,
              use_initial_l1=True, initial_l1_iters=3,
              irls_max_iter=50, loss="l1_2", a=5*np.pi/180, tol=1e-6, edge_weights=None):
    R0 = spanning_tree_init(N, edges, Rij, anchor=anchor)
    if use_initial_l1 and initial_l1_iters > 0:
        R1 = initial_l1_stage(N, edges, Rij, R0, anchor=anchor, iters=initial_l1_iters, edge_weights=edge_weights)
    else:
        R1 = R0
    R_abs = irls_stage(N, edges, Rij, R1, anchor=anchor, loss=loss, a=a, tol=tol, max_iter=irls_max_iter, edge_weights=edge_weights)
    return R_abs


def align_solution_gauge(reference: np.ndarray, candidate: np.ndarray, align_idx: int = 0) -> np.ndarray:
    G = np.transpose(candidate[int(align_idx)]) @ reference[int(align_idx)]
    return candidate @ G


def relative_rotation_angles_deg(R_a: np.ndarray, R_b: np.ndarray, align_idx: int = 0) -> np.ndarray:
    R_b_aligned = align_solution_gauge(R_a, R_b, align_idx=align_idx)
    delta = np.transpose(R_b_aligned, (0, 2, 1)) @ R_a
    return np.degrees(np.linalg.norm(so3_log(delta), axis=1))


def anchor_sensitivity_diagnostics(
    *,
    N: int,
    edges: np.ndarray,
    Rij: np.ndarray,
    base_anchor: int,
    base_solution: np.ndarray,
    use_initial_l1: bool,
    initial_l1_iters: int,
    irls_max_iter: int,
    loss: str,
    a: float,
    tol: float,
    edge_weights: np.ndarray | None,
    max_alt_anchors: int = 3,
):
    graph_diag = graph_diagnostics(N, edges, anchor=base_anchor)
    candidates = [item["node"] for item in graph_diag.get("top_degree_nodes", [])]
    alt_anchors = [int(v) for v in candidates if int(v) != int(base_anchor)]
    alt_anchors = alt_anchors[: max(0, int(max_alt_anchors))]
    out = {
        "base_anchor": int(base_anchor),
        "tested_anchors": [int(base_anchor)],
        "comparisons": [],
    }
    for alt_anchor in alt_anchors:
        try:
            alt_solution = rraa_solve(
                N,
                edges,
                Rij,
                anchor=alt_anchor,
                use_initial_l1=use_initial_l1,
                initial_l1_iters=initial_l1_iters,
                irls_max_iter=irls_max_iter,
                loss=loss,
                a=a,
                tol=tol,
                edge_weights=edge_weights,
            )
            ang = relative_rotation_angles_deg(base_solution, alt_solution, align_idx=base_anchor)
            out["tested_anchors"].append(int(alt_anchor))
            out["comparisons"].append(
                {
                    "anchor": int(alt_anchor),
                    "median_deg": float(np.median(ang)),
                    "p90_deg": float(np.quantile(ang, 0.9)),
                    "max_deg": float(np.max(ang)),
                }
            )
        except Exception as exc:
            out["comparisons"].append(
                {
                    "anchor": int(alt_anchor),
                    "status": "failed",
                    "error": str(exc),
                }
            )
    return out


def filter_edges_by_quality(edges, Rij, q_pair, threshold):
    q_pair = np.asarray(q_pair, dtype=np.float64).reshape(-1)
    keep = q_pair >= float(threshold)
    if not np.any(keep):
        raise RuntimeError(f"All RRAA edges removed by q_pair threshold={threshold}")
    return edges[keep], Rij[keep], q_pair[keep], keep

if __name__ == "__main__":
    import argparse, os, sys
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--anchor", type=int, default=0)
    p.add_argument("--no_initial_l1", action="store_true")
    p.add_argument("--initial_l1_iters", type=int, default=3)
    p.add_argument("--irls_max_iter", type=int, default=50)
    p.add_argument("--loss", type=str, default="l1_2", choices=["l2","l1","l1_2","geman","huber"])
    p.add_argument("--a_deg", type=float, default=5.0)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--enable_quality_weighting", action="store_true")
    p.add_argument("--rraa_use_qpair_weight", action="store_true")
    p.add_argument("--qpair_threshold", type=float, default=None)
    p.add_argument("--dump_quality_stats", action="store_true")
    p.add_argument("--diagnose_anchor_sensitivity", action="store_true")
    p.add_argument("--anchor_sensitivity_max_anchors", type=int, default=3)
    p.add_argument("--quality_config", type=str, default=None)
    args = p.parse_args()
    cli_args = sys.argv[1:]
    cli_qpair_threshold = "--qpair_threshold" in cli_args

    quality_payload = load_quality_config(args.quality_config)
    rraa_cfg = quality_payload.get("rraa", {}) if quality_payload else {}
    rraa_cfg_resolved = dict(rraa_cfg)
    if args.qpair_threshold is None and "qpair_threshold" in rraa_cfg:
        args.qpair_threshold = float(rraa_cfg["qpair_threshold"])
    if args.qpair_threshold is None:
        args.qpair_threshold = 0.0
    if "enable_quality_weighting" in rraa_cfg:
        args.enable_quality_weighting = bool(rraa_cfg["enable_quality_weighting"])
    if "rraa_use_qpair_weight" in rraa_cfg:
        args.rraa_use_qpair_weight = bool(rraa_cfg["rraa_use_qpair_weight"])
    if cli_qpair_threshold:
        rraa_cfg_resolved["qpair_threshold"] = float(args.qpair_threshold)

    d = np.load(args.npz)
    N = int(d["N"]); edges = d["edges"]; Rij = d["Rij"].astype(np.float64)
    q_pair = d["q_pair"].astype(np.float64) if "q_pair" in d.files else None

    diag = {
        "input_edges": int(edges.shape[0]),
        "config": vars(args),
        "graph_before_filter": graph_diagnostics(N, edges, anchor=args.anchor),
    }
    keep_mask = None
    edge_weights = None
    if args.enable_quality_weighting:
        if q_pair is None:
            print("[RRAA] quality weighting requested but q_pair is missing in npz; fallback to uniform weights.")
        else:
            if args.qpair_threshold > 0:
                edges, Rij, q_pair, keep_mask = filter_edges_by_quality(edges, Rij, q_pair, args.qpair_threshold)
            if args.rraa_use_qpair_weight:
                edge_weights = np.clip(q_pair, 1e-3, 1.0)
            diag["q_pair"] = summarize_array(q_pair)
            diag["kept_edges"] = int(edges.shape[0])
            diag["kept_ratio"] = safe_ratio(edges.shape[0], diag["input_edges"])
    diag["graph_after_filter"] = graph_diagnostics(N, edges, anchor=args.anchor)
    if diag["graph_before_filter"]["component_count"] != diag["graph_after_filter"]["component_count"]:
        diag["graph_fragmentation_delta"] = {
            "component_count_delta": int(diag["graph_after_filter"]["component_count"] - diag["graph_before_filter"]["component_count"]),
            "largest_component_ratio_delta": float(diag["graph_after_filter"]["largest_component_ratio"] - diag["graph_before_filter"]["largest_component_ratio"]),
            "anchor_component_ratio_delta": float(diag["graph_after_filter"]["anchor_component_ratio"] - diag["graph_before_filter"]["anchor_component_ratio"]),
        }

    R_abs = rraa_solve(
        N, edges, Rij,
        anchor=args.anchor,
        use_initial_l1=(not args.no_initial_l1),
        initial_l1_iters=args.initial_l1_iters,
        irls_max_iter=args.irls_max_iter,
        loss=args.loss,
        a=args.a_deg * np.pi/180.0,
        tol=args.tol,
        edge_weights=edge_weights,
    )
    if args.diagnose_anchor_sensitivity:
        diag["anchor_sensitivity"] = anchor_sensitivity_diagnostics(
            N=N,
            edges=edges,
            Rij=Rij,
            base_anchor=args.anchor,
            base_solution=R_abs,
            use_initial_l1=(not args.no_initial_l1),
            initial_l1_iters=args.initial_l1_iters,
            irls_max_iter=args.irls_max_iter,
            loss=args.loss,
            a=args.a_deg * np.pi / 180.0,
            tol=args.tol,
            edge_weights=edge_weights,
            max_alt_anchors=args.anchor_sensitivity_max_anchors,
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, R_abs)
    quality_record = make_quality_config_record(
        stage="rraa_fast",
        section="rraa",
        raw_payload=quality_payload,
        resolved_section=rraa_cfg_resolved,
        effective_section={
            "enable_quality_weighting": bool(args.enable_quality_weighting),
            "rraa_use_qpair_weight": bool(args.rraa_use_qpair_weight),
            "qpair_threshold": float(args.qpair_threshold),
            "loss": args.loss,
            "a_deg": float(args.a_deg),
        },
        mode=args.loss,
        auto_quality_refs=False,
        quality_config_path=args.quality_config,
        extra={
            "output_path": os.path.splitext(args.out)[0] + "_quality_config_used.json",
        },
    )
    save_quality_config_record(os.path.splitext(args.out)[0] + "_quality_config_used.json", quality_record)
    print("Saved:", args.out, "shape=", R_abs.shape)
    if args.enable_quality_weighting and q_pair is not None:
        print(f"[RRAA] q_pair threshold={args.qpair_threshold:.3f}, kept_edges={edges.shape[0]}")
        if edge_weights is not None:
            print(f"[RRAA] q_pair weighting on, median={np.median(edge_weights):.3f}, p10={np.quantile(edge_weights, 0.1):.3f}, p90={np.quantile(edge_weights, 0.9):.3f}")
    if args.dump_quality_stats:
        diag["quality_config_used"] = quality_record
        dump_json(
            os.path.splitext(args.out)[0] + "_stats.json",
            diag,
        )


# python tools/RRAA_fast.py \
#   --npz runs/strecha/fountain-P11/rraa_input/rraa_input_fountain-P11.npz \
#   --out runs/strecha/fountain-P11/rraa_output/R_abs_fountain-P11.npy \
#   --initial_l1_iters 3 \
#   --loss l1_2 \
#   --a_deg 5 \
#   --irls_max_iter 50  --gt_pose_txt 
