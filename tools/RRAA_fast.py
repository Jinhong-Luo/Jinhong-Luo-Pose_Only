import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, diags, eye
from scipy.sparse.linalg import splu, spsolve, cg

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
                     rho=1.0, admm_iter=150, admm_tol=1e-4):
    B = build_incidence_B(N, edges, anchor=anchor)
    R_abs = R_init.copy()
    R_abs[anchor] = np.eye(3)

    for _ in range(iters):
        dvE = compute_edge_residuals(R_abs, edges, Rij)  # (M,3)

        # solve 3 independent L1 problems
        x0 = solve_l1_admm(B, dvE[:,0], rho=rho, max_iter=admm_iter, tol=admm_tol)
        x1 = solve_l1_admm(B, dvE[:,1], rho=rho, max_iter=admm_iter, tol=admm_tol)
        x2 = solve_l1_admm(B, dvE[:,2], rho=rho, max_iter=admm_iter, tol=admm_tol)
        x_var = np.stack([x0, x1, x2], axis=1)  # (N-1,3)

        dV = unstack_var_to_full(x_var, N, anchor=anchor)
        R_abs = R_abs @ so3_exp(dV)
        R_abs[anchor] = np.eye(3)

        if np.linalg.norm(dV) < 1e-6:
            break

    return R_abs

def irls_stage(N, edges, Rij, R_init, anchor=0, loss="l1_2", a=5*np.pi/180, tol=1e-6, max_iter=50):
    B = build_incidence_B(N, edges, anchor=anchor)
    R_abs = R_init.copy()
    R_abs[anchor] = np.eye(3)

    for _ in range(max_iter):
        dvE = compute_edge_residuals(R_abs, edges, Rij)  # (M,3)
        s = np.linalg.norm(dvE, axis=1)
        w = robust_weight(s, loss=loss, a=a)  # (M,)

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
              irls_max_iter=50, loss="l1_2", a=5*np.pi/180, tol=1e-6):
    R0 = spanning_tree_init(N, edges, Rij, anchor=anchor)
    if use_initial_l1 and initial_l1_iters > 0:
        R1 = initial_l1_stage(N, edges, Rij, R0, anchor=anchor, iters=initial_l1_iters)
    else:
        R1 = R0
    R_abs = irls_stage(N, edges, Rij, R1, anchor=anchor, loss=loss, a=a, tol=tol, max_iter=irls_max_iter)
    return R_abs

if __name__ == "__main__":
    import argparse, os
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
    args = p.parse_args()

    d = np.load(args.npz)
    N = int(d["N"]); edges = d["edges"]; Rij = d["Rij"].astype(np.float64)

    R_abs = rraa_solve(
        N, edges, Rij,
        anchor=args.anchor,
        use_initial_l1=(not args.no_initial_l1),
        initial_l1_iters=args.initial_l1_iters,
        irls_max_iter=args.irls_max_iter,
        loss=args.loss,
        a=args.a_deg * np.pi/180.0,
        tol=args.tol
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, R_abs)
    print("Saved:", args.out, "shape=", R_abs.shape)


# python tools/RRAA_fast.py \
#   --npz runs/strecha/fountain-P11/rraa_input/rraa_input_fountain-P11.npz \
#   --out runs/strecha/fountain-P11/rraa_output/R_abs_fountain-P11.npy \
#   --initial_l1_iters 3 \
#   --loss l1_2 \
#   --a_deg 5 \
#   --irls_max_iter 50  --gt_pose_txt 
