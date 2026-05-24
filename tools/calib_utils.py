import os
import glob
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".ppm", ".bmp", ".tif", ".tiff"}


def read_kitti_K(calib_txt, cam="P0"):
    with open(calib_txt, "r") as f:
        for line in f:
            if line.startswith(cam + ":"):
                vals = [float(x) for x in line.split()[1:]]
                P = np.array(vals, np.float64).reshape(3, 4)
                return P[:, :3].copy()
    raise RuntimeError(f"Cannot find {cam}: in {calib_txt}")


def read_euroc_cam_yaml(sensor_yaml):
    import yaml

    with open(sensor_yaml, "r") as f:
        y = yaml.safe_load(f)
    fx, fy, cx, cy = y["intrinsics"]
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)
    dist = np.array(y.get("distortion_coefficients", []), dtype=np.float64).reshape(-1)
    model = y.get("distortion_model", "radial-tangential")
    return K, dist, model


def read_custom_K(K_npy):
    K = np.load(K_npy).astype(np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K must be (3,3), got {K.shape}")
    return K


def read_custom_Ks(Ks_npy, image_K_idx_npy=None):
    Ks = np.load(Ks_npy).astype(np.float64)
    if Ks.ndim != 3 or Ks.shape[1:] != (3, 3):
        raise ValueError(f"Ks must be (M,3,3), got {Ks.shape}")

    image_K_idx = None
    if image_K_idx_npy is not None:
        image_K_idx = np.load(image_K_idx_npy).astype(np.int64).reshape(-1)
        if image_K_idx.size > 0:
            if np.min(image_K_idx) < 0 or np.max(image_K_idx) >= Ks.shape[0]:
                raise ValueError(
                    f"image_K_idx has values outside [0, {Ks.shape[0] - 1}]: "
                    f"min={np.min(image_K_idx)}, max={np.max(image_K_idx)}"
                )
    return Ks, image_K_idx


def load_intrinsics(dataset, kitti_calib=None, kitti_cam="P0",
                    euroc_yaml=None, K_npy=None):
    K = None
    dist = None
    model = None

    if dataset == "kitti":
        if kitti_calib is None:
            raise ValueError("--dataset kitti requires --kitti_calib")
        K = read_kitti_K(kitti_calib, cam=kitti_cam)

    elif dataset == "euroc":
        if euroc_yaml is None:
            raise ValueError("--dataset euroc requires --euroc_yaml")
        K, dist, model = read_euroc_cam_yaml(euroc_yaml)

    elif dataset == "custom":
        if K_npy is None:
            raise ValueError("--dataset custom requires --K_npy")
        K = read_custom_K(K_npy)

    elif dataset == "none":
        pass

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return K, dist, model


def list_images_sorted(img_dir):
    files = []
    for name in os.listdir(img_dir):
        path = os.path.join(img_dir, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
            files.append(path)
    files = sorted(files)
    if len(files) == 0:
        raise RuntimeError(f"No images found in {img_dir}")
    return files


def infer_kitti_scene_id(kitti_calib=None, image_glob=None, img_dir=None):
    candidates = [kitti_calib, image_glob, img_dir]
    for path in candidates:
        if not path:
            continue
        norm = os.path.normpath(path.replace("*", ""))
        parts = [p for p in norm.split(os.sep) if p]
        for idx, part in enumerate(parts):
            if part.lower() == "image_0" and idx > 0:
                return parts[idx - 1]
        if parts:
            stem = parts[-1]
            if stem.lower() in {"calib.txt", "times.txt"} and len(parts) >= 2:
                return parts[-2]
    return None


def build_kitti_candidate_paths(scene_id, image_subdir="image_0", image_pattern="*.png"):
    if not scene_id:
        return []
    roots = [
        os.path.join("data", "raw", "KITTI", "sequences"),
        os.path.join("data", "raw", "KITTI"),
        os.path.join("data", "raw", "kitti", "sequences"),
        os.path.join("data", "raw", "kitti"),
    ]
    candidates = []
    for root in roots:
        scene_root = os.path.join(root, scene_id)
        candidates.append({
            "scene_root": scene_root,
            "calib": os.path.join(scene_root, "calib.txt"),
            "img_dir": os.path.join(scene_root, image_subdir),
            "image_glob": os.path.join(scene_root, image_subdir, image_pattern),
        })
    return candidates


def resolve_kitti_scene_inputs(kitti_calib=None, image_glob=None, img_dir=None):
    scene_id = infer_kitti_scene_id(kitti_calib=kitti_calib, image_glob=image_glob, img_dir=img_dir)
    image_subdir = "image_0"
    image_pattern = "*.png"

    if image_glob:
        norm_glob = image_glob.replace("\\", "/")
        image_pattern = os.path.basename(image_glob) or image_pattern
        if image_pattern == "*":
            image_pattern = "*.png"
        parts = [p for p in norm_glob.split("/") if p]
        for idx, part in enumerate(parts):
            if part.lower().startswith("image_") and part != image_pattern:
                image_subdir = part
                break
    elif img_dir:
        image_subdir = os.path.basename(os.path.normpath(img_dir)) or image_subdir

    tried = []
    for cand in build_kitti_candidate_paths(scene_id, image_subdir=image_subdir, image_pattern=image_pattern):
        entry = {
            "scene_root": cand["scene_root"],
            "calib": cand["calib"],
            "img_dir": cand["img_dir"],
            "image_glob": cand["image_glob"],
        }
        tried.append(entry)
        has_calib = os.path.isfile(cand["calib"])
        has_img_dir = os.path.isdir(cand["img_dir"])
        has_images = bool(glob.glob(cand["image_glob"]))
        if has_calib and has_img_dir and has_images:
            return {
                "scene_id": scene_id,
                "kitti_calib": cand["calib"],
                "img_dir": cand["img_dir"],
                "image_glob": cand["image_glob"],
                "tried": tried,
            }

    return {
        "scene_id": scene_id,
        "kitti_calib": kitti_calib,
        "img_dir": img_dir,
        "image_glob": image_glob,
        "tried": tried,
    }


def format_kitti_layout_help(scene_id, image_subdir="image_0"):
    scene_id = scene_id or "<scene_id>"
    lines = ["Expected one of these KITTI layouts:"]
    for cand in build_kitti_candidate_paths(scene_id, image_subdir=image_subdir):
        lines.append(f"  - {cand['calib']}")
        lines.append(f"    {cand['image_glob']}")
    return "\n".join(lines)
