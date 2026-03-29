import os
import glob
import numpy as np


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
    exts = ("*.png", "*.jpg", "*.jpeg", "*.ppm")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(img_dir, e))
    files = sorted(files)
    if len(files) == 0:
        raise RuntimeError(f"No images found in {img_dir}")
    return files
