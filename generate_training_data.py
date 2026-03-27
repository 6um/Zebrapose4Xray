import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image
from gvxrPython3 import gvxr as gvxr
from typing import Tuple, Dict


# ============================================================
# 1. Basic Configuration
# ============================================================

def build_config(base_folder: str) -> Dict:
    unit = "mm"

    # -------------------------
    # Detector
    # -------------------------
    detector_center = [0.0, 50.0, 0.0]
    detector_right_vector = [1.0, 0.0, 0.0]
    detector_up_vector = [0.0, 0.0, 1.0]
    number_of_pixels_horizontal = 512
    number_of_pixels_vertical = 512
    pixel_size_horizontal = 0.1
    pixel_size_vertical = 0.1

    # -------------------------
    # X-ray source
    # -------------------------
    source_location = [0.0, -50.0, 0.0]
    beam_type = "pointSource"
    spectrum_type = "monochromatic"
    spectrum_energy = 80.0
    unit_of_energy = "keV"
    number_of_photons = 1e6

    # -------------------------
    # Object / material
    # -------------------------
    label_name = "welsh-dragon"
    hu = 400

    # -------------------------
    # Camera geometry for PyVista
    # -------------------------
    source_view_direction = np.array(detector_center) - np.array(source_location)
    source_view_direction = source_view_direction / np.linalg.norm(source_view_direction)

    detector_height = pixel_size_vertical * number_of_pixels_vertical
    source_distance = np.linalg.norm(np.array(detector_center) - np.array(source_location))

    detector_view_angle_vertical = 2.0 * np.degrees(
        np.arctan(detector_height / 2.0 / source_distance)
    )

    return {
        "unit": unit,
        "detector_center": detector_center,
        "detector_right_vector": detector_right_vector,
        "detector_up_vector": detector_up_vector,
        "number_of_pixels_horizontal": number_of_pixels_horizontal,
        "number_of_pixels_vertical": number_of_pixels_vertical,
        "pixel_size_horizontal": pixel_size_horizontal,
        "pixel_size_vertical": pixel_size_vertical,
        "source_location": source_location,
        "beam_type": beam_type,
        "spectrum_type": spectrum_type,
        "spectrum_energy": spectrum_energy,
        "unit_of_energy": unit_of_energy,
        "number_of_photons": number_of_photons,
        "label_name": label_name,
        "hu": hu,
        "source_view_direction": source_view_direction.tolist(),
        "detector_view_angle_vertical": float(detector_view_angle_vertical),
        "file_name_mesh": os.path.join(base_folder, "welsh-dragon-small-centered.stl"),
        "binary_level_template": os.path.join(base_folder, "binary_level_{level}.vtk"),
    }


# ============================================================
# 2. Random Pose
# ============================================================

def create_random_pose(run_id: int) -> Dict:
    rng = np.random.default_rng(run_id)

    translation = [
        float(rng.uniform(-2.0, 2.0)),
        float(rng.uniform(-10.0, 10.0)),
        float(rng.uniform(-10.0, 10.0)),
    ]
    rotation = [
        float(rng.uniform(-180.0, 180.0)),
        float(rng.uniform(-180.0, 180.0)),
        float(rng.uniform(-180.0, 180.0)),
    ]
    scaling_factor = 0.4
    scaling = [scaling_factor, scaling_factor, scaling_factor]

    return {
        "translation": translation,
        "rotation": rotation,
        "scaling": scaling,
    }


# ============================================================
# 3. GVXR: X-ray Imaging
# ============================================================

def create_gvxr_context(cfg: Dict) -> None:
    gvxr.createWindow()
    gvxr.setWindowSize(
        cfg["number_of_pixels_horizontal"],
        cfg["number_of_pixels_vertical"],
    )

    if cfg["beam_type"] == "pointSource":
        gvxr.usePointSource()
    elif cfg["beam_type"] == "parallelBeam":
        gvxr.useParallelBeam()
    else:
        raise ValueError(f"Unknown beam type: {cfg['beam_type']}")

    if cfg["spectrum_type"] == "monochromatic":
        gvxr.setMonoChromatic(
            anEnergy=cfg["spectrum_energy"],
            aUnitOfEnergy=cfg["unit_of_energy"],
            aNumberOfPhotons=cfg["number_of_photons"],
        )
    else:
        raise ValueError(f"Unknown spectrum type: {cfg['spectrum_type']}")

    gvxr.setSourcePosition(*cfg["source_location"], cfg["unit"])
    gvxr.setDetectorPosition(*cfg["detector_center"], cfg["unit"])
    gvxr.setDetectorUpVector(*cfg["detector_up_vector"])
    gvxr.setDetectorRightVector(*cfg["detector_right_vector"])
    gvxr.setDetectorNumberOfPixels(
        cfg["number_of_pixels_horizontal"],
        cfg["number_of_pixels_vertical"],
    )
    gvxr.setDetectorPixelSize(
        cfg["pixel_size_horizontal"],
        cfg["pixel_size_vertical"],
        cfg["unit"],
    )


def load_scene_graph(cfg: Dict, random_pose: Dict) -> None:
    sx, sy, sz = random_pose["scaling"]
    rx, ry, rz = random_pose["rotation"]
    tx, ty, tz = random_pose["translation"]

    gvxr.loadMeshFile(cfg["label_name"], cfg["file_name_mesh"], cfg["unit"])
    gvxr.setHU(cfg["label_name"], cfg["hu"])

    gvxr.translateNode(cfg["label_name"], tx, ty, tz, cfg["unit"])
    gvxr.rotateNode(cfg["label_name"], 0, 0, rx)
    gvxr.rotateNode(cfg["label_name"], 0, ry, 0)
    gvxr.rotateNode(cfg["label_name"], rz, 0, 0)
    gvxr.scaleNode(cfg["label_name"], sx, sy, sz)


def render_xray_projection() -> Image.Image:
    x_ray_image = np.array(gvxr.computeXRayImage(), dtype=np.float32)

    max_val = np.max(x_ray_image)
    if max_val > 0:
        x_ray_image /= max_val

    x_ray_image = np.flip(x_ray_image, axis=0)

    x_ray_image -= np.min(x_ray_image)
    max_val = np.max(x_ray_image)
    if max_val > 0:
        x_ray_image /= max_val

    image_array = (x_ray_image * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(image_array, mode="L")


# ============================================================
# 4. PyVista: Perspective Projection Aligned with X-ray
# ============================================================

def apply_pose_to_mesh(mesh: pv.PolyData, random_pose: Dict) -> pv.PolyData:
    mesh = mesh.copy(deep=True)

    sx, sy, sz = random_pose["scaling"]
    mesh.scale([sx, sy, sz], inplace=True)

    rx, ry, rz = random_pose["rotation"]
    mesh.rotate_z(rz, inplace=True)
    mesh.rotate_y(ry, inplace=True)
    mesh.rotate_x(rx, inplace=True)

    tx, ty, tz = random_pose["translation"]
    mesh.translate([tx, ty, tz], inplace=True)

    return mesh


def get_plotter(cfg: Dict, off_screen: bool = True) -> pv.Plotter:
    plotter = pv.Plotter(
        window_size=(
            cfg["number_of_pixels_horizontal"],
            cfg["number_of_pixels_vertical"],
        ),
        off_screen=off_screen,
    )

    camera_location = cfg["source_location"]
    source_view_direction = np.array(cfg["source_view_direction"], dtype=np.float64)
    focal_point = np.array(camera_location) + source_view_direction
    view_up_vector = cfg["detector_up_vector"]

    plotter.camera_position = [camera_location, focal_point.tolist(), view_up_vector]
    plotter.camera.view_angle = cfg["detector_view_angle_vertical"]
    plotter.set_background("black")

    return plotter


def _to_gray(image_array: np.ndarray) -> np.ndarray:
    if image_array.ndim == 3:
        image_array = image_array[..., :3].mean(axis=2)
    return image_array.astype(np.float32)


def render_silhouette_mask(cfg: dict, mesh: pv.PolyData, threshold: float = 10.0) -> np.ndarray:
    plotter = get_plotter(cfg, off_screen=True)
    plotter.add_mesh(
        mesh,
        color="white",
        show_edges=False,
        lighting=False,
    )

    image_array = plotter.screenshot(return_img=True)
    plotter.close()

    gray = _to_gray(image_array)
    mask = (gray > threshold).astype(np.uint8)
    return mask


def render_bit_image(
    cfg: dict,
    mesh: pv.PolyData,
    bit_values: np.ndarray,
    threshold: float = 10.0,
) -> np.ndarray:
    render_mesh = mesh.copy(deep=True)

    n_points = render_mesh.n_points
    n_cells = render_mesh.n_cells

    if len(bit_values) == n_points:
        render_mesh.point_data["bit"] = bit_values.astype(np.float32)
        scalars_name = "bit"
        preference = "point"
    elif len(bit_values) == n_cells:
        render_mesh.cell_data["bit"] = bit_values.astype(np.float32)
        scalars_name = "bit"
        preference = "cell"
    else:
        raise ValueError(
            f"bit_values length {len(bit_values)} does not match "
            f"n_points={n_points} or n_cells={n_cells}"
        )

    plotter = get_plotter(cfg, off_screen=True)
    plotter.add_mesh(
        render_mesh,
        scalars=scalars_name,
        clim=[0.0, 1.0],
        cmap="gray",
        show_edges=False,
        show_scalar_bar=False,
        lighting=False,
        preference=preference,
    )

    image_array = plotter.screenshot(return_img=True)
    plotter.close()

    gray = _to_gray(image_array)
    bit_img = (gray > threshold).astype(np.uint8)
    return bit_img


# ============================================================
# 5. 10-bit Code Generation
# ============================================================

def create_code_stack(
    cfg: dict,
    random_pose: dict,
    num_levels: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    first_level_file = cfg["binary_level_template"].format(level=1)
    base_mesh = pv.read(first_level_file)
    base_mesh = apply_pose_to_mesh(base_mesh, random_pose)

    mask = render_silhouette_mask(cfg, base_mesh)

    h = cfg["number_of_pixels_vertical"]
    w = cfg["number_of_pixels_horizontal"]
    code_stack = np.zeros((num_levels, h, w), dtype=np.uint8)

    for level in range(1, num_levels + 1):
        vtk_file = cfg["binary_level_template"].format(level=level)
        mesh = pv.read(vtk_file)
        mesh = apply_pose_to_mesh(mesh, random_pose)

        if "clusters" not in mesh.array_names:
            raise KeyError(f"'clusters' not found in {vtk_file}")

        clusters = np.asarray(mesh["clusters"])
        bit_values = (clusters % 2).astype(np.uint8)

        bit_img = render_bit_image(cfg, mesh, bit_values)
        bit_img = bit_img * mask
        code_stack[level - 1] = bit_img

    return mask.astype(np.uint8), code_stack.astype(np.uint8)


def pack_code_stack_to_uint16(code_stack: np.ndarray) -> np.ndarray:
    if code_stack.ndim != 3 or code_stack.shape[0] != 10:
        raise ValueError("code_stack must have shape (10, H, W)")

    packed = np.zeros(code_stack.shape[1:], dtype=np.uint16)
    for i in range(10):
        packed |= (code_stack[i].astype(np.uint16) << i)
    return packed


def make_code_visualization(packed_code: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vis = (packed_code.astype(np.float32) / 1023.0 * 255.0).clip(0, 255).astype(np.uint8)
    vis[mask == 0] = 0
    return vis


# ============================================================
# 6. Save a Sample
# ============================================================

def save_sample(
    output_root: str,
    run_id: int,
    xray_image: Image.Image,
    mask: np.ndarray,
    code_stack: np.ndarray,
    pose: Dict,
) -> None:
    sample_dir = os.path.join(output_root, f"sample_{run_id:06d}")
    os.makedirs(sample_dir, exist_ok=True)

    xray_path = os.path.join(sample_dir, "xray.png")
    mask_png_path = os.path.join(sample_dir, "mask.png")
    mask_npy_path = os.path.join(sample_dir, "mask.npy")
    code_stack_path = os.path.join(sample_dir, "code_stack.npy")
    packed_code_path = os.path.join(sample_dir, "packed_code.npy")
    code_vis_path = os.path.join(sample_dir, "code_vis.png")
    meta_path = os.path.join(sample_dir, "sample_meta.json")

    xray_image.save(xray_path)

    mask_png = (mask * 255).astype(np.uint8)
    Image.fromarray(mask_png, mode="L").save(mask_png_path)
    np.save(mask_npy_path, mask.astype(np.uint8))

    np.save(code_stack_path, code_stack.astype(np.uint8))

    packed_code = pack_code_stack_to_uint16(code_stack)
    np.save(packed_code_path, packed_code)

    code_vis = make_code_visualization(packed_code, mask)
    Image.fromarray(code_vis, mode="L").save(code_vis_path)

    meta = {
        "run_id": run_id,
        "translation": pose["translation"],
        "rotation": pose["rotation"],
        "scaling": pose["scaling"],
        "files": {
            "xray": str(Path(xray_path).name),
            "mask_png": str(Path(mask_png_path).name),
            "mask_npy": str(Path(mask_npy_path).name),
            "code_stack_npy": str(Path(code_stack_path).name),
            "packed_code_npy": str(Path(packed_code_path).name),
            "code_vis_png": str(Path(code_vis_path).name),
        },
        "shapes": {
            "mask": list(mask.shape),
            "code_stack": list(code_stack.shape),
        },
        "semantic_note": {
            "mask": "0=background, 1=object",
            "code_stack": "shape=(10,H,W), each plane is one binary code bit",
            "packed_code": "uint16 packed from 10 bits, value in [0,1023]",
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# ============================================================
# 7. Generate a Single Sample
# ============================================================

def generate_one_sample(cfg: Dict, output_root: str, run_id: int) -> None:
    print(f"[INFO] Generating run_id={run_id}")

    pose = create_random_pose(run_id)
    print(f"[INFO] Pose = {pose}")

    create_gvxr_context(cfg)
    load_scene_graph(cfg, pose)
    xray_image = render_xray_projection()

    mask, code_stack = create_code_stack(cfg, pose, num_levels=10)

    save_sample(
        output_root=output_root,
        run_id=run_id,
        xray_image=xray_image,
        mask=mask,
        code_stack=code_stack,
        pose=pose,
    )

    print(f"[INFO] Saved sample_{run_id:06d}")


# ============================================================
# 8. Batch Generation (Using subprocess)
# ============================================================

def generate_dataset_with_subprocess(
    base_folder: str,
    output_root: str,
    start_id: int,
    num_samples: int,
) -> None:
    os.makedirs(output_root, exist_ok=True)

    script_path = os.path.abspath(__file__)

    for run_id in range(start_id, start_id + num_samples):
        cmd = [
            sys.executable,
            script_path,
            "--worker",
            "--base_folder", base_folder,
            "--output_root", output_root,
            "--run_id", str(run_id),
        ]

        print(f"[INFO] Launch subprocess for run_id={run_id}")
        print("[INFO] CMD =", " ".join(cmd))

        result = subprocess.run(cmd, check=False)

        if result.returncode != 0:
            print(f"[ERROR] run_id={run_id} failed with return code {result.returncode}")
        else:
            print(f"[INFO] run_id={run_id} done")


# ============================================================
# 9. Command Line Arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate training data: X-ray input + 10-bit code stack + mask."
    )
    parser.add_argument(
        "--base_folder",
        type=str,
        default=None,
        help="Folder containing STL and binary_level_*.vtk. Default: script folder.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--start_id",
        type=int,
        default=0,
        help="Starting run_id.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate.",
    )

    # worker mode
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Run as subprocess worker to generate exactly one sample.",
    )
    parser.add_argument(
        "--run_id",
        type=int,
        default=None,
        help="Used only in --worker mode.",
    )

    return parser.parse_args()


# ============================================================
# 10. Main Function
# ============================================================

def main() -> None:
    args = parse_args()

    if args.base_folder is None:
        base_folder = os.path.dirname(os.path.abspath(__file__))
    else:
        base_folder = os.path.abspath(args.base_folder)

    output_root = os.path.abspath(args.output_root)

    print(f"[INFO] base_folder = {base_folder}")
    print(f"[INFO] output_root = {output_root}")

    if args.worker:
        if args.run_id is None:
            raise ValueError("--worker mode requires --run_id")

        print(f"[INFO] worker run_id = {args.run_id}")
        cfg = build_config(base_folder)
        generate_one_sample(
            cfg=cfg,
            output_root=output_root,
            run_id=args.run_id,
        )
    else:
        print(f"[INFO] start_id = {args.start_id}")
        print(f"[INFO] num_samples = {args.num_samples}")

        generate_dataset_with_subprocess(
            base_folder=base_folder,
            output_root=output_root,
            start_id=args.start_id,
            num_samples=args.num_samples,
        )


if __name__ == "__main__":
    main()

