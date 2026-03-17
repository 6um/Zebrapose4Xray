import os
import math
import argparse
import numpy as np
from PIL import Image
from gvxrPython3 import gvxr as gvxr


# -----------------------------
# Default parameters (demo-style)
# -----------------------------
UNIT = "mm"

# Detector parameters
DETECTOR_CENTER = [0.0, 50.0, 0.0]
DETECTOR_RIGHT_VECTOR = [1.0, 0.0, 0.0]
DETECTOR_UP_VECTOR = [0.0, 0.0, 1.0]
NUMBER_OF_PIXELS_HORIZONTAL = 512
NUMBER_OF_PIXELS_VERTICAL = 512
PIXEL_SIZE_HORIZONTAL = 0.1
PIXEL_SIZE_VERTICAL = 0.1

# Source parameters
SOURCE_LOCATION = [0.0, -50.0, 0.0]
BEAM_TYPE = "pointSource"
SPECTRUM_TYPE = "monochromatic"
SPECTRUM_ENERGY = 80.0
UNIT_OF_ENERGY = "keV"
NUMBER_OF_PHOTONS = 1e6

# Sample parameters
DEFAULT_SCALING_FACTOR = 0.45
DEFAULT_HU = 400
DEFAULT_TRANSLATION = [0.0, 0.0, 0.0]
DEFAULT_ROTATION = [0.0, 0.0, 0.0]


def create_gvxr_context():
    print("simpleGVXR:", gvxr.getVersionOfSimpleGVXR())
    print("coreGVXR:", gvxr.getVersionOfCoreGVXR())

    gvxr.createWindow()
    gvxr.setWindowSize(NUMBER_OF_PIXELS_HORIZONTAL, NUMBER_OF_PIXELS_VERTICAL)

    if BEAM_TYPE == "pointSource":
        gvxr.usePointSource()
    elif BEAM_TYPE == "parallelBeam":
        gvxr.useParallelBeam()
    else:
        raise ValueError(f"Unknown beam type: {BEAM_TYPE}")

    if SPECTRUM_TYPE == "monochromatic":
        gvxr.setMonoChromatic(
            anEnergy=SPECTRUM_ENERGY,
            aUnitOfEnergy=UNIT_OF_ENERGY,
            aNumberOfPhotons=NUMBER_OF_PHOTONS,
        )
    else:
        raise ValueError(f"Unknown spectrum type: {SPECTRUM_TYPE}")

    gvxr.setSourcePosition(*SOURCE_LOCATION, UNIT)
    gvxr.setDetectorPosition(*DETECTOR_CENTER, UNIT)
    gvxr.setDetectorUpVector(*DETECTOR_UP_VECTOR)
    gvxr.setDetectorRightVector(*DETECTOR_RIGHT_VECTOR)
    gvxr.setDetectorNumberOfPixels(
        NUMBER_OF_PIXELS_HORIZONTAL,
        NUMBER_OF_PIXELS_VERTICAL,
    )
    gvxr.setDetectorPixelSize(
        PIXEL_SIZE_HORIZONTAL,
        PIXEL_SIZE_VERTICAL,
        UNIT,
    )


def load_mesh_scene(label_name, mesh_file, scaling_factor, hu, translation, rotation):
    sx = sy = sz = scaling_factor
    tx, ty, tz = translation
    rx, ry, rz = rotation

    gvxr.loadMeshFile(label_name, mesh_file, UNIT)
    gvxr.setHU(label_name, hu)

    # Keep the same operation order as your demo.
    gvxr.translateNode(label_name, tx, ty, tz, UNIT)
    gvxr.rotateNode(label_name, 0, 0, rx)
    gvxr.rotateNode(label_name, 0, ry, 0)
    gvxr.rotateNode(label_name, rz, 0, 0)
    gvxr.scaleNode(label_name, sx, sy, sz)


def render_xray_projection(invert=True):
    x_ray_image = np.array(gvxr.computeXRayImage(), dtype=np.float32)

    # Robust normalization to avoid checkerboard-like artifacts from bad scaling.
    x_ray_image = np.flip(x_ray_image, axis=0)

    max_val = float(np.max(x_ray_image))
    min_val = float(np.min(x_ray_image))

    if not np.isfinite(max_val) or not np.isfinite(min_val):
        raise RuntimeError("X-ray image contains NaN/Inf values.")

    if max_val - min_val < 1e-12:
        norm = np.zeros_like(x_ray_image, dtype=np.float32)
    else:
        norm = (x_ray_image - min_val) / (max_val - min_val)

    # For human-readable X-ray PNGs, darker attenuation is often easier to inspect.
    if invert:
        norm = 1.0 - norm

    img = (norm * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img, mode="L"), norm


def rotate_mesh_for_angle(label_name, angle_deg, axis="z"):
    if axis.lower() == "x":
        gvxr.rotateNode(label_name, angle_deg, 0, 0)
    elif axis.lower() == "y":
        gvxr.rotateNode(label_name, 0, angle_deg, 0)
    elif axis.lower() == "z":
        gvxr.rotateNode(label_name, 0, 0, angle_deg)
    else:
        raise ValueError("axis must be one of: x, y, z")


def generate_projections_for_mesh(
    mesh_file,
    output_dir,
    num_angles=100,
    scaling_factor=DEFAULT_SCALING_FACTOR,
    hu=DEFAULT_HU,
    translation=None,
    rotation=None,
    axis="z",
):
    if translation is None:
        translation = list(DEFAULT_TRANSLATION)
    if rotation is None:
        rotation = list(DEFAULT_ROTATION)

    mesh_name = os.path.splitext(os.path.basename(mesh_file))[0]
    mesh_out_dir = os.path.join(output_dir, mesh_name)
    os.makedirs(mesh_out_dir, exist_ok=True)

    print(f"\n=== Processing mesh: {mesh_name} ===")
    print(f"Input: {mesh_file}")
    print(f"Output folder: {mesh_out_dir}")

    # Fresh context for each mesh, to avoid scene accumulation.
    create_gvxr_context()

    label_name = mesh_name
    load_mesh_scene(
        label_name=label_name,
        mesh_file=mesh_file,
        scaling_factor=scaling_factor,
        hu=hu,
        translation=translation,
        rotation=rotation,
    )

    angles = np.linspace(0.0, 360.0, num_angles, endpoint=False)
    projections = []
    step = 360.0 / num_angles

    for i, angle_deg in enumerate(angles):
        if i > 0:
            rotate_mesh_for_angle(label_name, step, axis=axis)

        image, arr = render_xray_projection(invert=True)
        image_path = os.path.join(mesh_out_dir, f"angle_{i:03d}_{angle_deg:06.2f}deg.png")
        image.save(image_path)
        projections.append(arr)
        print(f"[{i + 1:03d}/{num_angles}] saved: {image_path}")

    np.save(os.path.join(mesh_out_dir, "projections.npy"), np.stack(projections, axis=0))
    np.savetxt(os.path.join(mesh_out_dir, "angles.txt"), angles, fmt="%.6f")

    # Close the OpenGL window for this mesh.
    try:
        gvxr.destroyWindow()
    except Exception:
        pass


def find_mesh_files(input_dir):
    files = []
    for name in sorted(os.listdir(input_dir)):
        if name.lower().endswith(".stl"):
            files.append(os.path.join(input_dir, name))
    return files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate demo-style gVXR X-ray projections for every STL in a folder."
    )
    parser.add_argument("--input_dir", required=True, help="Folder containing STL files")
    parser.add_argument("--output_dir", required=True, help="Folder to save output images")
    parser.add_argument("--num_angles", type=int, default=100, help="Number of angles per mesh")
    parser.add_argument("--scaling_factor", type=float, default=DEFAULT_SCALING_FACTOR)
    parser.add_argument("--hu", type=float, default=DEFAULT_HU)
    parser.add_argument("--tx", type=float, default=DEFAULT_TRANSLATION[0])
    parser.add_argument("--ty", type=float, default=DEFAULT_TRANSLATION[1])
    parser.add_argument("--tz", type=float, default=DEFAULT_TRANSLATION[2])
    parser.add_argument("--rx", type=float, default=DEFAULT_ROTATION[0])
    parser.add_argument("--ry", type=float, default=DEFAULT_ROTATION[1])
    parser.add_argument("--rz", type=float, default=DEFAULT_ROTATION[2])
    parser.add_argument("--axis", choices=["x", "y", "z"], default="z", help="Rotation axis")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    mesh_files = find_mesh_files(args.input_dir)

    if not mesh_files:
        raise FileNotFoundError(f"No STL files found in: {args.input_dir}")

    translation = [args.tx, args.ty, args.tz]
    rotation = [args.rx, args.ry, args.rz]

    print(f"Found {len(mesh_files)} STL files.")
    for mesh_file in mesh_files:
        generate_projections_for_mesh(
            mesh_file=mesh_file,
            output_dir=args.output_dir,
            num_angles=args.num_angles,
            scaling_factor=args.scaling_factor,
            hu=args.hu,
            translation=translation,
            rotation=rotation,
            axis=args.axis,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

    '''



python generate_gvxr_xrays_demo_style.py \
  --input_dir /home/haoguang/Zebrapose4Xray/mesh_clusters_depth5 \
  --output_dir /home/haoguang/Zebrapose4Xray/mesh_clusters_depth5_xray \
  --num_angles 100
    '''