import os
import glob
import numpy as np
from PIL import Image
from gvxrPython3 import gvxr as gvxr

# =========================
# 参数设置
# =========================

unit = "mm"

# detector parameters

detector_center = [0., 80., 0.]
source_location = [0., -80., 0.]

number_of_pixels_horizontal = 768
number_of_pixels_vertical = 768

pixel_size_horizontal = 0.1
pixel_size_vertical = 0.1
detector_right_vector = [1., 0., 0.]
detector_up_vector = [0., 0., 1.]

beam_type = "pointSource"

# spectrum parameters
spectrum_type = "monochromatic"
spectrum_energy = 80.0
unit_of_energy = "keV"
number_of_photons = 1e6

# sample parameters
label_name = "mesh_object"
hu = 400
scaling_factor = 0.45

# base pose
translation = [0.0, 0.0, 0.0]
rotation = [0.0, 0.0, 0.0]  # base rotation x,y,z


# =========================
# gvxr context
# =========================

def create_gvxr_context():
    print("SimpleGVXR version:", gvxr.getVersionOfSimpleGVXR())
    print("CoreGVXR version:", gvxr.getVersionOfCoreGVXR())

    gvxr.createWindow()
    gvxr.setWindowSize(number_of_pixels_horizontal, number_of_pixels_vertical)

    if beam_type == "pointSource":
        gvxr.usePointSource()
    else:
        gvxr.useParallelBeam()

    if spectrum_type == "monochromatic":
        gvxr.setMonoChromatic(
            anEnergy=spectrum_energy,
            aUnitOfEnergy=unit_of_energy,
            aNumberOfPhotons=number_of_photons
        )
    else:
        raise ValueError(f"Unsupported spectrum_type: {spectrum_type}")

    gvxr.setSourcePosition(*source_location, unit)

    gvxr.setDetectorPosition(*detector_center, unit)
    gvxr.setDetectorUpVector(*detector_up_vector)
    gvxr.setDetectorRightVector(*detector_right_vector)
    gvxr.setDetectorNumberOfPixels(
        number_of_pixels_horizontal,
        number_of_pixels_vertical
    )
    gvxr.setDetectorPixelSize(
        pixel_size_horizontal,
        pixel_size_vertical,
        unit
    )


# =========================
# mesh loading
# =========================

def load_scene_graph(mesh_file):
    sx = sy = sz = scaling_factor
    tx, ty, tz = translation
    rx, ry, rz = rotation

    gvxr.loadMeshFile(label_name, mesh_file, unit)

    try:
        gvxr.moveToCentre(label_name)
    except Exception as e:
        print(f"[Warning] moveToCentre failed: {e}")

    gvxr.setHU(label_name, hu)

    # 顺序尽量保持和你 demo 接近
    gvxr.translateNode(label_name, tx, ty, tz, unit)

    # base rotation
    gvxr.rotateNode(label_name, 0, 0, rx)
    gvxr.rotateNode(label_name, 0, ry, 0)
    gvxr.rotateNode(label_name, rz, 0, 0)

    gvxr.scaleNode(label_name, sx, sy, sz)


# =========================
# render
# =========================

def render_xray_projection(invert=False):
    x_ray_image = gvxr.computeXRayImage()
    x_ray_image = np.array(x_ray_image, dtype=np.float32)

    x_min = np.min(x_ray_image)
    x_max = np.max(x_ray_image)

    if x_max - x_min > 1e-12:
        x_ray_image = (x_ray_image - x_min) / (x_max - x_min)
    else:
        x_ray_image = np.zeros_like(x_ray_image, dtype=np.float32)

    # 图像坐标修正
    x_ray_image = np.flip(x_ray_image, axis=0)

    if invert:
        x_ray_image = 1.0 - x_ray_image

    image_uint8 = np.clip(x_ray_image * 255.0, 0, 255).astype(np.uint8)
    image_pil = Image.fromarray(image_uint8, mode="L")

    return image_pil, x_ray_image


# =========================
# main render yz 10x10
# =========================

def render_mesh_yz_100(mesh_file, output_root):
    mesh_name = os.path.splitext(os.path.basename(mesh_file))[0]
    mesh_out_dir = os.path.join(output_root, mesh_name)
    os.makedirs(mesh_out_dir, exist_ok=True)

    # y,z 各 10 个角度
    y_angles = np.linspace(0.0, 360.0, 10, endpoint=False)
    z_angles = np.linspace(0.0, 360.0, 10, endpoint=False)

    projections = []
    pose_infos = []
    pose_id = 0

    print(f"\n=== Processing {mesh_name} ===")

    for y_angle in y_angles:
        for z_angle in z_angles:
            print(
                f"[{mesh_name}] pose {pose_id:03d}/099 "
                f"y={y_angle:.2f}, z={z_angle:.2f}"
            )

            # 清空变换
            gvxr.resetSceneTransformation()

            # 某些版本可选清场
            try:
                gvxr.removePolygonMeshesFromXRayRenderer()
            except Exception:
                pass

            try:
                gvxr.removePolygonMeshesFromSceneGraph()
            except Exception:
                pass

            load_scene_graph(mesh_file)

            # 只绕 Y/Z 旋转
            gvxr.rotateNode(label_name, 0, y_angle, 0)
            gvxr.rotateNode(label_name, 0, 0, z_angle)

            image_pil, image_float = render_xray_projection(invert=False)

            file_name = (
                f"pose_{pose_id:03d}_"
                f"y{y_angle:06.2f}_z{z_angle:06.2f}.png"
            )
            file_path = os.path.join(mesh_out_dir, file_name)
            image_pil.save(file_path)

            projections.append(image_float)
            pose_infos.append([pose_id, y_angle, z_angle])

            pose_id += 1

    projections = np.stack(projections, axis=0)
    np.save(os.path.join(mesh_out_dir, "projections.npy"), projections)

    pose_infos = np.array(pose_infos, dtype=np.float32)
    np.savetxt(
        os.path.join(mesh_out_dir, "poses_yz.txt"),
        pose_infos,
        fmt=["%d", "%.8f", "%.8f"],
        header="pose_id y_angle_deg z_angle_deg",
        comments=""
    )

    print(f"[Done] {mesh_name}: saved {pose_id} images to {mesh_out_dir}")


# =========================
# main
# =========================

def main():
    input_dir = "/home/haoguang/Zebrapose4Xray/mesh_clusters_depth5"

    output_dir = "/home/haoguang/Zebrapose4Xray/mesh_clusters_depth5_xray"

    os.makedirs(output_dir, exist_ok=True)

    stl_files = sorted(glob.glob(os.path.join(input_dir, "*.stl")))
    if len(stl_files) == 0:
        raise FileNotFoundError(f"No STL files found in: {input_dir}")

    print(f"Found {len(stl_files)} STL files")

    create_gvxr_context()

    for mesh_file in stl_files:
        try:
            render_mesh_yz_100(mesh_file, output_dir)
        except Exception as e:
            print(f"[Error] Failed on {mesh_file}: {e}")

    print("\nAll done.")


if __name__ == "__main__":
    main()