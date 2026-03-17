import os
import glob
import numpy as np
from PIL import Image
from gvxrPython3 import gvxr as gvxr

# =========================
# 参数
# =========================

unit = "mm"

detector_center = [0., 80., 0.]
source_location = [0., -80., 0.]

number_of_pixels_horizontal = 768
number_of_pixels_vertical = 768

pixel_size_horizontal = 0.1
pixel_size_vertical = 0.1
detector_right_vector = [1., 0., 0.]
detector_up_vector = [0., 0., 1.]


spectrum_energy = 80.0
unit_of_energy = "keV"
number_of_photons = 1e6

label_name = "mesh_object"

scaling_factor = 0.45

import trimesh

def compute_auto_scale(mesh_file, target_size_mm=18.0):
    """
    根据 mesh 包围盒自动计算缩放，让最大边长缩到 target_size_mm 左右。
    """
    mesh = trimesh.load(mesh_file, force='mesh')
    bounds = mesh.bounds  # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    extents = bounds[1] - bounds[0]
    max_extent = float(np.max(extents))

    if max_extent < 1e-9:
        return 1.0

    scale = target_size_mm / max_extent
    return scale


hu = 400


# =========================
# gvxr context
# =========================

def create_gvxr_context():

    print(gvxr.getVersionOfSimpleGVXR())
    print(gvxr.getVersionOfCoreGVXR())

    gvxr.createWindow()
    gvxr.setWindowSize(number_of_pixels_horizontal,
                       number_of_pixels_vertical)

    gvxr.usePointSource()

    gvxr.setMonoChromatic(
        anEnergy=spectrum_energy,
        aUnitOfEnergy=unit_of_energy,
        aNumberOfPhotons=number_of_photons
    )

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
# load mesh
# =========================

def load_mesh(mesh_file, scale):
    gvxr.loadMeshFile(label_name, mesh_file, unit)

    try:
        gvxr.moveToCentre(label_name)
    except Exception:
        pass

    gvxr.setHU(label_name, hu)
    gvxr.scaleNode(label_name, scale, scale, scale)


# =========================
# render
# =========================

def render_xray():

    xray = np.array(gvxr.computeXRayImage(), dtype=np.float32)

    xray -= np.min(xray)
    xray /= np.max(xray) + 1e-9

    xray = np.flip(xray, axis=0)

    img = (xray*255).astype(np.uint8)

    return Image.fromarray(img, mode="L"), xray


# =========================
# render poses
# =========================

def render_poses(mesh_file, output_dir):

    mesh_name = os.path.splitext(os.path.basename(mesh_file))[0]

    mesh_out = os.path.join(output_dir, mesh_name)

    os.makedirs(mesh_out, exist_ok=True)

    angles = np.linspace(0, 360, 5, endpoint=False)

    projections = []

    pose_id = 0

    for ax in angles:
        for ay in angles:
            for az in angles:

                print(mesh_name, "pose", pose_id,
                      "angles", ax, ay, az)

                gvxr.resetSceneTransformation()

                try:
                    gvxr.removePolygonMeshesFromSceneGraph()
                except:
                    pass

                scale = compute_auto_scale(mesh_file, target_size_mm=18.0)
                load_mesh(mesh_file, scale)

                gvxr.rotateNode(label_name, ax, 0, 0)
                gvxr.rotateNode(label_name, 0, ay, 0)
                gvxr.rotateNode(label_name, 0, 0, az)

                img, arr = render_xray()

                filename = f"pose_{pose_id:03d}_x{ax:.0f}_y{ay:.0f}_z{az:.0f}.png"

                img.save(os.path.join(mesh_out, filename))

                projections.append(arr)

                pose_id += 1

    projections = np.stack(projections)

    np.save(os.path.join(mesh_out, "projections.npy"), projections)

    print("saved", pose_id, "images")






# =========================
# main
# =========================

def main():

    input_dir = "/home/haoguang/Zebrapose4Xray/mesh_clusters_depth5"

    output_dir = "/home/haoguang/Zebrapose4Xray/mesh_clusters_depth5_xray"

    os.makedirs(output_dir, exist_ok=True)

    stl_files = sorted(glob.glob(os.path.join(input_dir, "*.stl")))

    print("found", len(stl_files), "meshes")

    create_gvxr_context()

    for mesh in stl_files:

        render_poses(mesh, output_dir)

    print("done")


if __name__ == "__main__":
    main()

