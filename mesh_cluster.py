import os
import numpy as np
import pyvista as pv
from sklearn.cluster import KMeans


def ensure_triangular(mesh: pv.PolyData) -> pv.PolyData:
    """确保 mesh 是三角面片."""
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()
    return mesh


def get_face_centers_and_ids(mesh: pv.PolyData):
    """
    返回：
    - centers: (n_faces, 3) 每个三角面的质心
    - face_ids: 原始 face 的 cell id
    """
    mesh = ensure_triangular(mesh)

    # faces 格式: [3, v0, v1, v2, 3, v0, v1, v2, ...]
    faces = mesh.faces.reshape(-1, 4)[:, 1:]   # (n_faces, 3)
    vertices = mesh.points                      # (n_vertices, 3)

    # 每个三角面的质心
    centers = vertices[faces].mean(axis=1)

    face_ids = np.arange(len(faces))
    return centers, face_ids


def extract_submesh_by_face_ids(mesh: pv.PolyData, face_ids: np.ndarray) -> pv.PolyData:
    """根据 face/cell id 提取子 mesh，并清理无用点."""
    if len(face_ids) == 0:
        return pv.PolyData()

    submesh = mesh.extract_cells(face_ids)
    # extract_cells 可能返回 UnstructuredGrid，转成 PolyData
    submesh = submesh.extract_surface().clean()
    submesh = ensure_triangular(submesh)
    return submesh


def split_mesh_kmeans(mesh: pv.PolyData, random_state=0):
    """
    对当前 mesh 按 face centroid 做 2-cluster KMeans 切分.
    返回 submesh0, submesh1
    """
    mesh = ensure_triangular(mesh)
    centers, face_ids = get_face_centers_and_ids(mesh)

    n_faces = len(face_ids)
    if n_faces < 2:
        return None, None

    # KMeans 二分类
    kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(centers)

    ids0 = face_ids[labels == 0]
    ids1 = face_ids[labels == 1]

    # 极端情况下某一类为空
    if len(ids0) == 0 or len(ids1) == 0:
        return None, None

    submesh0 = extract_submesh_by_face_ids(mesh, ids0)
    submesh1 = extract_submesh_by_face_ids(mesh, ids1)

    if submesh0.n_cells == 0 or submesh1.n_cells == 0:
        return None, None

    return submesh0, submesh1


def recursive_kmeans_save(
    mesh: pv.PolyData,
    out_dir: str,
    max_depth: int = 5,
    current_prefix: str = "",
    random_state: int = 0
):
    """
    递归二分保存:
    - depth=1: 0,1
    - depth=2: 00,01,10,11
    ...
    max_depth=5 表示最终保存到长度为 5 的编码
    """
    mesh = ensure_triangular(mesh)

    # 当前层还要继续分
    if len(current_prefix) >= max_depth:
        return

    submesh0, submesh1 = split_mesh_kmeans(mesh, random_state=random_state)

    # 如果无法继续分，就停止
    if submesh0 is None or submesh1 is None:
        print(f"[停止] prefix='{current_prefix}' 无法继续二分，当前 cells={mesh.n_cells}")
        return

    name0 = current_prefix + "0"
    name1 = current_prefix + "1"

    path0 = os.path.join(out_dir, f"{name0}.stl")
    path1 = os.path.join(out_dir, f"{name1}.stl")

    submesh0.save(path0)
    submesh1.save(path1)

    print(f"[保存] {path0} | points={submesh0.n_points}, faces={submesh0.n_cells}")
    print(f"[保存] {path1} | points={submesh1.n_points}, faces={submesh1.n_cells}")

    # 继续递归
    recursive_kmeans_save(
        submesh0, out_dir, max_depth=max_depth,
        current_prefix=name0, random_state=random_state
    )
    recursive_kmeans_save(
        submesh1, out_dir, max_depth=max_depth,
        current_prefix=name1, random_state=random_state
    )


if __name__ == "__main__":
    file = "/Users/haoguangwang/TUM/25ws/Zebrapose/welsh-dragon-small-centered.stl"
    out_dir = "mesh_clusters_depth5"

    os.makedirs(out_dir, exist_ok=True)

    mesh = pv.read(file)
    mesh = ensure_triangular(mesh)

    print("原始 mesh:")
    print(f"  points = {mesh.n_points}")
    print(f"  faces  = {mesh.n_cells}")

    recursive_kmeans_save(
        mesh=mesh,
        out_dir=out_dir,
        max_depth=5,
        current_prefix="",
        random_state=42
    )

    print(f"\n完成，输出目录: {out_dir}")