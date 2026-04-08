import cv2
import numpy as np
import open3d as o3d
from tdt3d.core import TwoDToThreeDTool, PointFilterConfig, ProjectionInput

scene_id = "scene0207_00"
image_ids = ["00000", "00020", "00040"]   # 先挑 3 張試

intrinsic_path = f"{scene_id}/intrinsic.txt"
intrinsic_matrix = np.loadtxt(intrinsic_path).astype(np.float64)

if intrinsic_matrix.shape != (4, 4):
    raise ValueError(f"intrinsic_matrix shape error: {intrinsic_matrix.shape}")

world_to_axis_align_matrix = np.eye(4, dtype=np.float64)

views = []

for image_id in image_ids:
    rgb_path = f"{scene_id}/{image_id}.jpg"
    depth_path = f"{scene_id}/{image_id}.png"
    extrinsic_path = f"{scene_id}/{image_id}.txt"

    rgb = cv2.imread(rgb_path)
    if rgb is None:
        raise FileNotFoundError(rgb_path)

    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise FileNotFoundError(depth_path)

    extrinsic_matrix = np.loadtxt(extrinsic_path).astype(np.float64)
    if extrinsic_matrix.shape != (4, 4):
        raise ValueError(f"{image_id} extrinsic shape error: {extrinsic_matrix.shape}")

    print(f"\nSelecting ROI for {image_id} ...")
    x, y, w, h = cv2.selectROI(f"Select ROI - {image_id}", rgb, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    if w == 0 or h == 0:
        print(f"Skip {image_id} because no ROI selected")
        continue

    H, W = rgb.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[int(y):int(y+h), int(x):int(x+w)] = 1

    mask_depth = cv2.resize(
        mask.astype(np.uint8),
        (depth_image.shape[1], depth_image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    print(f"{image_id} masked nonzero depth:", np.count_nonzero(depth_image[mask_depth > 0]))

    views.append(
        ProjectionInput(
            depth_image=depth_image,
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
            world_to_axis_align_matrix=world_to_axis_align_matrix,
            mask=mask,
            color_image=rgb,
        )
    )

if not views:
    raise ValueError("No valid views selected")

tool = TwoDToThreeDTool(
    point_filter=PointFilterConfig(filter_type="none"),
    depth_scale=0.001,
    project_color=False,
)

points, bbox = tool.run_multi_view(views, do_post_process=True)

print("multi-view points shape:", points.shape)
print("multi-view bbox:", bbox)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])

o3d.visualization.draw_geometries([pcd], window_name="Multi-view 3D Point Cloud")