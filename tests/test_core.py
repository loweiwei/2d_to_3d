import numpy as np

from tdt3d import ProjectionInput, TwoDToThreeDTool


def test_calculate_aabb_basic():
    tool = TwoDToThreeDTool()
    points = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 4.0, 6.0],
    ])
    bbox = tool.calculate_aabb(points)
    assert np.allclose(bbox, np.array([1.0, 2.0, 3.0, 2.0, 4.0, 6.0]))


def test_project_mask_to_3d_nonempty():
    tool = TwoDToThreeDTool(depth_scale=1.0)
    depth = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    mask = np.array([[1, 0], [0, 0]], dtype=bool)
    K = np.eye(4)
    T = np.eye(4)
    points = tool.project_mask_to_3d(
        ProjectionInput(
            depth_image=depth,
            intrinsic_matrix=K,
            extrinsic_matrix=T,
            mask=mask,
        )
    )
    assert points.shape == (1, 3)
