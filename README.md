# 2D to 3D Toolkit

A lightweight Python toolkit for turning **2D masks** into **3D point clouds** and **3D axis-aligned bounding boxes (AABB)**.

This package focuses on the geometry stage only:

```text
2D mask -> 3D points -> point filtering -> 3D bounding box
```

It is designed for projects that already have a 2D mask from a detector, segmenter, or VLM pipeline and need a clean, reusable 2D->3D module.

## Features

- Single-view 2D mask -> 3D point projection
- Multi-view point fusion by concatenation
- Optional mask post-processing
  - erosion
  - dilation
  - connected components filtering
- Optional point filtering
  - statistical outlier removal (Open3D)
  - truncated filtering
- 3D AABB output in the format:

```python
[cx, cy, cz, dx, dy, dz]
```

## Installation

### Option 1: local editable install

```bash
git clone <your-repo-url>
cd two_d_to_three_d_toolkit
pip install -e .
```

### Option 2: normal install

```bash
pip install .
```

## Requirements

Core dependencies:

- numpy
- opencv-python

Optional:

- open3d (only needed for `filter_type="statistical"`)

Install manually if needed:

```bash
pip install numpy opencv-python
pip install open3d
```

## Quick Start

### Single view

```python
import numpy as np
from tdt3d import TwoDToThreeDTool

mask = np.load("mask.npy")
depth_image = "depth.png"
K = np.load("intrinsic.npy")
pose = np.load("extrinsic.npy")
axis_align = np.load("axis_align.npy")

tool = TwoDToThreeDTool()
points, bbox = tool.run_single_view(
    mask=mask,
    depth_image=depth_image,
    intrinsic_matrix=K,
    extrinsic_matrix=pose,
    world_to_axis_align_matrix=axis_align,
)

print(points.shape)
print(bbox)
```

### Multi-view

```python
import numpy as np
from tdt3d import ProjectionInput, TwoDToThreeDTool

axis_align = np.load("axis_align.npy")
K = np.load("intrinsic.npy")

views = [
    ProjectionInput(
        depth_image="depth1.png",
        intrinsic_matrix=K,
        extrinsic_matrix=np.load("extrinsic1.npy"),
        world_to_axis_align_matrix=axis_align,
        mask=np.load("mask1.npy"),
    ),
    ProjectionInput(
        depth_image="depth2.png",
        intrinsic_matrix=K,
        extrinsic_matrix=np.load("extrinsic2.npy"),
        world_to_axis_align_matrix=axis_align,
        mask=np.load("mask2.npy"),
    ),
]

tool = TwoDToThreeDTool()
points, bbox = tool.run_multi_view(views)
```

## Input Expectations

### Mask

- 2D array with shape `(H, W)`
- can be boolean or 0/1

### Intrinsic / extrinsic matrices

This toolkit expects:

- `intrinsic_matrix.shape == (4, 4)`
- `extrinsic_matrix.shape == (4, 4)`

That matches pipelines that store camera matrices in homogeneous form.

### Depth units

By default, the toolkit assumes the depth image is stored in **millimeters** and converts it to meters with:

```python
value_in_meters = value_in_depth_image * 0.001
```

If your depth is already in meters, initialize the tool with:

```python
tool = TwoDToThreeDTool(depth_scale=1.0)
```

## API

### `TwoDToThreeDTool`

Main class.

#### `run_single_view(...)`

Runs the full single-view pipeline:

```text
mask -> points -> filtered points -> AABB
```

Returns:

- `points`: `Nx3` or `Nx6`
- `bbox`: `[cx, cy, cz, dx, dy, dz]`

#### `run_multi_view(views, do_post_process=True)`

Runs the multi-view pipeline:

```text
multiple masks -> fused points -> filtered points -> AABB
```

#### `post_process_mask(mask)`

Applies optional mask refinement.

#### `project_mask_to_3d(data)`

Projects one 2D mask into 3D.

#### `filter_points(points)`

Applies configured point filtering.

#### `calculate_aabb(points)`

Returns the axis-aligned 3D bounding box.

## Configuration

### MorphologyConfig

```python
from tdt3d import MorphologyConfig

morph = MorphologyConfig(
    erosion=True,
    dilation=False,
    kernel_size=3,
    keep_largest_components=True,
    num_components=1,
)
```

### PointFilterConfig

```python
from tdt3d import PointFilterConfig

filt = PointFilterConfig(
    filter_type="statistical",  # statistical | truncated | none
    nb_neighbors=20,
    std_ratio=1.0,
)
```

## Example with custom config

```python
from tdt3d import (
    MorphologyConfig,
    PointFilterConfig,
    TwoDToThreeDTool,
)

tool = TwoDToThreeDTool(
    morphology=MorphologyConfig(
        erosion=True,
        dilation=False,
        kernel_size=3,
        keep_largest_components=True,
        num_components=1,
    ),
    point_filter=PointFilterConfig(
        filter_type="truncated",
        tx=0.05,
        ty=0.05,
        tz=0.05,
    ),
    project_color=False,
)
```

## Project Structure

```text
two_d_to_three_d_toolkit/
├── tdt3d/
│   ├── __init__.py
│   └── core.py
├── examples/
│   ├── demo_single_view.py
│   └── demo_multi_view.py
├── tests/
│   └── test_core.py
├── README.md
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

## Notes

- This toolkit does not generate masks by itself.
- It assumes you already have a 2D mask from another model.
- The current bounding box output is **AABB**, not oriented bounding box.
- Multi-view fusion is currently implemented as **point concatenation**, which is a clean baseline for many research prototypes.

## Recommended workflow

1. Start with `run_single_view`
2. Verify the generated points and bbox
3. Move to `run_multi_view`
4. Plug it back into your larger grounding / agent pipeline

## License

MIT
