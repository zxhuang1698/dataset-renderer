# BlenderProc Object Renderer

This repository contains a script to render 3D objects using BlenderProc with HDRI backgrounds. The script normalizes the objects, sets up the camera, and renders images from different viewpoints. Additionally, it generates point clouds from the rendered objects.

## Requirements

- Python 3.6+
- BlenderProc
- NumPy
- OpenCV
- Pillow
- Trimesh
- argparse

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/zxhuang1698/dataset-renderer
    cd dataset-renderer
    ```

2. Install the required Python packages:
    ```sh
    pip install blenderproc numpy opencv-python pillow trimesh
    ```

## Usage

To render a 3D object, run the following command:

```sh
python worker.py --input path/to/your/model.obj --output path/to/output/directory

