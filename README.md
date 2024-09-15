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

Download the hdri environment maps:
```sh
blenderproc download haven your_hdri_path
```

To render a single 3D object, run the following command:

```sh
python worker.py --input path/to/your/model.obj --output path/to/output/directory --hdri_path your_hdri_path/hdris
```
More rendering options are available in the workers. Please take a look for any specific requirements.

To render a dataset (e.g. gso) in parallel, make sure the texture and models lie in the same directory and run the following command:
```sh
python distributed.py --num_workers 6 --dataset_dir your_gso_path --output_dir your_output_path --dataset gso --config configs/gso.json
```
