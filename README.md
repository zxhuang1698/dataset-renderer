# BlenderProc Object Renderer

This repository contains scripts to render 3D objects using BlenderProc with HDRI backgrounds. The scripts normalize the objects, set up the camera, and render images from different viewpoints. Additionally, they generate point clouds from the rendered objects.

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

Download the HDRI environment maps:
```sh
blenderproc download haven your_hdri_path
```

### Rendering a Single Object

To render a single 3D object, run the following command:

```sh
python worker.py --input path/to/your/model.obj --output path/to/output/directory --hdri_path your_hdri_path/hdris
```
More rendering options are available in the `worker.py` script. Please review the script for any specific requirements.

### Preparing Object Lists and Sharding

Before rendering a dataset, you need to prepare a list of objects and create shards. This is done using the `create_list.py` script:

1. Modify the `create_list.py` script to set the correct paths:
   - Set `data_root` to the path of your dataset (e.g., GSO dataset)
   - Set `raw_list_path` to the path of your object list file
   - Set `num_shards` to the desired number of shards

2. Make any other modifications to the script as needed (e.g. object naming, etc.)

3. Run the script:
```sh
cd data_prep/gso
python create_list.py
```
This will create shard list files in the `shard_lists` directory, each containing a subset of objects to render.

### Rendering a Dataset

To render a dataset (e.g., GSO) in parallel, follow these steps:

1. Prepare the shard lists as described above.

2. Run the distributed rendering script:
```sh
python distributed.py --num_workers 6 --output_dir your_output_path --config configs/gso.json --list_file path/to/your/shard_lists/0.txt
```

The `distributed.py` script uses a list file to refer to the objects being rendered. Make sure to provide the correct path to your shard list file using the `--list_file` argument.

To render multiple shards, you can run the `distributed.py` script multiple times with different shard list files, or create a shell script to automate the process.

## Configuration

The `configs/gso.json` file contains common rendering parameters. You can modify this file to adjust rendering settings for all objects in a dataset.

For more options and customization, please refer to the individual script files and their command-line arguments.