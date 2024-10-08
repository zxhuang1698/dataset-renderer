import os
import json
import logging
import multiprocessing
import subprocess
import shutil
from pathlib import Path
from argparse import ArgumentParser

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def read_object_paths(list_file):
    with open(list_file, 'r') as f:
        lines = f.readlines()
    # Each line is structured as "name_of_object  path_to_object", so we take both parts
    object_paths_and_names = [(line.split()[1], line.split()[0]) for line in lines]
    return object_paths_and_names

def render_object(args):
    (obj_path, obj_name), output_dir, worker_script, common_args = args
    obj_output_dir = os.path.join(output_dir, obj_name)
    os.makedirs(obj_output_dir, exist_ok=True)
    
    # Convert common_args to list and add '--' before each key
    common_args_list = []
    for key, value in common_args.items():
        common_args_list.append('--' + str(key))
        common_args_list.append(str(value))

    cmd = [
        "blenderproc", "run", worker_script,
        "--input", obj_path,
        "--output", obj_output_dir
    ] + common_args_list

    logging.info(f"Rendering {obj_path} to {obj_output_dir}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        logging.info(f"Successfully rendered {obj_path}")
    else:
        logging.error(f"Failed to render {obj_path}: {result.stderr}")

    return {
        "object": obj_path,
        "output": obj_output_dir,
        "status": "success" if result.returncode == 0 else "failure",
        "cmd": ' '.join(cmd),
        "error": result.stderr if result.returncode != 0 else None
    }

def main():
    parser = ArgumentParser(description="Distributed rendering of objects using BlenderProc.")
    parser.add_argument('--output_dir', type=str, default='outputs/test', help='Path to the output directory')
    parser.add_argument('--worker_script', type=str, default='worker.py', help='Path to the worker script (worker.py)')
    parser.add_argument('--num_workers', type=int, default=6, help='Number of parallel workers')
    parser.add_argument('--config', type=str, default='configs/gso.json', 
                        help='Path to the JSON file containing common arguments for the worker script')
    parser.add_argument('--list_file', type=str, default='data_prep/gso/shard_lists/0.txt',
                        help='Path to the list file containing object paths')
    args = parser.parse_args()

    shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, 'rendering.log'))

    # read the config file and copy it to the output directory
    with open(args.config, 'r') as f:
        common_args = json.load(f)
    shutil.copy(args.config, os.path.join(args.output_dir, 'config.json'))

    obj_files_and_names = read_object_paths(args.list_file)
    logging.info(f"Found {len(obj_files_and_names)} object files in {args.list_file}")

    pool_args = [(obj, args.output_dir, args.worker_script, common_args) for obj in obj_files_and_names]
    with multiprocessing.Pool(args.num_workers) as pool:
        results = pool.map(render_object, pool_args)

    results_file = os.path.join(args.output_dir, 'rendering_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info(f"Rendering completed. Results saved to {results_file}")

if __name__ == "__main__":
    main()