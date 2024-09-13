import blenderproc as bproc
import numpy as np
import os
import json
import math
from pathlib import Path
import random
from PIL import Image
import trimesh
import cv2
import argparse
from typing import List

def find_minimum_bounding_cuboid(cuboids):
    """
    Find the minimum cuboid that contains all cuboids.
    
    Args:
    cuboids (list of numpy arrays): A list where each element is an 8x3 numpy array representing a cuboid's vertices.
    
    Returns:
    numpy array: An 8x3 array representing the vertices of the minimum bounding cuboid.
    """
    # Initialize the minimum and maximum values with large/small values
    global_min = np.inf * np.ones(3)
    global_max = -np.inf * np.ones(3)

    # Iterate over all cuboids and find global min/max
    for cuboid in cuboids:
        cuboid_min = np.min(cuboid, axis=0)  # Minimum x, y, z for the current cuboid
        cuboid_max = np.max(cuboid, axis=0)  # Maximum x, y, z for the current cuboid

        # Update the global minimum and maximum
        global_min = np.minimum(global_min, cuboid_min)
        global_max = np.maximum(global_max, cuboid_max)

    # Construct the 8 vertices of the bounding cuboid
    # The vertices are combinations of the min and max values for x, y, z
    bounding_cuboid = np.array([
        [global_min[0], global_max[1], global_min[2]],  # Vertex 1: (min, max, min)
        [global_min[0], global_min[1], global_min[2]],  # Vertex 2: (min, min, min)
        [global_min[0], global_min[1], global_max[2]],  # Vertex 3: (min, min, max)
        [global_min[0], global_max[1], global_max[2]],  # Vertex 4: (min, max, max)
        [global_max[0], global_max[1], global_min[2]],  # Vertex 5: (max, max, min)
        [global_max[0], global_min[1], global_min[2]],  # Vertex 6: (max, min, min)
        [global_max[0], global_min[1], global_max[2]],  # Vertex 7: (max, min, max)
        [global_max[0], global_max[1], global_max[2]],  # Vertex 8: (max, max, max)
    ])

    return bounding_cuboid

def normalize_objs(objs: List[bproc.types.MeshObject]):
    # get the bounding box that contains all objects
    # each 8x3 np array describing the object aligned bounding box coordinates in world coordinates
    bounding_cuboids = [obj.get_bound_box() for obj in objs]
    bounding_cuboid = find_minimum_bounding_cuboid(bounding_cuboids)
    # get the center and scale as the middle point and the largest dimension
    center = (bounding_cuboid[1] + bounding_cuboid[7]) / 2
    length = np.max(bounding_cuboid[7] - bounding_cuboid[1])
    scale = 1 / length
    # normalize all objects
    for obj in objs:
        obj.set_location(obj.get_location() - center)
        obj.persist_transformation_into_mesh()
        obj.set_scale(obj.get_scale() * scale)
        obj.persist_transformation_into_mesh()

def render_gso_object(obj_path, output_base_path, hdri_path):
    bproc.init()
    # Load object
    objs = bproc.loader.load_obj(obj_path)
    
    # Reset the rotation of all objects
    for obj in objs:
        obj.set_rotation_euler([0, 0, 0])
        obj.persist_transformation_into_mesh()
    
    normalize_objs(objs)

    # Set HDRI background
    bproc.world.set_world_background_hdr_img(hdri_path)

    # Set up camera
    fov = 0.698132
    bproc.camera.set_resolution(512, 512)
    bproc.camera.set_intrinsics_from_blender_params(lens=fov, lens_unit="FOV")

    # Define rendering settings
    bproc.renderer.set_max_amount_of_samples(128)
    bproc.renderer.set_noise_threshold(0.01)

    # Create output directories
    images_path = os.path.join(output_base_path, "images")
    annos_path = os.path.join(output_base_path, "annos")
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(annos_path, exist_ok=True)

    azims = [0, 90, 180, 270]
    elevs = [0, 15, 30, 45]
    annos = {
        "camera_dist": [],
        "azimuth": [],
        "polar": [],
        "fov_rad": [],
    }
    # Render from different viewpoints
    for elev in elevs:
        azim_offset = random.uniform(-45, 45)
        for azim in azims:
            # Calculate camera position
            azim_rad = (azim + azim_offset) * math.pi / 180
            polar_rad = (90 - elev) * math.pi / 180
            r = 1.8  # camera distance

            x = r * math.sin(polar_rad) * math.cos(azim_rad)
            y = r * math.sin(polar_rad) * math.sin(azim_rad)
            z = r * math.cos(polar_rad)
            location = np.array([x, y, z])

            # Set camera pose
            rotation_matrix = bproc.camera.rotation_from_forward_vec(-location)
            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
            bproc.camera.add_camera_pose(cam2world_matrix)
            
            # Write annotation
            annos["camera_dist"].append(r)
            annos["azimuth"].append(azim_rad)
            annos["polar"].append(polar_rad)
            annos["fov_rad"].append(fov)

    # Render
    data = bproc.renderer.render()
    
    # Save images and annotations
    num_images = len(data["colors"])
    for i in range(num_images):
        rgb = data["colors"][i]
        # save image with opencv
        image_path = os.path.join(images_path, f"image_{i:04d}.png")
        cv2.imwrite(image_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        
        # save annotation
        anno = {
            "camera_dist": annos["camera_dist"][i],
            "azimuth": annos["azimuth"][i],
            "polar": annos["polar"][i],
            "fov_rad": annos["fov_rad"][i],
        }
        anno_path = os.path.join(annos_path, f"anno_{i:04d}.json")
        with open(anno_path, 'w') as f:
            json.dump(anno, f, indent=2)

    # Generate point cloud using trimesh
    all_meshes = []
    for obj in objs:
        mesh = obj.get_mesh()
        vertices = np.array([v.co for v in mesh.vertices])
        faces = np.array([f.vertices for f in mesh.polygons])
        
        # Create trimesh object
        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        all_meshes.append(tri_mesh)
        
    # Combine all meshes into a single mesh
    combined_mesh = trimesh.util.concatenate(all_meshes)
    
    # Sample points on the surface
    points = combined_mesh.sample(10000)
    
    # Save point cloud as PLY
    point_cloud = trimesh.points.PointCloud(points)
    point_cloud.export(os.path.join(output_base_path, "pointcloud.ply"))

# Function to get all HDRI files recursively
def get_hdri_files(hdri_base_path):
    hdri_files = []
    for root, dirs, files in os.walk(hdri_base_path):
        for file in files:
            if file.endswith('.exr') or file.endswith('.hdr'):
                hdri_files.append(os.path.join(root, file))
    return hdri_files

def main():
    parser = argparse.ArgumentParser(description="Render a single object with a randomly selected HDRI.")
    parser.add_argument('--input', type=str, help='Path to the input model file',
                        default="examples/mesh/model.obj")
    parser.add_argument('--output', type=str, help='Path to the output directory',
                        default="examples/output")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    hdri_base_path = "/home/zixuan32/projects/rendering/blender_proc/assets/hdris"
    os.makedirs(output_path, exist_ok=True)

    # Get list of HDRI files
    hdri_files = get_hdri_files(hdri_base_path)

    if os.path.exists(input_path):
        hdri_path = random.choice(hdri_files)
        print(f"Rendering object: {input_path}")
        print(f"Using HDRI: {hdri_path}")
        render_gso_object(input_path, output_path, hdri_path)
    else:
        print(f"Object file not found: {input_path}")

# blenderproc run worker.py
if __name__ == "__main__":
    main()