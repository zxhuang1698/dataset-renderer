import blenderproc as bproc
import numpy as np
import os
import json
import math
import random
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

def uniform_cameras(samples=100, radius=1.8):
    '''
    Generate camera positions evenly distributed on a sphere.
    https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    Returns a list of tuples (x, y, z) representing the camera positions.
    '''
    cameras = []
    phi = math.pi * (math.sqrt(5.) - 1.)
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2
        r = math.sqrt(1 - y * y)
        theta = phi * i
        x = math.cos(theta) * r
        z = math.sin(theta) * r
        cameras.append((x * radius, y * radius, z * radius))
    return cameras, None

def orbit_cameras(n_azim=8, n_elev=4, radius=1.8, elev_range=(0, 60), random_offset=False):
    '''
    Generate camera positions on multiple orbits evenly distributed in the elevation range.
    Returns a list of tuples (x, y, z) representing the camera positions and annotations.
    '''
    cameras = []
    annos = {"azimuth": [], "polar": []}
    azims = np.linspace(0, 360, n_azim, endpoint=False)
    elevs = np.linspace(elev_range[0], elev_range[1], n_elev)
    for elev in elevs:
        azim_offset = np.random.uniform(-180/n_azim, 180/n_azim) if random_offset else 0
        for azim in azims:
            azim_rad = (azim + azim_offset) * math.pi / 180
            polar_rad = (90 - elev) * math.pi / 180
            x = radius * math.sin(polar_rad) * math.cos(azim_rad)
            y = radius * math.sin(polar_rad) * math.sin(azim_rad)
            z = radius * math.cos(polar_rad)
            cameras.append((x, y, z))
            annos["azimuth"].append(azim_rad)
            annos["polar"].append(polar_rad)
    return cameras, annos

def render_gso_object(
        obj_path, output_base_path, hdri_path, 
        fov=0.598132, camera_distance=2.4, resolution=512,
        n_azim=8, n_elev=4, generate_pc=True, n_sphere_cam=100,
        random_offset=False, save_depths=False, save_normals=False, 
        save_diffuses=False, fov_range=0, camera_distance_range=0,
        max_elev=60
    ):
                      
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
    original_fov = fov
    fov = np.random.uniform(
        fov - fov_range, fov + fov_range
    ) if fov_range > 0 else fov
    # Adjust camera distance proportionally to the focal length
    focal_ratio = math.tan(original_fov / 2) / math.tan(fov / 2)
    camera_distance *= focal_ratio
    camera_distance = np.random.uniform(
        camera_distance - camera_distance_range, camera_distance + camera_distance_range
    ) if camera_distance_range > 0 else camera_distance
    bproc.camera.set_resolution(resolution, resolution)
    bproc.camera.set_intrinsics_from_blender_params(lens=fov, lens_unit="FOV")

    # Define rendering settings
    bproc.renderer.set_max_amount_of_samples(128)
    bproc.renderer.set_noise_threshold(0.01)

    # Create output directories
    images_path = os.path.join(output_base_path, "images")
    depths_path = os.path.join(output_base_path, "depths")
    normals_path = os.path.join(output_base_path, "normals")
    diffuses_path = os.path.join(output_base_path, "diffuses")
    annos_path = os.path.join(output_base_path, "annos")
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(annos_path, exist_ok=True)
    if save_depths:
        os.makedirs(depths_path, exist_ok=True)
    if save_normals:
        os.makedirs(normals_path, exist_ok=True)
    if save_diffuses:
        os.makedirs(diffuses_path, exist_ok=True)
    
    # Generate camera poses
    camera_locs, annos = orbit_cameras(
        n_azim=n_azim, n_elev=n_elev, radius=camera_distance, 
        elev_range=(0, max_elev), random_offset=random_offset
    )
    if generate_pc:
        assert n_sphere_cam > 0
        sphere_cam_locs, _ = uniform_cameras(samples=n_sphere_cam, radius=camera_distance)
        camera_locs += sphere_cam_locs
    for loc in camera_locs:
        loc_np = np.array(loc)
        rotation_matrix = bproc.camera.rotation_from_forward_vec(-loc_np)
        cam2world_matrix = bproc.math.build_transformation_mat(loc_np, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)

    # Set output format
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_diffuse_color_output()
    bproc.renderer.set_output_format(enable_transparency=True)
    
    # Render
    data = bproc.renderer.render()
    
    # Save images and annotations
    num_images = len(annos["azimuth"]) 
    for i in range(num_images):
        img = data["colors"][i]
        # save image with opencv
        image_path = os.path.join(images_path, f"{i:04d}.png")
        cv2.imwrite(image_path, cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
        
        # save diffuse color
        diffuse = data["diffuse"][i]
        diffuse_path = os.path.join(diffuses_path, f"{i:04d}.png")
        if save_diffuses:
            cv2.imwrite(diffuse_path, cv2.cvtColor(diffuse, cv2.COLOR_RGB2BGR))
        
        # save depth as npy
        depth = data["depth"][i]
        depth_path = os.path.join(depths_path, f"{i:04d}.npy")
        if save_depths:
            np.save(depth_path, depth)
        
        # save normals as npy
        normal = data["normals"][i]
        normal_path = os.path.join(normals_path, f"{i:04d}.npy")
        if save_normals:
            np.save(normal_path, normal)
        
        # save annotation
        anno = {
            "camera_dist": camera_distance,
            "azimuth": annos["azimuth"][i],
            "polar": annos["polar"][i],
            "fov_rad": fov,
        }
        anno_path = os.path.join(annos_path, f"{i:04d}.json")
        with open(anno_path, 'w') as f:
            json.dump(anno, f, indent=2)
            
    if generate_pc:
        # generate point cloud using back projection
        pc_all = []
        for i in range(num_images, num_images + n_sphere_cam):
            pc_xyz = bproc.camera.pointcloud_from_depth(data["depth"][i], frame=i) # [H, W, 3]
            pc_xyz = pc_xyz[data["depth"][i] < 100] # [N, 3]
            pc_rgb = data["diffuse"][i] # [H, W, 3]
            pc_rgb = pc_rgb[data["depth"][i] < 100] # [N, 3]
            pc = np.concatenate([pc_xyz, pc_rgb], axis=-1) # [N, 6]
            pc_all.append(pc)

        # Concatenate all point clouds and randomly sample 10000 points
        points = np.concatenate(pc_all, axis=0)
        points = points[np.random.choice(points.shape[0], 10000, replace=False)]
        
        # Save point cloud as PLY
        point_cloud = trimesh.PointCloud(vertices=points[:, :3], colors=points[:, 3:])
        point_cloud.export(os.path.join(output_base_path, "pointcloud.ply"))

# Function to get all HDRI files recursively
def get_hdri_files(hdri_base_path):
    hdri_files = []
    for root, dirs, files in os.walk(hdri_base_path):
        for file in files:
            if file.endswith('.exr') or file.endswith('.hdr'):
                hdri_files.append(os.path.join(root, file))
    return hdri_files

def boolean_string(s: str) -> bool:
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    elif s.lower() == 'true':
        return True
    else:
        return False

def main():
    parser = argparse.ArgumentParser(description="Render a single object with a randomly selected HDRI.")
    parser.add_argument('--input', type=str, help='Path to the input model file',
                        default="examples/mesh/model.obj")
    parser.add_argument('--output', type=str, help='Path to the output directory',
                        default="examples/output")
    parser.add_argument('--hdri_path', type=str, help='Path to the HDRI file', 
                        default="/home/zixuan32/projects/rendering/blender_proc/assets/hdris")
    parser.add_argument('--base_fov', type=float, 
                        help='Base field of view in radians (actual fov affected by fov_range)', default=0.598132)
    parser.add_argument('--base_cam_dist', type=float, 
                        help='Base camera distance from the object (actual cam_dist affected by fov_range and camera_distance_range)', default=2.4)
    parser.add_argument('--resolution', type=int, help='Resolution of the output images', default=512)
    parser.add_argument('--n_azim', type=int, help='Number of azimuth angles', default=8)
    parser.add_argument('--n_elev', type=int, help='Number of orbits', default=4)
    parser.add_argument('--generate_pc', type=boolean_string, help='Generate point cloud', default=True)
    parser.add_argument('--n_sphere_cam', type=int, help='Number of cameras on the sphere', default=32)
    parser.add_argument('--random_offset', type=boolean_string, help='Random offset for azimuth angles', default=False)
    parser.add_argument('--save_depths', type=boolean_string, help='Save depth rendering', default=False)
    parser.add_argument('--save_normals', type=boolean_string, help='Save normal rendering', default=False)
    parser.add_argument('--save_diffuses', type=boolean_string, help='Save diffuse rendering', default=False)
    parser.add_argument('--fov_range', type=float, help='Range of field of view in radians', default=0)
    parser.add_argument('--cam_dist_range', type=float, help='Range of camera distance from the object', default=0)
    parser.add_argument('--max_elev', type=float, help='Max elevation angle in degrees', default=60)
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    os.makedirs(output_path, exist_ok=True)

    # Get list of HDRI files, which can be downloaded with e.g.
    # blenderproc download haven <hdri_path>
    hdri_files = get_hdri_files(args.hdri_path)
    if os.path.exists(input_path):
        hdri_path = random.choice(hdri_files)
        print(f"Rendering object: {input_path}")
        print(f"Using HDRI: {hdri_path}")
        render_gso_object(
            input_path, output_path, hdri_path, 
            fov=args.base_fov, camera_distance=args.base_cam_dist, resolution=args.resolution,
            n_azim=args.n_azim, n_elev=args.n_elev, generate_pc=args.generate_pc, 
            n_sphere_cam=args.n_sphere_cam, random_offset=args.random_offset,
            save_depths=args.save_depths, save_normals=args.save_normals,
            save_diffuses=args.save_diffuses, fov_range=args.fov_range, 
            camera_distance_range=args.cam_dist_range, max_elev=args.max_elev
        )
    else:
        print(f"Object file not found: {input_path}")

# blenderproc run worker.py
if __name__ == "__main__":
    main()