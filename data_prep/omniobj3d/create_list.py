import os
import shutil

def read_object_list(raw_list_path):
    # read each line in render.list and take the first part separated by a space
    raw_list = open(raw_list_path, 'r').readlines()
    object_list = [x.split(' ')[0] for x in raw_list]
    return object_list

def create_shard_list(data_root, object_list, num_shards):
    # write the objects to a set of new files
    # structured as "name_of_object  path_to_object" 
    # number of files is num_shards
    shard_size = len(object_list) // num_shards + 1
    shutil.rmtree('shard_lists', ignore_errors=True)
    os.makedirs('shard_lists', exist_ok=True)
    for i in range(num_shards):
        with open(f'shard_lists/{i}.txt', 'w') as f:
            for obj in object_list[i*shard_size:(i+1)*shard_size]:
                obj_path = os.path.join(data_root, obj, "Scan", "Scan.obj")
                f.write(f"{obj}  {obj_path}\n")
            actual_shard_size = len(object_list[i*shard_size:(i+1)*shard_size])
            print(f"Created shard_lists/{i}.txt with {actual_shard_size} objects.")
        f.close()

def main():
    data_root = '/home/zixuan32/datasets/omniobj3d_mesh'
    raw_list_path = 'object_list.txt'
    num_shards = 1
    object_list = read_object_list(raw_list_path)
    create_shard_list(data_root, object_list, num_shards)

if __name__ == '__main__':
    main()