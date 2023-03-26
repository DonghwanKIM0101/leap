import os
from glob import glob
from os.path import join
from pathlib import Path

MANIFOLD_PATH = '/hdd1/Manifold/build'
SRC_DATASET_PATH = '/hdd1/HuMMan'
SPLITS = ['train', 'test']

os.chdir(MANIFOLD_PATH)

for split in SPLITS:
    split_path = join(SRC_DATASET_PATH, f"{split}.txt")
    with open(split_path, 'r') as split_file:
        split_dirs = split_file.read()
        split_dirs = split_dirs.replace('\n', ' ').split()
    
    for seq_idx, seq_dir in enumerate(split_dirs):
        print(f'Processing {seq_idx+1} out of {len(split_dirs)} seqs...')
        seq_path = join(SRC_DATASET_PATH, seq_dir, 'textured_meshes')
        result_path = join(SRC_DATASET_PATH, seq_dir, 'tight_meshes')
        Path(result_path).mkdir(parents=True, exist_ok=True)

        for frame_idx in range(len(glob(join(seq_path, "*.obj")))):
            frame_format = "%06d"%(6*frame_idx)
            mesh_path = join(seq_path, f"{frame_format}.obj")
            result_mesh_path = join(result_path, f"{frame_format}.obj")
            if (not os.path.isfile(result_mesh_path)):
                os.system(f"./manifold {mesh_path} {result_mesh_path}")