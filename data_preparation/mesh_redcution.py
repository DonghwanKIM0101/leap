import pymeshlab as ml

import argparse
from glob import glob
import json
import os
from os.path import basename, join, splitext
from pathlib import Path
from tqdm import tqdm

import numpy as np
from training_code.datasets.mesh_utils import *
from trimesh import Trimesh, repair, load_mesh

TARGET = 6890

def main(args):
    for split in args.splits:

        split_path = join(args.src_dataset_path, f"{split}.txt")
        with open(split_path, 'r') as split_file:
            split_dirs = split_file.read()
            split_dirs = split_dirs.replace('\n', ' ').split()

        for seq_idx, seq_dir in enumerate(split_dirs):
            print(f'Processing {seq_idx+1} out of {len(split_dirs)} seqs...')
            seq_path = join(args.src_dataset_path, seq_dir, 'tight_meshes')
            result_path = join(args.src_dataset_path, seq_dir, 'reduced_meshes')
            Path(result_path).mkdir(parents=True, exist_ok=True)

            for frame_idx in range(len(glob(join(seq_path, "*.obj")))):
                frame_format = "%06d"%(6*frame_idx)

                mesh_path = join(seq_path, f"{frame_format}.obj")
                result_mesh_path = join(result_path, f"{frame_format}.obj")

                if (os.path.isfile(result_mesh_path) or not os.path.isfile(mesh_path)):
                    continue

                ms = ml.MeshSet()
                ms.load_new_mesh(mesh_path)
                m = ms.current_mesh()
                # print('input:', mesh_path)
                # print('input mesh has', m.vertex_number(), 'vertex and', m.face_number(), 'faces')
                
                numFaces = 100 + 2 * TARGET

                while (ms.current_mesh().vertex_number() > TARGET):
                    ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=numFaces, preservenormal=True)
                    #Refine our estimation to slowly converge to TARGET vertex number
                    numFaces = numFaces - (ms.current_mesh().vertex_number() - TARGET)

                m = ms.current_mesh()
                # print('output mesh has', m.vertex_number(), 'vertex and', m.face_number(), 'faces')
                ms.save_current_mesh(result_mesh_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Preprocess HuMMan dataset.')
    parser.add_argument('--src_dataset_path', type=str, required=True,
                        help='Path to HuMMan dataset.')
    parser.add_argument('--splits', type=list, required=False, default=['train', 'test'],
                        help='Split of HuMMan to use, separated by comma.')

    main(parser.parse_args())
