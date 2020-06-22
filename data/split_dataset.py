from glob import glob
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
import sys

np.random.seed(27)

SPLIT_DIR = "lmd_full_split"


def move_file(file, label):
    Path(f"{SPLIT_DIR}/{label}/").mkdir(parents=True, exist_ok=True)
    
    file_name = Path(file).name
    shutil.move(file, f"{SPLIT_DIR}/{label}/" + file_name)


def split_dataset(dataset_dir):
    files = glob(f"{dataset_dir}/**/*.mid", recursive=True)
    n_files = len(files)

    split = np.array([0.8, 0.1, 0.1])
    split_lengths = (split * n_files).astype(np.int32)
    split_lengths[0] = n_files - split_lengths[1:].sum()

    indices = np.random.permutation(n_files)

    index = split_lengths.cumsum()

    indices_train = indices[:index[0]]
    indices_val = indices[index[0]:index[1]]
    indices_test = indices[index[1]:]

    moved_files = []
    print("Split up dataset...")
    for label, label_indices in zip(["train", "val", "test"], [indices_train, indices_val, indices_test]):
        print(f"Copy {len(label_indices)} files to {label}.")

        for idx in tqdm(label_indices):
            assert files[idx] not in moved_files
            move_file(files[idx], label)
            moved_files.append(files[idx])


def clean_up(dataset_dir):
    print("Clean up files...")
    shutil.rmtree(dataset_dir)


def main(dataset_dir):
    split_dataset(dataset_dir)
    # clean_up(dataset_dir)


if __name__ == "__main__":
    main(sys.argv[1])
