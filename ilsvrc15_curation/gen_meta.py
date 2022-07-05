import os
from pathlib import Path

import numpy as np


def prepare_imagenet_vid(train_subsets: set = None):
    meta_dir = Path('datasets/meta')
    with np.load(os.path.join(meta_dir, 'imagenet.npz'), allow_pickle=True) as data:
        trn_list = data['trn_lst']
        trn_lb = data['trn_lb']
        val_list = data['val_lst']
        val_lb = data['val_lb']

    if train_subsets:
        def subset_selector(arr: np.array):
            mask = [i[0][0] in train_subsets for i in arr]
            return arr[mask]
        subset_selector_func = np.vectorize(pyfunc=subset_selector)
        trn_list = subset_selector_func(trn_list)
    print(f"trn_list: {len(trn_list)}, trn_lb: {len(trn_lb)}; val_list: {len(val_list)}, val_lb: {len(val_lb)}")
    np.save(os.path.join(meta_dir, 'train.npy'), trn_list)
    np.save(os.path.join(meta_dir, 'valid.npy'), val_list)


def set_working_dir():
    if os.getcwd().endswith("ilsvrc15_curation"):
        parent_dir = Path(os.getcwd()).parent
        os.chdir(parent_dir)
    return os.getcwd()


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disable GPU
    set_working_dir()
    prepare_imagenet_vid(train_subsets=set("a"))
