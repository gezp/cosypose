import numpy as np
import pandas as pd

from cosypose.config import LOCAL_DATA_DIR, BOP_DS_DIR

from .bop_object_datasets import BOPObjectDataset

from .urdf_dataset import BOPUrdfDataset



def make_object_dataset(ds_name):
    ds = None
    if ds_name == 'tless.cad':
        ds = BOPObjectDataset(BOP_DS_DIR / 'tless/models_cad')
    elif ds_name == 'tless.eval' or ds_name == 'tless.bop':
        ds = BOPObjectDataset(BOP_DS_DIR / 'tless/models_eval')

    # YCBV
    elif ds_name == 'ycbv.bop':
        ds = BOPObjectDataset(BOP_DS_DIR / 'ycbv/models')
    elif ds_name == 'ycbv.bop-compat':
        # BOP meshes (with their offsets) and symmetries
        # Replace symmetries of objects not considered symmetric in PoseCNN
        ds = BOPObjectDataset(BOP_DS_DIR / 'ycbv/models_bop-compat')
    elif ds_name == 'ycbv.bop-compat.eval':
        # PoseCNN eval meshes and symmetries, WITH bop offsets
        ds = BOPObjectDataset(BOP_DS_DIR / 'ycbv/models_bop-compat_eval')

    # Other BOP
    elif ds_name == 'hb':
        ds = BOPObjectDataset(BOP_DS_DIR / 'hb/models')
    elif ds_name == 'icbin':
        ds = BOPObjectDataset(BOP_DS_DIR / 'icbin/models')
    elif ds_name == 'itodd':
        ds = BOPObjectDataset(BOP_DS_DIR / 'itodd/models')
    elif ds_name == 'lm':
        ds = BOPObjectDataset(BOP_DS_DIR / 'lm/models')
    elif ds_name == 'tudl':
        ds = BOPObjectDataset(BOP_DS_DIR / 'tudl/models')

    else:
        raise ValueError(ds_name)
    return ds


def make_urdf_dataset(ds_name):
    if isinstance(ds_name, list):
        ds_index = []
        for ds_name_n in ds_name:
            dataset = make_urdf_dataset(ds_name_n)
            ds_index.append(dataset.index)
        dataset.index = pd.concat(ds_index, axis=0)
        return dataset

    # BOP
    if ds_name == 'tless.cad':
        ds = BOPUrdfDataset(LOCAL_DATA_DIR / 'urdfs' / 'tless.cad')
    elif ds_name == 'tless.reconst':
        ds = BOPUrdfDataset(LOCAL_DATA_DIR / 'urdfs' / 'tless.reconst')
    elif ds_name == 'ycbv':
        ds = BOPUrdfDataset(LOCAL_DATA_DIR / 'urdfs' / 'ycbv')
    elif ds_name == 'hb':
        ds = BOPUrdfDataset(LOCAL_DATA_DIR / 'urdfs' / 'hb')
    elif ds_name == 'icbin':
        ds = BOPUrdfDataset(LOCAL_DATA_DIR / 'urdfs' / 'icbin')
    elif ds_name == 'itodd':
        ds = BOPUrdfDataset(LOCAL_DATA_DIR / 'urdfs' / 'itodd')
    elif ds_name == 'lm':
        ds = BOPUrdfDataset(LOCAL_DATA_DIR / 'urdfs' / 'lm')
    elif ds_name == 'tudl':
        ds = BOPUrdfDataset(LOCAL_DATA_DIR / 'urdfs' / 'tudl')

    # Custom scenario
    elif 'custom' in ds_name:
        scenario = ds_name.split('.')[1]
        ds = BOPUrdfDataset(LOCAL_DATA_DIR / 'scenarios' / scenario / 'urdfs')

    elif ds_name == 'camera':
        ds = OneUrdfDataset(ASSET_DIR / 'camera/model.urdf', 'camera')
    else:
        raise ValueError(ds_name)
    return ds


def make_texture_dataset(ds_name):
    if ds_name == 'shapenet':
        ds = TextureDataset(LOCAL_DATA_DIR / 'texture_datasets' / 'shapenet')
    else:
        raise ValueError(ds_name)
    return ds
