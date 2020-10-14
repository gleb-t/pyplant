import os
from typing import Any, Optional, Dict

import numpy as np

from pyplant.pyplant import IngredientTypeSpec, Warehouse


class HdfArraySpec(IngredientTypeSpec):

    def __init__(self):
        import h5py

        super().__init__()
        self.isAllocatable = True

        self.h5Files = {}  # type: Dict[str, h5py.File]

    @classmethod
    def get_name(cls) -> str:
        return 'hdf_array'

    def store(self, warehouse: Warehouse, name: str, value: Any) -> Optional[Dict[str, Any]]:
        # H5py takes care of storing to disk on-the-fly.
        # Just flush the data, to make sure that it is persisted.
        self.h5Files[name].flush()
        return None

    def fetch(self, warehouse: Warehouse, name: str, meta: Optional[Dict[str, Any]] = None) -> Any:
        import h5py

        h5FilePath = self._get_hdf_array_filepath(warehouse, name)
        if name not in self.h5Files:
            if not os.path.exists(h5FilePath):
                return None

            # If the array isn't in the manifest, but exists on disk, attempt to open it.
            try:
                self.h5Files[name] = h5py.File(h5FilePath, 'a')
            except OSError as e:
                warehouse.logger.warning("Failed to open the HDF array at '{}' with error: {}"
                                         .format(h5FilePath, str(e)))
                warehouse.logger.info("Deleting the corrupted file.")
                os.remove(h5FilePath)
                return None

        dataset = None
        try:
            if 'data' in self.h5Files[name]:
                # Try accessing the dataset. (Can crush for corrupted files.)
                dataset = self.h5Files[name]['data']
        except BaseException as e:
            warehouse.logger.warning("Suppressed an error while accessing HDF-dataset '{}' Details: {}"
                                     .format(name, e))

        if dataset is None:
            warehouse.logger.info("The HDF file at '{}' has no dataset. Corrupted file, deleting.".format(h5FilePath))
            try:
                self.h5Files[name].close()
                os.remove(h5FilePath)
            except BaseException as e:
                warehouse.logger.warning("Suppressed an error while removing HDF-dataset '{}' Details: {}"
                                         .format(name, e))
            return None

        return dataset

    def allocate(self, warehouse: Warehouse, name: str, **kwargs) -> Any:
        import h5py

        shape = kwargs.pop('shape')
        dtype = kwargs.pop('dtype') or np.float32

        dataset = self.fetch(warehouse, name)
        h5FilePath = self._get_hdf_array_filepath(warehouse, name)

        # todo should probably always recreate the array, its cheap and consistent with BNAs.

        # If the dataset already exists, but has a wrong shape/type, recreate it.
        if dataset is not None and (dataset.shape != shape or dataset.dtype != dtype):
            try:
                self.h5Files[name].close()
                os.remove(h5FilePath)
            except RuntimeError as e:
                warehouse.logger.warning("Suppressed an error while removing dataset '{}' Details: {}"
                                         .format(name, e))
            dataset = None

        if dataset is None:
            self.h5Files[name] = h5py.File(h5FilePath, 'a')
            dataset = self.h5Files[name].create_dataset('data', shape=shape, dtype=dtype, **kwargs)

        return dataset

    def deallocate(self, warehouse: Warehouse, name: str):
        h5FilePath = self._get_hdf_array_filepath(warehouse, name)

        if name not in self.h5Files:
            raise RuntimeError("Cannot deallocate HDF array '{}', it doesn't exist.".format(name))

        if not os.path.exists(h5FilePath):
            raise RuntimeError("Cannot deallocate HDF array '{}', the file doesn't exist: '{}'."
                               .format(name, h5FilePath))

        self.h5Files[name].close()
        del self.h5Files[name]
        os.unlink(h5FilePath)

    def close(self):
        for name, h5File in list(self.h5Files.items()):
            h5File.close()
            del self.h5Files[name]

    def is_instance(self, value) -> bool:
        import h5py
        return isinstance(value, h5py.Dataset)

    def _get_hdf_array_filepath(self, warehouse: Warehouse, name: str):
        return os.path.join(warehouse.baseDir, name + '.hdf')


class ScipySparseSpec(IngredientTypeSpec):

    @classmethod
    def get_name(cls) -> str:
        return 'scipy_sparse'

    def store(self, warehouse: Warehouse, name: str, value: Any) -> Optional[Dict[str, Any]]:
        import scipy.sparse as sp
        sp.save_npz(os.path.join(warehouse.baseDir, '{}.npz'.format(name)), value)

    def fetch(self, warehouse: Warehouse, name: str, meta: Optional[Dict[str, Any]]) -> Any:
        import scipy.sparse as sp
        return sp.load_npz(os.path.join(warehouse.baseDir, '{}.npz'.format(name)))

    def is_instance(self, value) -> bool:
        import scipy.sparse as sp
        return sp.issparse(value)
