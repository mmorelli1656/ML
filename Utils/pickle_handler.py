# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 09:54:54 2025

@author: WKS
"""

import pickle
from pathlib import Path
from typing import Any, Optional, Union, Callable

class PickleHandler:
    """
    Utility class for saving and loading Python objects using pickle.

    This class provides static methods to serialize and deserialize Python
    objects to and from files, with basic error handling and optional logging.

    Methods
    -------
    save_pickle(obj, file_path, logger=None)
        Serialize and save an object to a file.
    load_pickle(file_path, logger=None)
        Load and deserialize an object from a pickle file.

    Examples
    --------
    >>> PickleHandler.save_pickle(data, "data.pkl")
    >>> loaded_data = PickleHandler.load_pickle("data.pkl")
    """

    @staticmethod
    def save_pickle(
        obj: Any, 
        file_path: Union[str, Path], 
        logger: Optional[Callable[[str], None]] = None
    ) -> None:
        """
        Save a Python object to a file using pickle.

        Parameters
        ----------
        obj : Any
            Python object to serialize.
        file_path : str or Path
            Path to the file where the object will be saved.
        logger : callable, optional
            Optional logging function, e.g. print or logger.info.
            If None, defaults to print().
        """
        if logger is None:
            logger = print

        file_path = Path(file_path)
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(obj, f)
            logger(f"[PickleHandler] Object saved to: {file_path}")
        except Exception as e:
            logger(f"[PickleHandler][ERROR] Failed to save object to {file_path}: {e}")
            raise

    @staticmethod
    def load_pickle(
        file_path: Union[str, Path], 
        logger: Optional[Callable[[str], None]] = None
    ) -> Any:
        """
        Load a Python object from a pickle file.

        Parameters
        ----------
        file_path : str or Path
            Path to the pickle file to load.
        logger : callable, optional
            Optional logging function, e.g. print or logger.info.
            If None, defaults to print().

        Returns
        -------
        Any
            The deserialized Python object.
        """
        if logger is None:
            logger = print

        file_path = Path(file_path)
        try:
            with open(file_path, 'rb') as f:
                obj = pickle.load(f)
            logger(f"[PickleHandler] Object loaded from: {file_path}")
            return obj
        except Exception as e:
            logger(f"[PickleHandler][ERROR] Failed to load object from {file_path}: {e}")
            raise
