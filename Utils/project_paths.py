# -*- coding: utf-8 -*-

"""
ProjectPaths: Manage the directory structure of a local project.

This class locates the project directory dynamically by searching in candidate base paths 
(e.g., OneDrive folders or custom locations) and provides convenient methods to access 
and create standard subdirectories like datasets, results, and figures.

It no longer handles adding GitHub subdirectories to sys.path: use submodules with 
`add_submodule_to_path` for that purpose.
"""

import getpass
from pathlib import Path
from typing import Optional, Tuple, Union

class ProjectPaths:
    """
    Manage the directory structure of a local project with standardized subfolders.

    Parameters
    ----------
    project : str
        Name of the project folder inside one of the base search paths.
    search_paths : list of str or Path, optional
        List of base directories to search for the project folder.
        If None, default OneDrive paths are used.

    Attributes
    ----------
    project : str
        Project name.
    base_path : Path
        Absolute path to the root of the project.
    datasets_root : Path
        Path to the '01_Datasets' directory inside the project.
    results_root : Path
        Path to the '02_Results' directory inside the project.
    figures_root : Path
        Path to the '03_Figures' directory inside the project.
    user : str
        Current system username.
    """

    def __init__(self, project: str, search_paths: Optional[list] = None):
        self.project = project
        self.user = getpass.getuser()
        self.base_path = self._find_project_path(search_paths)
        
        # Initialize root folders
        self.datasets_root = self.base_path / "01_Datasets"
        self.results_root = self.base_path / "02_Results"
        self.figures_root = self.base_path / "03_Figures"

    def _find_project_path(self, search_paths: Optional[list]) -> Path:
        """Locate the project folder in the given search paths."""
        default_paths = [
            Path(fr"C:\Users\{self.user}\OneDrive - Università degli Studi di Bari (1)\Projects"),
            Path(fr"C:\Users\{self.user}\OneDrive - Università degli Studi di Bari\Projects")
        ]

        candidate_paths = [Path(p) for p in search_paths] if search_paths else default_paths

        for base in candidate_paths:
            project_path = base / self.project
            if project_path.exists():
                print(f"[INFO] Working directory set to: {project_path}")
                return project_path

        raise FileNotFoundError(
            f"[ERROR] Project '{self.project}' not found in provided search paths for user '{self.user}'."
        )

    def _get_subfolder_path(
        self,
        root: Path,
        *path_parts: str,
        subfolder: Optional[str] = None,
        create: bool = True
    ) -> Path:
        """Builds and optionally creates a nested path inside a given root."""
        path = root.joinpath(*path_parts)
        if subfolder:
            path = path / subfolder
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def get_datasets_path(
        self, *path_parts: str, processed: Union[bool, str] = False
    ) -> Union[Path, Tuple[Path, Path]]:
        """Return paths inside '01_Datasets'."""
        raw_path = self._get_subfolder_path(self.datasets_root, "raw", *path_parts)
        proc_path = self._get_subfolder_path(self.datasets_root, "processed", *path_parts)

        if processed == "both":
            print(f"[INFO] Dataset folder (raw): {raw_path}")
            print(f"[INFO] Dataset folder (processed): {proc_path}")
            return raw_path, proc_path

        selected = proc_path if processed else raw_path
        label = "processed" if processed else "raw"
        print(f"[INFO] Dataset folder ({label}): {selected}")
        return selected

    def get_results_path(
        self, *path_parts: str, plots: bool = False
    ) -> Tuple[Path, Optional[Path]]:
        """Return a path inside '02_Results', optionally with a 'plots' subfolder."""
        base_path = self._get_subfolder_path(self.results_root, *path_parts)
        plots_path = None
        if plots:
            plots_path = self._get_subfolder_path(base_path, subfolder="plots")

        print(f"[INFO] Results folder: {base_path}")
        if plots_path:
            print(f"[INFO] Plots folder: {plots_path}")

        return base_path, plots_path

    def get_figures_path(self, *path_parts: str) -> Path:
        """Return a path inside '03_Figures'."""
        path = self._get_subfolder_path(self.figures_root, *path_parts)
        print(f"[INFO] Figures folder: {path}")
        return path
