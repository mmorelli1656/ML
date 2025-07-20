# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 14:47:01 2025

@author: WKS
"""

#%% Libreries

import os
import sys
from pathlib import Path


#%% Functions

class ProjectPaths:
    def __init__(self, project, append_subdirs=None):
        self.project = project
        self.base_path = self._set_project_directory()
        self.image_root = self.base_path / "03_Images"
        self.results_root = self.base_path / "04_Results"
        self._append_to_syspath(append_subdirs)

    def _set_project_directory(self):
        possible_users = ["mik16", "WKS"]
        for user in possible_users:
            one_drive_paths = [
                Path(fr"C:\Users\{user}\OneDrive - Università degli Studi di Bari (1)\Projects"),
                Path(fr"C:\Users\{user}\OneDrive - Università degli Studi di Bari\Projects")
            ]
            for base in one_drive_paths:
                project_path = base / self.project
                if project_path.exists():
                    os.chdir(project_path)
                    print(f"Directory di lavoro impostata su: {project_path}")
                    return project_path

        raise FileNotFoundError(
            f"Nessuna directory valida trovata per il progetto '{self.project}' in OneDrive."
        )

    def _append_to_syspath(self, subdirs):
        if not subdirs:
            return

        possible_users = ["mik16", "WKS"]
        for user in possible_users:
            github_root = Path(fr"C:\Users\{user}\Github")
            if github_root.exists():
                for sub in subdirs:
                    full_path = github_root / sub
                    if full_path.exists():
                        sys.path.append(str(full_path))
                        print(f"Aggiunto al sys.path: {full_path}")
                break
        else:
            print("Nessun path aggiunto: directory 'Github' non trovata per utenti conosciuti.")

    def get_save_paths(self, *path_parts):
        image_path = self.image_root.joinpath(*path_parts)
        results_path = self.results_root.joinpath(*path_parts)

        image_path.mkdir(parents=True, exist_ok=True)
        results_path.mkdir(parents=True, exist_ok=True)

        print(f"Cartella immagini: {image_path}")
        print(f"Cartella risultati: {results_path}")

        return image_path, results_path
    
    
#%% Example

# Imposta la directory di lavoro
paths = ProjectPaths("Rete", append_subdirs=None)

# Regione di interesse
region = "Campania"

# Imposta la directory delle immagini e dei risultati
img_path, res_path = paths.get_save_paths(region)