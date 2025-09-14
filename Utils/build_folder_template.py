# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 09:47:51 2025

@author: mik16
"""

from pathlib import Path
import argparse


class ProjectStructure:
    """
    Create and manage a standard project folder structure.

    The structure includes a main project folder, top-level folders
    with `README.txt` files, and predefined subfolders.

    Attributes
    ----------
    project_name : str
        Name of the main project folder.
    destination : Path
        Path where the project folder will be created.
    project_path : Path
        Full path to the created project folder.
    """

    # Top-level folders and their descriptions
    TOP_LEVEL_FOLDERS = {
        "01_Datasets": """Contains the datasets used in the project.
- raw/: original datasets received or downloaded
- processed/: transformed or generated datasets""",

        "02_Results": "Results of the experiments: outputs, metrics, saved models, etc.",

        "03_Figures": "Figures useful for communication: diagrams, figures for articles or presentations.",

        "04_Documents": """Project documentation:
- 01_References/: third-party material (slides, articles, papers)
- 02_Presentations/: presentations created by you
- 03_Reports/: technical reports
- 04_Bureaucracy/: bureaucratic material
- 05_Notes/: internal notes or meeting notes"""
    }

    # Subfolders to be created (without README)
    ALL_SUBFOLDERS = [
        "01_Datasets/raw",
        "01_Datasets/processed",
        "04_Documents/01_References",
        "04_Documents/02_Presentations",
        "04_Documents/03_Reports",
        "04_Documents/04_Bureaucracy",
        "04_Documents/05_Notes"
    ]

    def __init__(self, project_name: str, destination: Path):
        """
        Initialize the ProjectStructure object.

        Parameters
        ----------
        project_name : str
            Name of the main project folder.
        destination : Path
            Path where the project folder will be created.
        """
        self.project_name = project_name
        self.destination = Path(destination)
        self.project_path = self.destination / self.project_name

    def create(self):
        """
        Create the project folder structure.

        This includes:
        - the main project folder,
        - top-level folders with a `README.txt` describing their purpose,
        - predefined subfolders without README files.

        Returns
        -------
        Path
            The path to the created project folder.
        """
        self.project_path.mkdir(parents=True, exist_ok=True)

        # Create all top-level folders with README.txt
        for folder, description in self.TOP_LEVEL_FOLDERS.items():
            folder_path = self.project_path / folder
            folder_path.mkdir(parents=True, exist_ok=True)
            readme_file = folder_path / "README.txt"
            readme_file.write_text(description.strip() + "\n", encoding="utf-8")

        # Create all subfolders without README.txt
        for subfolder in self.ALL_SUBFOLDERS:
            sub_path = self.project_path / subfolder
            sub_path.mkdir(parents=True, exist_ok=True)

        print(f"Structure created in: {self.project_path.resolve()}!")
        return self.project_path

    @staticmethod
    def parse_args():
        """
        Parse command line arguments.

        Returns
        -------
        argparse.Namespace
            Parsed arguments with:
            - project_name: name of the project folder
            - destination: path where the project structure will be created
        """
        parser = argparse.ArgumentParser(description="Create a standard structure for a new project.")
        parser.add_argument("project_name", help="Name of the project (used as the main folder name)")
        parser.add_argument("destination", help="Path where the project folder will be created")
        return parser.parse_args()


# Example
project_name = input("Insert project name: ")
destination = Path(r"C:\Users\mik16\OneDrive - Università degli Studi di Bari (1)\Projects")
ps = ProjectStructure(project_name, destination)
ps.create()

