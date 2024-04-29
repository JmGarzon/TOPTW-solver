"""
This file contains the implementation of the Solutions class, which represents a collection of solutions for a problem instance.
The Solutions class provides methods for computing statistics, converting to a dictionary representation, and saving to a TSV file.

Made by: Jose Garzon, April 2024
Heuristic Methods for Optimization
Universidad EAFIT
"""

import os
import numpy as np
import logging


class Solutions:
    def __init__(
        self,
        nodes_count,
        T_max,
        criteria,
        paths_count,
        solutions_count,
        enable_random_noise,
        optimal_score,
        solutions_df,
        instance_name,
    ):
        """
        Initialize a Solutions object.

        Parameters:
        - criteria: The criteria function used in the constructive method.
        - nodes_count: The total number of nodes in the problem instance.
        - T_max: The maximum time limit for the solution.
        - paths_count: Number of paths used in the constructive method.
        - solutions_count: Number of solutions generated.
        - enable_random_noise: Whether random noise was enabled in the constructive method.
        - optimal_score: The optimal score for the problem instance.
        - solutions_df: DataFrame containing the constructed solutions.
        - instance_name: Name of the problem instance.
        """
        self.criteria = criteria
        self.nodes_count = nodes_count
        self.T_max = T_max
        self.paths_count = paths_count
        self.solutions_count = solutions_count
        self.enable_random_noise = enable_random_noise
        self.instance_name = os.path.splitext(instance_name)[0]
        self.solutions_df = solutions_df
        self.compute_stats()

    def compute_stats(self):
        """
        Compute statistics for GAP and execution time.
        """
        if "gap" in self.solutions_df.columns:
            self.avg_gap = np.mean(self.solutions_df["gap"])
            self.min_gap = np.min(self.solutions_df["gap"])
            self.max_gap = np.max(self.solutions_df["gap"])
        else:
            raise ValueError("DataFrame does not contain 'gap' column.")

        if "execution_time" in self.solutions_df.columns:
            self.avg_time = np.mean(self.solutions_df["execution_time"])
            self.min_time = np.min(self.solutions_df["execution_time"])
            self.max_time = np.max(self.solutions_df["execution_time"])
        else:
            raise ValueError("DataFrame does not contain 'execution_time' column.")

    def to_dict(self):
        """
        Return a dictionary representation of the Solutions object.
        """
        data = {
            "instance_name": self.instance_name,
            "nodes_count": self.nodes_count,
            "T_max": self.T_max,
            "criteria_function": self.criteria.__name__,
            "paths_count": self.paths_count,
            "solutions_count": self.solutions_count,
            "enable_random_noise": self.enable_random_noise,
            "average_gap": self.avg_gap,
            "minimum_gap": self.min_gap,
            "maximum_gap": self.max_gap,
            "average_execution_time": self.avg_time,
            "minimum_execution_time": self.min_time,
            "maximum_execution_time": self.max_time,
        }
        return data

    def to_tsv(self, folder_path):
        """
        Save solutions DataFrame to a TSV file.

        Parameters:
        - folder_path: Path to the folder where the TSV file will be saved.
        """
        # Create folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Check if naming convention file exists, and create it if it doesn't
        naming_convention_file = os.path.join(folder_path, "naming_convention.txt")
        if not os.path.exists(naming_convention_file):
            naming_explanation = (
                "File naming convention:\n"
                "<InstanceName>_c<CriteriaFunction>_p<PathsCount>_m<SolutionsCount>_r<RandomNoise>.tsv\n"
                "Example: pr01_simple_revenue_2_10_True.tsv"
            )
            with open(naming_convention_file, "w") as f:
                f.write(naming_explanation)

        # Generate file name based on instance name and solver parameters
        file_name = f"{self.instance_name}_c{self.criteria.__name__}_p{self.paths_count}_m{self.solutions_count}_r{self.enable_random_noise}.tsv"
        file_path = os.path.join(folder_path, file_name)

        # Save DataFrame to TSV
        self.solutions_df.to_csv(file_path, sep="\t", index=False)
        logging.info(f"\t - TSV file saved at: {file_path}")
