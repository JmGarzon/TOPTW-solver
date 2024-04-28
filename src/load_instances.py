import pandas as pd
import os


def read_instances(folder_path):
    """
    Read TOPTW (Time-Dependent Orienteering Problem with Time Windows) test instances from text files in a folder.

    The files are expected to have the following format:

    ************************
    * TOPTW test instances *
    ************************

    The first line of each file contains the following data:
        k v N t
    Where:
        k = not relevant
        v = with this number of paths, all vertices can be visited
        N = number of vertices
        t = not relevant

    The second line contains:
        D Q
    Where:
        D = not relevant (in many files, this number is missing)
        Q = not relevant

    The remaining lines contain the data of each point.
    For each point, the line contains the following data:
        i x y d S f a list O C
    Where:
        i = vertex number
        x = x coordinate
        y = y coordinate
        d = service duration or visiting time
        S = profit of the location
        f = not relevant
        a = not relevant
        list = not relevant (length of the list depends on a)
        O = opening of time window (earliest time for start of service)
        C = closing of time window (latest time for start of service)

    Remarks:
        - The first point (index 0) is the starting AND ending point.
        - The number of paths (P) is not included in the data file. This value can vary (1,2,3, etc.).
        - The time budget per path (Tmax) equals the closing time of the starting point.
        - The Euclidean distance is used and rounded down to the first decimal for the Solomon instances
          and to the second decimal for the instances of Cordeau et al.

    Parameters:
    - folder_path: Path to the folder containing the TOPTW test instance files.

    Returns:
    - instances: A list of dictionaries, each containing information about a TOPTW test instance.
    """
    instances = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            instance_data = {"filename": filename}
            with open(os.path.join(folder_path, filename), "r") as file:
                lines = file.readlines()
                _, instance_data["max_paths"], instance_data["N"], _ = map(
                    int, lines[0].strip().split()
                )
                _, _ = map(int, lines[1].strip().split())
                points_data = []
                instance_data["points"] = []
                for line in lines[2:]:
                    if line.strip():
                        values = line.strip().split()
                        i, x, y, d, S, f, a, *rest = map(float, values)
                        O, C = map(float, rest[-2:])
                        points_data.append(
                            {
                                "i": int(i),
                                "x": x,
                                "y": y,
                                "duration": d,
                                "profit": S,
                                "opening_time": O,
                                "closing_time": C,
                            }
                        )
                instance_data["points"] = pd.DataFrame(points_data).set_index("i")

            instances.append(instance_data)
    return instances
