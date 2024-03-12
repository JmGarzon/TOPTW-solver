import os
import pandas as pd

def read_instances(folder_path):
    instances = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            instance_data = {'filename': filename}
            with open(os.path.join(folder_path, filename), 'r') as file:
                lines = file.readlines()
                _, instance_data['max_paths'], instance_data['N'], _ = map(int, lines[0].strip().split())
                _, _ = map(int, lines[1].strip().split())
                points_data  = []
                instance_data['points'] = []
                for line in lines[2:]:
                    if line.strip():
                        values = line.strip().split()
                        i, x, y, d, S, f, a, *rest = map(float, values)
                        O, C = map(float, rest[-2:])
                        points_data .append({'i': int(i), 'x': x, 'y': y, 'duration': d, 'profit': S, 'opening_time': O, 'closing_time': C})
                instance_data['points'] = pd.DataFrame(points_data)        
            instances.append(instance_data)
    return instances

INSTANCES_PATH = "instances\pr01_10"

if __name__ == "__main__":
    instances = read_instances(INSTANCES_PATH)

    for instance in instances:
        print("Filename:", instance['filename'])
        print("max_paths, N:", instance['max_paths'], instance['N'])
        print("Points DataFrame:")
        print(instance['points'])
        print()