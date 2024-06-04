import pandas as pd
import os
import glob


def process_files(directory):
    # Dictionary to store the results
    results = []

    # List all TSV files in the directory
    files = glob.glob(os.path.join(directory, "pr*.tsv"))

    for file in files:
        # Extract instance name and path number from the file name
        base_name = os.path.basename(file)
        parts = base_name.split("_")
        print(base_name)
        print(parts)
        instance_name = parts[0]
        parts[1] = parts[1][1:]
        criteria_function = "_".join(parts[1:4])
        paths_count = parts[4][1:]
        solutions_count = parts[5][1:]
        enable_random_noise = parts[6].split(".")[0] == "True"

        # Read the TSV file into a DataFrame
        df = pd.read_csv(file, sep="\t")

        # Calculate the required statistics
        avg_gap = df["gap"].mean()
        max_gap = df["gap"].max()
        min_gap = df["gap"].min()

        avg_execution_time = df["execution_time"].mean()
        max_execution_time = df["execution_time"].max()
        min_execution_time = df["execution_time"].min()

        # Append the results to the list
        results.append(
            {
                "instance_name": instance_name,
                "criteria_function": criteria_function,
                "paths_count": int(paths_count),
                "average_gap": avg_gap,
                "minimum_gap": min_gap,
                "maximum_gap": max_gap,
                "average_execution_time": avg_execution_time,
                "minimum_execution_time": min_execution_time,
                "maximum_execution_time": max_execution_time,
            }
        )

    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results)

    # Save the DataFrame to a TSV file
    results_df.to_csv("aggregated_solutions.tsv", sep="\t", index=False)
    print(results_df)


def main():
    directory = r"C:\Users\jmgarzonv\Downloads\TOPTW\TOPTW-solver\results\ITERATED_LOCAL_SEARCH"  # Replace with the path to your TSV files
    process_files(directory)


if __name__ == "__main__":
    main()
