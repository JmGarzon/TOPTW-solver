import os
import logging
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
from solutions import Solutions
from toptw_solver import TOPTWSolver

# For the spinner
import threading
import itertools
import time
import sys


INSTANCES_PATH = r"instances\pr01_10"
METHOD_NAME = r"ITERATED_LOCAL_SEARCH"
RESULTS_PATH = r".\results\\" + METHOD_NAME
LOGGING_PATH = r".\logs\\" + METHOD_NAME
LOGGING_LEVEL = logging.INFO

# Parameters for the evaluation
SOLUTIONS_COUNT = 5
PATH_COUNT_LIST = [1, 2, 3, 4]


def configure_logging(name, console=False):
    """
    Configure the logging settings for the TOPTW Solver.

    Args:
        name (str): The name of the current process.
        console (bool, optional): Flag to enable logging to the console. Defaults to False.
    """

    if not os.path.exists(LOGGING_PATH):
        os.makedirs(LOGGING_PATH)
    logfile_path = os.path.join(LOGGING_PATH, f"logfile__{name}.log")

    handlers = [logging.FileHandler(logfile_path, mode="w")]
    if console:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=LOGGING_LEVEL,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logging.info("Starting TOPTW Solver: ILS Method\n\n")


def process_instance(instance):
    """
    Process an instance of the TOPTW problem.

    Args:
        instance (dict): The instance data.

    Returns:
        list: A list of aggregated solutions data.
    """
    instance_name = instance.get("filename")
    configure_logging(instance_name)

    solver = TOPTWSolver(instance)
    logging.info("----------------------------------------")
    logging.info(
        f"Processing: {solver.filename} - N: {solver.nodes_count} - T_max: {solver.Tmax}"
    )
    logging.info("----------------------------------------")
    comparison_parameters = {
        "solutions_count": SOLUTIONS_COUNT,
        "random_noise_flag": True,
        "path_count_list": PATH_COUNT_LIST,
        "criteria_list": [solver.benefit_insertion_ratio],
    }
    aggregated_solutions_data = []
    for path_count in comparison_parameters["path_count_list"]:
        for criteria in comparison_parameters["criteria_list"]:
            logging.info(
                f"* Parameters: solutions_count={comparison_parameters['solutions_count']}, path_count={path_count}, criteria={criteria.__name__}, random_noise={comparison_parameters['random_noise_flag']}"
            )
            solutions = solver.ILS(
                criteria,
                path_count,
                comparison_parameters["solutions_count"],
                comparison_parameters["random_noise_flag"],
            )
            solutions.to_tsv(RESULTS_PATH)
            aggregated_solutions_data.append(solutions.to_dict())

    return aggregated_solutions_data


def save_aggregated_solutions(aggregated_solutions_data):
    """
    Save aggregated solutions data to a TSV file.

    Args:
        aggregated_solutions_data (list): A list of dictionaries containing the aggregated solutions data.
    Returns:
        None

    """
    aggregated_solutions = pd.DataFrame.from_records(aggregated_solutions_data)
    aggregated_solutions_path = os.path.join(RESULTS_PATH, "aggregated_solutions.tsv")
    aggregated_solutions.to_csv(aggregated_solutions_path, sep="\t", index=False)
    logging.info("Aggregated Solutions:")
    logging.info(aggregated_solutions)
    logging.info(f"\t - TSV file saved at: {aggregated_solutions_path}")


def spinner(done_event):
    """
    Display a spinner to indicate activity.

    Args:
        done_event (threading.Event): An event to signal when the activity is complete.
    """
    for c in itertools.cycle(["|", "/", "-", "\\"]):
        if done_event.is_set():
            sys.stdout.flush()
            break
        sys.stdout.write("\rProcessing " + c)
        sys.stdout.flush()
        time.sleep(0.1)


if __name__ == "__main__":
    from load_instances import read_instances

    configure_logging("main", console=True)

    instances = read_instances(INSTANCES_PATH)
    aggregated_solutions_data = []
    count = 0

    logging.info("Assigning instances to workers.")
    logging.info(
        "Please follow the progress of each worker in their respective log files...\n"
    )

    # Start the spinner thread to show activity
    done_event = threading.Event()
    spinner_thread = threading.Thread(target=spinner, args=(done_event,))
    spinner_thread.start()

    try:
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {
                executor.submit(process_instance, instance): instance
                for instance in instances
            }
            count = 0
            for future in as_completed(futures):
                instance = futures[future]
                try:
                    result = future.result()
                    aggregated_solutions_data.append(result[0])
                    count += 1
                except Exception as exc:
                    logging.exception(
                        f"Instance {instance} generated an exception: {exc}"
                    )

        logging.info("----------------------------------------")
        logging.info(f"Total instances processed: {count}")
        logging.info("----------------------------------------")

        save_aggregated_solutions(aggregated_solutions_data)

    except KeyboardInterrupt:
        logging.info("Process interrupted by user.")
    finally:
        done_event.set()  # Signal the spinner thread to stop
        spinner_thread.join()  # Wait for the spinner thread to finish
