import argparse

from probcal.evaluation.calibration_evaluator import CalibrationResults


def append_to_file(filename, number):
    with open(filename, "a") as file:
        file.write(f"{number}\n")


def main(results_path, data_path):
    results = CalibrationResults.load(results_path)
    append_to_file(data_path, results.mean_mcmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run calibration evaluation with config file.")
    parser.add_argument("--results", required=True, help="Path to the results file")
    parser.add_argument("--data", required=True, help="Path to the data file")
    args = parser.parse_args()

    main(args.results, args.data)
