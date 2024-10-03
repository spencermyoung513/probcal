import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUTPUT_DIR = os.path.join("probcal", "figures", "artifacts")


def main(fpath: str) -> None:
    """
    Load the results from the CSV file and plot the MCMD sensitivity to OOD perturbations.

    Args:
        fpath (str): The path to the CSV file containing the results.

    Returns:
        None: This function does not return a value but plots the results.
    """

    # Load the results from the CSV file
    cols = ["MCMD Run 1", "MCMD Run 2", "MCMD Run 3", "MCMD Run 4", "MCMD Run 5"]
    df = pd.read_csv(fpath)
    df = pd.melt(
        df,
        id_vars=["Dataset", "Head", "OOD type", "Perturbation"],
        value_vars=cols,
        var_name="Run",
        value_name="MCMD",
    )

    for ood_type, dta in df.groupby("OOD type"):
        # Plot the results
        sns.set(style="whitegrid")
        # Create a custom color palette
        palette = sns.color_palette("deep")
        plt.figure(figsize=(6, 6))
        sns.scatterplot(
            data=dta,
            x="Perturbation",
            y="MCMD",
            hue="Head",
            style="Head",
            markers="o",
            alpha=0.7,
            palette=palette,
        )
        plt.savefig(os.path.join(OUTPUT_DIR, f"ood_scatter_{ood_type}.pdf"))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--csv-fpath", type=str)
    args = args.parse_args()

    main(args.csv_fpath)
