import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind


def _transform_df(df: pd.DataFrame) -> pd.DataFrame:
    df["coef_var_cce"] = df["CCE (std)"] / df["Mean CCE"]
    df["coef_var_ece"] = df["ECE (std)"] / df["Mean ECE"]
    df["coef_var_nll"] = df["NLL (std)"] / df["Mean NLL"]
    return df


def _hyp_test(arr1: np.ndarray, arr2: np.ndarray) -> float:
    arr1 = arr1[~np.isnan(arr1)]
    arr2 = arr2[~np.isnan(arr2)]

    _, p_value = ttest_ind(arr1, arr2, equal_var=False, alternative="greater")
    return p_value


def _get_kde_plot(
    arr1: np.ndarray, arr2: np.ndarray, label1: str, label2: str, title: str, output_path: str
):
    sns.kdeplot(arr1, fill=True, label=label1, clip=(0, np.inf))
    sns.kdeplot(arr2, fill=True, label=label2, clip=(0, np.inf))
    plt.legend()
    plt.title(title)
    plt.savefig(output_path)
    plt.close()


def main(boostrap_results_path: str, no_boostrap_results_path: str):

    df_bs = pd.read_csv(boostrap_results_path)
    df_bs = _transform_df(df_bs)

    for g, df_g in df_bs.groupby("Dataset"):

        p_value_cce_ece = _hyp_test(df_g["coef_var_cce"].values, df_g["coef_var_ece"].values)
        p_value_cce_nll = _hyp_test(df_g["coef_var_cce"].values, df_g["coef_var_nll"].values)

        print("Dataset:", g)
        print(f"Mean CCE: {df_g['coef_var_cce'].mean():.3f}")
        print(f"Mean ECE: {df_g['coef_var_ece'].mean():.3f}")
        print(
            f"CCE vs ECE p-value: {p_value_cce_ece:.3f}",
        )

        _get_kde_plot(
            df_g["coef_var_cce"].values,
            df_g["coef_var_ece"].values,
            "CCE",
            "ECE",
            r"CCE vs ECE Coeff. of Variation ($\frac{\sigma}{\mu}$) "
            f"- p value: {p_value_cce_ece:.3f}",
            f"./artifacts/{g}_cce_ece_kde.pdf",
        )

        _get_kde_plot(
            df_g["coef_var_cce"].values,
            df_g["coef_var_nll"].values,
            "CCE",
            "NLL",
            r"CCE vs NLL Coeff. of Variation ($\frac{\sigma}{\mu}$) "
            f"- p value: {p_value_cce_nll:.3f}",
            f"./artifacts/{g}_cce_nll_kde.pdf",
        )


if __name__ == "__main__":

    main(
        boostrap_results_path="./all_results_bootstrap.csv",
        no_boostrap_results_path="./all_results_no_bootstrap.csv",
    )
