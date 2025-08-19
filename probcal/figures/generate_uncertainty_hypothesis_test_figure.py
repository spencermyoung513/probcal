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


def _hyp_test(arr1: np.ndarray, arr2: np.ndarray, alternative: str = "two-sided") -> float:
    """
    Performs a hypothesis test (t-test) on two arrays of data to determine if there is a significant difference between them.

    Args:
        arr1 (np.ndarray): The first array of data.
        arr2 (np.ndarray): The second array of data.
        alternative (str): Defines the alternative hypothesis. Options are 'two-sided', 'less', or 'greater'.

    Returns:
        float: The p-value from the t-test, indicating the probability of observing the data given that the null hypothesis is true.
    """

    arr1 = arr1[~np.isnan(arr1)]
    arr2 = arr2[~np.isnan(arr2)]

    _, p_value = ttest_ind(arr1, arr2, equal_var=False, alternative=alternative)
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


def main(bootstrap_results_path: str):

    df_bs = pd.read_csv(bootstrap_results_path)
    df_bs = _transform_df(df_bs)

    p_value_cce_ece = _hyp_test(df_bs["coef_var_cce"].values, df_bs["coef_var_ece"].values)
    p_value_cce_nll = _hyp_test(df_bs["coef_var_cce"].values, df_bs["coef_var_nll"].values)

    p_value_cce_ece_greater = _hyp_test(
        df_bs["coef_var_cce"].values, df_bs["coef_var_ece"].values, alternative="greater"
    )

    _get_kde_plot(
        df_bs["coef_var_cce"].values,
        df_bs["coef_var_ece"].values,
        "CCE",
        "ECE",
        r"CCE vs ECE Coeff. of Variation ($\frac{\sigma}{\mu}$) "
        f"- p value: {p_value_cce_ece:.3f}",
        "./artifacts/cce_ece_kde.pdf",
    )

    _get_kde_plot(
        df_bs["coef_var_cce"].values,
        df_bs["coef_var_nll"].values,
        "CCE",
        "NLL",
        r"CCE vs NLL Coeff. of Variation ($\frac{\sigma}{\mu}$) "
        f"- p value: {p_value_cce_nll:.3f}",
        "./artifacts/cce_nll_kde.pdf",
    )

    print("Overall: ")
    print(f"Mean CCE: {df_bs['coef_var_cce'].mean():.3f}")
    print(f"Mean ECE: {df_bs['coef_var_ece'].mean():.3f}")
    print(
        f"CCE vs ECE p-value (two-sided): {p_value_cce_ece:.3f}",
    )
    print(
        f"CCE vs ECE p-value (greater): {p_value_cce_ece_greater:.3f}",
    )

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
        bootstrap_results_path="./all_results_bootstrap.csv",
    )
