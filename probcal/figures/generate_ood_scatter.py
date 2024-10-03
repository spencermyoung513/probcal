import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


fpath = "~/Downloads/ood_results.csv"
cols = ["MCMD Run 1", "MCMD Run 2", "MCMD Run 3", "MCMD Run 4", "MCMD Run 5"]

df = pd.read_csv(fpath)
df = pd.melt(
    df,
    id_vars=["Dataset", "Head", "OOD type", "Perturbation"],
    value_vars=cols,
    var_name="Run",
    value_name="MCMD",
)


# Plot the results
sns.set(style="whitegrid")
# Create a custom color palette
palette = sns.color_palette("deep")
plt.figure(figsize=(6, 6))
sns.scatterplot(
    data=df,
    x="Perturbation",
    y="MCMD",
    hue="Head",
    style="Head",
    markers="o",
    alpha=0.7,
    palette=palette,
)
plt.title("MCMD Sensitivity - OOD to P(X) on COCO-People")
plt.show()
