#metrics
#./logs/mnist_rotate_{degrees}_results/test_metrics.yamls
#yaml configured like this
#ece: 0.000000
#mean_cce: 0.000000
#test_acc: 0.000000
#for degress, [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
#construct a plot of ece, mean_cce, test_acc compared to each other for each degree
#plotting

import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def plot_metrics():
    degrees = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
    ece = []
    mean_cce = []
    test_acc = []
    for degree in degrees:
        with open(f"./logs/mnist_rotate_{degree}_results/test_metrics.yaml", "r") as f:
            data = yaml.safe_load(f)
            ece.append(data["ece"])
            mean_cce.append(data["mean_cce"])
            test_acc.append(data["test_acc"])
    fig, ax = plt.subplots(3, 1, figsize=(10, 6))
    ax[0].plot(degrees, ece, label="ECE")
    ax[0].set_title("ECE")
    ax[1].plot(degrees, mean_cce, label="Mean CCE")
    ax[1].set_title("Mean CCE")
    ax[2].plot(degrees, test_acc, label="Test Accuracy")
    ax[2].set_title("Test Accuracy")
    plt.show()
    fig.savefig("metrics.png")

def table_metrics():
    degrees = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
    ece = []
    mean_cce = []
    test_acc = []
    for degree in degrees:
        with open(f"./logs/mnist_rotate_{degree}_results/test_metrics.yaml", "r") as f:
            data = yaml.safe_load(f)
            ece.append(data["ece"])
            mean_cce.append(data["mean_cce"])
            test_acc.append(data["test_acc"])
    df = pd.DataFrame({"Degrees": degrees, "ECE": ece, "Mean CCE": mean_cce, "Test Accuracy": test_acc})
    print(df)
    df.to_csv("metrics.csv", index=False)


if __name__ == "__main__":
    print("Plotting metrics")
    plot_metrics()
    print("Creating table")
    table_metrics()
