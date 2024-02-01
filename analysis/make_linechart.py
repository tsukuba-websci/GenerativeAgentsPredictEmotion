import os

import matplotlib.pyplot as plt
import pandas as pd


def make_linechart(panas, panas_no_norm, title, output_file):
    plt.plot(x, panas[0], label="P", color="green")
    plt.plot(x, panas[1], label="N", color="red")
    plt.plot(
        x,
        panas_no_norm[0],
        label="P (without norm)",
        color="green",
        linestyle="dashed",
    )
    plt.plot(
        x,
        panas_no_norm[1],
        label="N (without norm)",
        color="red",
        linestyle="dashed",
    )

    plt.title(title)
    plt.xlabel("Scene")
    plt.ylabel("Score")
    plt.legend()
    plt.xticks(x)
    plt.yticks([i for i in range(10, 51, 10)])
    plt.savefig(output_file)
    plt.close()


def extract_scores(df, id):
    return df[df["situation_id"] == id].values[0][1:]


if __name__ == "__main__":
    df = pd.read_csv("data/source/emotion_bench/text/processed/situations.csv", delimiter="|")

    positive = pd.read_csv("results/positive_scores.csv")
    negative = pd.read_csv("results/negative_scores.csv")

    positive_no_norm = pd.read_csv("results/positive_scores_no_norm.csv")
    negative_no_norm = pd.read_csv("results/negative_scores_no_norm.csv")

    x = [i for i in range(1, 6)]

    os.makedirs("results/visualize", exist_ok=True)

    factor_prior = ""
    count = 1

    panas_factor = []
    panas_no_norm_factor = []

    for id, factor in zip(df["ID"], df["Factor"]):
        panas = [extract_scores(positive, id), extract_scores(negative, id)]
        panas_no_norm = [
            extract_scores(positive_no_norm, id),
            extract_scores(negative_no_norm, id),
        ]

        panas[0][0] = panas_no_norm[0][0]
        panas[1][0] = panas_no_norm[1][0]

        if factor != factor_prior:
            count = 1
            factor_prior = factor
            panas_factor = []
            panas_no_norm_factor = []
        else:
            count += 1
        panas_factor.append(panas)
        panas_no_norm_factor.append(panas_no_norm)

        title = f"{factor} {count}"
        output_file = f"results/visualize/{factor}-{count}.png"
        make_linechart(panas, panas_no_norm, title, output_file)
