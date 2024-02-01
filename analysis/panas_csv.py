import argparse
import csv
import os
import re

import dotenv
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm

dotenv.load_dotenv(".env")


def panas_result_to_dict(text):
    lines = text.split("\n")
    emotions_dict = {}

    for line in lines:
        line = re.sub(r"^\d+\.\s*", "", line)
        try:
            key, value = line.split(": ")
            emotions_dict[key] = int(value)
        except ValueError:
            print(f"ValueError: {line}")
            pass

    return emotions_dict


def sum_emotions_by_category(emotions_dict):
    positive_emotions = {
        "Attentive",
        "Active",
        "Alert",
        "Excited",
        "Enthusiastic",
        "Determined",
        "Inspired",
        "Proud",
        "Interested",
        "Strong",
    }
    negative_emotions = {
        "Hostile",
        "Irritable",
        "Ashamed",
        "Guilty",
        "Distressed",
        "Upset",
        "Scared",
        "Afraid",
        "Jittery",
        "Nervous",
    }

    positive_sum = 0
    negative_sum = 0

    for emotion, value in emotions_dict.items():
        if emotion in positive_emotions:
            positive_sum += value
        elif emotion in negative_emotions:
            negative_sum += value

    return positive_sum, negative_sum


def write_to_csv(output_file, situation_id, scores):
    with open(output_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                situation_id,
                scores[0],
                scores[1],
                scores[2],
                scores[3],
                scores[4],
            ]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--norm",
        type=str,
        choices=["norm", "no_norm"],
        default="norm",
        help="norm or no_norm",
    )
    norm = parser.parse_args().norm

    os.makedirs("results", exist_ok=True)

    if norm == "norm":
        experiment_id_suffix = ""
        positive_output_file = "results/positive_scores.csv"
        negative_output_file = "results/negative_scores.csv"
    else:
        experiment_id_suffix = "_no_norm"
        positive_output_file = "results/positive_scores_no_norm.csv"
        negative_output_file = "results/negative_scores_no_norm.csv"

    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    scene_len = 5
    num = 2

    for file in [positive_output_file, negative_output_file]:
        with open(file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["situation_id", "1", "2", "3", "4", "5"])

    df = pd.read_csv("data/processed/situations.csv", delimiter="|")

    for situation_id, factor in tqdm(zip(df["ID"], df["Factor"])):
        positive_scores = [0 for _ in range(scene_len)]
        negative_scores = [0 for _ in range(scene_len)]

        for iter in range(num):
            experiment_id = f"sit-{situation_id}-run-{iter}_for_arxiv{experiment_id_suffix}"
            with neo4j_driver.session() as session:
                result = session.run(
                    """
                    MATCH (m:Memory {experiment_id: $experiment_id})
                    RETURN m.panas_scores, m.memory_number
                    ORDER BY m.memory_number
                """,
                    experiment_id=experiment_id,
                )

                for record in result:
                    panas_result = panas_result_to_dict(record["m.panas_scores"])
                    positive_score, negative_score = sum_emotions_by_category(panas_result)

                    positive_scores[record["m.memory_number"] - 1] += positive_score
                    negative_scores[record["m.memory_number"] - 1] += negative_score

        positive_scores_ave = [score / num for score in positive_scores]
        negative_scores_ave = [score / num for score in negative_scores]

        write_to_csv(positive_output_file, situation_id, positive_scores_ave)
        write_to_csv(negative_output_file, situation_id, negative_scores_ave)

    neo4j_driver.close()
