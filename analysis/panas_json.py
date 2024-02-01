import argparse
import json
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
            pass

    return emotions_dict


def compute_emotion_stats(panas_dicts):
    results = {}
    for panas in panas_dicts.values():
        for key, value in panas.items():
            results.setdefault(key, []).append(value)

    stats = {
        key: {"average": sum(values) / len(values), "max": max(values), "min": min(values)}
        for key, values in results.items()
    }
    return stats


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

    if norm == "norm":
        experiment_id_suffix = ""
        output_file = "results/panas.json"
    else:
        experiment_id_suffix = "_no_norm"
        output_file = "results/panas_no_norm.json"

    memory_content = []
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    df = pd.read_csv("data/source/emotion_bench/text/processed/situations.csv", delimiter="|")

    num = 10

    all_results_list = []
    for iter in tqdm(range(num)):
        situation_results_dict = {}

        for situation_id in tqdm(df["ID"], leave=False):
            panas_dicts = {}
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
                    panas_dict = panas_result_to_dict(record["m.panas_scores"])
                    panas_dicts.setdefault(record["m.memory_number"], {}).update(panas_dict)

            situation_results_dict[situation_id] = compute_emotion_stats(panas_dicts)

        all_results_list.append(situation_results_dict)

    neo4j_driver.close()

    with open(output_file, "w") as f:
        json.dump(all_results_list, f, indent=4)
