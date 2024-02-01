import sys

sys.path.append(".")

import json
import re

from libs.agent import CaptioningAgent


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


if __name__ == "__main__":
    output_file = "results/default_panas.json"
    results = []

    for iter in range(50):
        experiment_id = f"default-run-{iter}"

        agent1 = CaptioningAgent(
            name=experiment_id,
            situation="",
            saliency_weight=1.0,
            reinforcement_weight=1.0,
            recency_weight=1.1,
            relevance_weight=1.0,
            delete=True,
        )

        panas_scores = ""
        while not agent1.check_panas_format(panas_scores):
            panas_scores = agent1.chain_panas(prompt=agent1.panas_test_template).run(
                scenario="", emotions=agent1.shuffle_emotions()
            )

        emotions_dict = panas_result_to_dict(panas_scores)
        results.append(emotions_dict)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
