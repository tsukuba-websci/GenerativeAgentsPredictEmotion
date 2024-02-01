import sys

sys.path.append(".")
import datetime

import dotenv
import pandas as pd
from tqdm import tqdm

from libs.agent import CaptioningAgent

dotenv.load_dotenv(".env")

if __name__ == "__main__":
    n = 10
    saliency_weight = 1.0
    reinforcement_weight = 0.0
    recency_weight = 1.0
    relevance_weight = 1.0
    date = datetime.datetime.now().strftime("%Y-%m-%d")

    df = pd.read_csv("data/source/emotion_bench/text/processed/situations.csv", delimiter="|")

    # Run with a limited number of situations
    # df = df.head(3)

    for iter in range(n):
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            situation = row["Situation"]
            situation_id = row["ID"]
            episodic_memories = row["Memories"].split("~")
            print(f"Situation: {situation}")

            experiment_id = f"sit-{situation_id}-run-{iter}_for_arxiv"
            # experiment_id = f"sit-{situation_id}-run-{iter}_for_arxiv_no_norm"

            agent1 = CaptioningAgent(
                name=experiment_id,
                situation=situation,
                saliency_weight=saliency_weight,
                reinforcement_weight=reinforcement_weight,
                recency_weight=recency_weight,
                relevance_weight=relevance_weight,
                delete=True,
            )
            agent1.run_text_no_retrieval(memories=episodic_memories)
            # agent1.run_text_no_norm(memories=episodic_memories)
            # agent1.run_text(memories=episodic_memories)
