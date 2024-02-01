import csv
import json
import os

import dotenv
import pandas as pd
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from tqdm import tqdm

dotenv.load_dotenv(".env")


def chain(llm, prompt: PromptTemplate) -> LLMChain:
    return LLMChain(llm=llm, prompt=prompt)


def generate_episodic_memories(situation: str, emotion: str, llm: ChatOpenAI):
    # Prompt 1: Create 5 basic scenes
    prompt_1_template = PromptTemplate.from_template(
        """
        Task:\n
        Your task is turn a situation into a story of 5 parts. Each part should be an episodic memory of the protagnonist. Each scene should tell a part of the story in a truely neutral and objective way - do not appeal to emotions or use any emotional words.\n

        Example 1:\n
        Input:\n
        Situation: You missed your flight and you are stuck at the airport.\n
        Feeling: Annoyed.\n
        1: I wake up late, 2: I arrive at the airport and there is a long line at security, 3: I get to the gate and I realise the plane is gone, 4: I go to the customer service desk and they tell me that I have to wait until tomorrow, 5: I go to the hotel and get a room for the night.\n

        Guidelines:\n
        1: You must return a JSON in the format of number:story.\n
        2: The parts of the story should follow logically.\n
        3: Do not use emotional words to describe the story. \n
        4. It must be in the first person. You are unaware of other peoples experiences. \n

        Input:\n
        Situation: {situation}\n
        Emotion: {emotion}
        """
    )

    # Prompt 2: Develop the scennes into a story
    prompt_2_template = PromptTemplate.from_template(
        """
        Task:\n
        Your task is to turn a five part series of scenes into a five part story. Each part should be an episodic memory of the protagnonist. The scenes should expand upon the input but should be neutral and should not use emotional words. The scene should not appeal to emotions.\n

        Example 1:\n
        Input:\n
        1: I wake up late, 2: I arrive at the airport and there is a long line at security, 3: I get to the gate and I realise the plane is gone, 4: I go to the customer service desk and they tell me that I have to wait until tomorrow, 5: I go to the hotel and get a room for the night.
        Feeling: Annoyed.\n
        Output:\n
        1: My alarm clock goes off, and I pressed the snooze button because I wanted more sleep. It goes off again after 15 minutes and I realised that I am already running late for my flight, 2: I rush through the doors of the airport and look around to try and find security. I see a sign and I rush in that direction. I notice that there is already a long line at security. 3: After going through security I need to look at my boarding pass to find which gate I have to go to. After looking through all my belongings I find the boarding pass and run to the gate. Im unsure if there will be enough time. When I arrive at the gate there is nobody there. 4: I approach airport staff and they inform me that everyone has already boarded and that it is too late for me to get on the flight. Boarding closed 5 minutes ago. They tell me I should to a hotel for the night and get a flight tomorrow instead. 5: I search online for a hotel. They all seem very expensive near the airport. I find a reasonably priced hotel and go there. I spend the night in the hotel.

        Guidelines:\n
        1: You must return a JSON in the format of number:story.\n
        2: Do not use emotional words to describe the story.\n
        3. It must be in the first person. You are unaware of other peoples experiences.\n
        Input:\n
        {scenes}\n
        Emotion: {emotion}
        """
    )

    scenes = (
        chain(llm=llm, prompt=prompt_1_template)
        .run(situation=situation, emotion=emotion)
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    episodic_memories = (
        chain(llm=llm, prompt=prompt_2_template)
        .run(scenes=scenes, emotion=emotion)
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:
        # Try to parse the generated JSON
        episodic_memories_json = json.loads(episodic_memories)
    except json.JSONDecodeError as e:
        # Handle JSON parsing errors
        print(f"JSON parsing error: {e}")
        episodic_memories_json = None

    return episodic_memories_json


if __name__ == "__main__":
    llm = ChatOpenAI(
        model_name="gpt-4-1106-preview",
        max_tokens=1500,
        request_timeout=120,
        max_retries=20,
    )

    # # Load the sitations
    df = pd.read_csv("data/raw/situations.csv")
    # Remove the factors themselves
    df.drop(0, inplace=True)

    df = df.melt(var_name="Emotion", value_name="Situation")
    df.dropna(inplace=True)
    df["Factor"] = df["Emotion"]
    df["Emotion"] = df["Emotion"].str.replace(r"-\d+", "", regex=True)

    # Reset the index
    df.reset_index(drop=True, inplace=True)
    df.index.rename("ID", inplace=True)

    output_file = "data/processed/situations.csv"
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    factor_counts = 0
    factor_prior = ""
    results = []

    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter="|")
        writer.writerow(["ID", "Emotion", "Factor", "Situation", "Memories"])

        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            situation = row["Situation"]
            situation_id = index
            emotion = row["Emotion"]
            factor = row["Factor"]

            if factor == factor_prior:
                factor_counts += 1
            else:
                writer.writerows(results)
                results = []

                factor_counts = 0
                factor_prior = factor

            if factor_counts < 5:
                episodic_memories = generate_episodic_memories(situation=situation, emotion=emotion, llm=llm)

                if episodic_memories:
                    joined_memories = "~".join(list(episodic_memories.values()))
                else:
                    joined_memories = ""

                results.append([situation_id, emotion, factor, situation, joined_memories])

        writer.writerows(results)

    print(f"Memories created and saved to {output_file}")
