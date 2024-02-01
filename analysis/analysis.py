import argparse
import json
import os
from statistics import mean, stdev

import pandas as pd
import scipy.stats as stats

emotions_list = [
    "Anger",
    "Anxiety",
    "Depression",
    "Frustration",
    "Jealousy",
    "Guilt",
    "Fear",
    "Embarrassment",
]

factors_list = [
    "Anger-0",
    "Anger-1",
    "Anger-2",
    "Anger-3",
    "Anger-4",
    "Anxiety-0",
    "Anxiety-1",
    "Anxiety-2",
    "Anxiety-3",
    "Depression-0",
    "Depression-1",
    "Depression-2",
    "Depression-3",
    "Depression-4",
    "Depression-5",
    "Frustration-0",
    "Frustration-1",
    "Frustration-2",
    "Frustration-3",
    "Jealousy-0",
    "Jealousy-1",
    "Jealousy-2",
    "Jealousy-3",
    "Guilt-0",
    "Guilt-1",
    "Guilt-2",
    "Guilt-3",
    "Fear-0",
    "Fear-1",
    "Fear-2",
    "Fear-3",
    "Fear-4",
    "Embarrassment-0",
    "Embarrassment-1",
    "Embarrassment-2",
    "Embarrassment-3",
]

order_list = {
    "Interested": 1,
    "Distressed": 2,
    "Excited": 3,
    "Upset": 4,
    "Strong": 5,
    "Guilty": 6,
    "Scared": 7,
    "Hostile": 8,
    "Enthusiastic": 9,
    "Proud": 10,
    "Irritable": 11,
    "Alert": 12,
    "Ashamed": 13,
    "Inspired": 14,
    "Nervous": 15,
    "Determined": 16,
    "Attentive": 17,
    "Jittery": 18,
    "Active": 19,
    "Afraid": 20,
}

categories = [
    {"cat_name": "Positive Affect", "cat_questions": [1, 3, 5, 9, 10, 12, 14, 16, 17, 19]},
    {"cat_name": "Negative Affect", "cat_questions": [2, 4, 6, 7, 8, 11, 13, 15, 18, 20]},
]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="average",
        choices=["average", "max", "min"],
        help="average or max or min",
    )
    parser.add_argument(
        "--norm",
        type=str,
        choices=["norm", "no_norm"],
        default="norm",
        help="norm or no_norm",
    )
    return parser.parse_args()


def setup_files(norm, mode):
    input_file = "results/panas.json" if norm == "norm" else "results/panas_no_norm.json"
    directory = "results/Proposed" if norm == "norm" else "results/Proposed_no_norm"
    output_file = f"{directory}/{mode}.md"
    os.makedirs(directory, exist_ok=True)

    return input_file, output_file


def convert_default_data(default_json):
    defaults = json.load(open(default_json))

    default_scores = []
    for d in defaults:
        default_score = {}
        for feeling, score in d.items():
            default_score[order_list[feeling]] = score
        default_scores.append(default_score)

    return default_scores


def convert_proposal_data(testing_json, mode):
    df = pd.read_csv("data/processed/situations.csv", delimiter="|")
    factors = df[["ID", "Factor"]]

    testing = json.load(open(testing_json))

    organized_data = {
        emotion: {factor: {"factor_name": factor, "data": []} for factor in factors_list if factor.startswith(emotion)}
        for emotion in emotions_list
    }

    for d in testing:
        for situation_id, value in d.items():
            dict = {}
            for feeling, scores in value.items():
                dict[order_list[feeling]] = scores[mode]

            factor = factors[factors["ID"] == int(situation_id)]["Factor"].values[0]
            emotion = factor.split("-")[0]
            organized_data[emotion][factor]["data"].append(dict)

    return organized_data


def compute_statistics(data_list):
    cat_list = []
    results = []

    for cat in categories:
        scores_list = []

        for data in data_list:
            scores = []
            for key in data:
                if key in cat["cat_questions"]:
                    scores.append(data[key])

            scores_list.append(sum(scores))

        results.append([mean(scores_list), stdev(scores_list), len(scores_list)])
        cat_list.append(cat["cat_name"])

    return results, cat_list  # ([mean, std, size], cat_list)


def hypothesis_testing(result1, result2, cat_list, significance_level, title):
    output_list = f"| {title} |"
    output_text = f"### {title}\n"

    for i, cat_name in enumerate(cat_list):
        output_text += f"\n##### {cat_name}"

        # Extract the mean, std and size for both data sets
        mean1, std1, n1 = result1[i]
        mean2, std2, n2 = result2[i]
        # Add an epsilon to prevent the zero standard deviarion
        epsilon = 1e-8
        std1 += epsilon
        std2 += epsilon

        output_text += "\n- **Statistic**:\n"
        output_text += f"Corresponding Factor:\tmean1 = {mean1:.1f},\tstd1 = {std1:.1f},\tn1 = {n1}\n"
        output_text += f"Default:\tmean2 = {mean2:.1f},\tstd2 = {std2:.1f},\tn2 = {n2}\n"

        # Perform F-test
        output_text += "\n- **F-Test:**\n\n"

        if std1 > std2:
            f_value = std1**2 / std2**2
            df1, df2 = n1 - 1, n2 - 1
        else:
            f_value = std2**2 / std1**2
            df1, df2 = n2 - 1, n1 - 1

        p_value = (1 - stats.f.cdf(f_value, df1, df2)) * 2
        equal_var = True if p_value > significance_level else False

        output_text += f"\tf-value = {f_value:.4f}\t($df_1$ = {df1}, $df_2$ = {df2})\n\n"
        output_text += f"\tp-value = {p_value:.4f}\t(two-tailed test)\n\n"
        output_text += "\tNull hypothesis $H_0$ ($s_1^2$ = $s_2^2$): "

        if p_value > significance_level:
            output_text += f"\tSince p-value ({p_value:.4f}) > α ({significance_level}), $H_0$ cannot be rejected.\n\n"
            output_text += "\t**Conclusion ($s_1^2$ = $s_2^2$):** The variance of LLM's average responses in this factor is statistically equal to the variance of general.\n\n"
        else:
            output_text += f"\tSince p-value ({p_value:.4f}) < α ({significance_level}), $H_0$ is rejected.\n\n"
            output_text += "\t**Conclusion ($s_1^2$ ≠ $s_2^2$):** The variance of LLM's average responses in this factor is statistically unequal to the variance of general.\n\n"

        # Performing T-test
        output_text += (
            "- **Two Sample T-Test (Equal Variance):**\n\n"
            if equal_var
            else "- **Two Sample T-test (Welch's T-Test):**\n\n"
        )

        df = (
            n1 + n2 - 2
            if equal_var
            else ((std1**2 / n1 + std2**2 / n2) ** 2)
            / ((std1**2 / n1) ** 2 / (n1 - 1) + (std2**2 / n2) ** 2 / (n2 - 1))
        )
        t_value, p_value = stats.ttest_ind_from_stats(mean1, std1, n1, mean2, std2, n2, equal_var=equal_var)

        output_text += f"\tt-value = {t_value:.4f}\t($df$ = {df:.1f})\n\n"
        output_text += f"\tp-value = {p_value:.4f}\t(two-tailed test)\n\n"

        output_text += "\tNull hypothesis $H_0$ ($µ_1$ = $µ_2$): "
        if p_value > significance_level:
            output_text += f"\tSince p-value ({p_value:.4f}) > α ({significance_level}), $H_0$ cannot be rejected.\n\n"
            output_text += "\t**Conclusion ($µ_1$ = $µ_2$):** The average of LLM's responses in this factor is assumed to be equal to the average of general.\n\n"

            # output_list += f' $-$ ({'+' if (mean1 - mean2) > 0 else ''}{(mean1 - mean2):.1f}) |'
            # output_list += f' $-$ ({'+' if (mean1 - mean2) > 0 else ''}{(mean1 - mean2):.1f}) |'
            output_list += f""" $-$ ({'+' if (mean1 - mean2) > 0 else ''}{(mean1 - mean2):.1f}) |"""

        else:
            output_text += f"Since p-value ({p_value:.4f}) < α ({significance_level}), $H_0$ is rejected.\n\n"
            if t_value > 0:
                output_text += "\tAlternative hypothesis $H_1$ ($µ_1$ > $µ_2$): "
                output_text += (
                    f"\tSince p-value ({(1-p_value/2):.1f}) > α ({significance_level}), $H_1$ cannot be rejected.\n\n"
                )
                output_text += "\t**Conclusion ($µ_1$ > $µ_2$):** The average of LLM's responses in this factor is assumed to be larger than the average of general.\n\n"

                output_list += f" $\\uparrow$ (+{(mean1 - mean2):.1f}) |"
            else:
                output_text += "\tAlternative hypothesis $H_1$ ($µ_1$ < $µ_2$): "
                output_text += (
                    f"\tSince p-value ({(1-p_value/2):.1f}) > α ({significance_level}), $H_1$ cannot be rejected.\n\n"
                )
                output_text += "\t**Conclusion ($µ_1$ < $µ_2$):** The average of LLM's responses in this factor is assumed to be smaller than the average of general emotion.\n\n"

                output_list += f" $\\downarrow$ ({(mean1 - mean2):.1f}) |"

    output_list += f" {result1[0][2]} |\n"
    return (output_text, output_list)


if __name__ == "__main__":
    args = parse_arguments()
    input_file, output_file = setup_files(args.norm, args.mode)

    default_data = convert_default_data("results/default_panas.json")
    general_results, cat_list = compute_statistics(default_data)

    data = convert_proposal_data(input_file, args.mode)
    significance_level = 0.01

    overall_list = "# PANAS Results Analysis\n"
    markdown_output = ""  # overall markdown output text
    overall_data = []

    overall_list += "| Emotions | " + " | ".join(cat_list) + " | N |\n"
    overall_list += "| :---: |" + " | ".join([":---:" for i in cat_list]) + " | :---: |\n"
    overall_list += (
        "| Default |"
        + " | ".join([f"{r[0]:.1f} $\\pm$ {r[1]:.1f}" for r in general_results])
        + f" | {general_results[0][2]} |\n"
    )

    # Analyze the results for each emotion
    for emotion in data:
        emotion_output = ""  # markdown output text for the current emotion

        emotion_list = f"## {emotion}\n"
        emotion_list += "| Factors | " + " | ".join(cat_list) + " | N |\n"
        emotion_list += "| :---: |" + " | ".join([":---:" for i in cat_list]) + " | :---: |\n"
        emotion_list += (
            "| Default |"
            + " | ".join([f"{r[0]:.1f} $\\pm$ {r[1]:.1f}" for r in general_results])
            + f" | {general_results[0][2]} |\n"
        )

        emotion_data = []  # the data that belongs to the current emotion

        # Analyze the results for each factor
        for factor in data[emotion]:
            emotion_data += data[emotion][factor]["data"]
            overall_data += data[emotion][factor]["data"]
            results, _ = compute_statistics(data[emotion][factor]["data"])
            text_msg, list_msg = hypothesis_testing(
                results,
                general_results,
                cat_list,
                significance_level,
                data[emotion][factor]["factor_name"],
            )
            emotion_output += text_msg
            emotion_list += list_msg

        results, _ = compute_statistics(emotion_data)
        text_msg, list_msg = hypothesis_testing(results, general_results, cat_list, significance_level, "Overall")
        emotion_output += text_msg
        emotion_list += list_msg
        overall_list += list_msg.replace("Overall", emotion)
        markdown_output += emotion_list + "\n" + emotion_output

    markdown_output += "## Emotions Overall\n"
    results, _ = compute_statistics(overall_data)
    text_msg, list_msg = hypothesis_testing(results, general_results, cat_list, significance_level, "Overall")
    markdown_output += text_msg
    overall_list += list_msg

    with open(output_file, "w") as f:
        f.write(overall_list + "\n\n" + markdown_output)
