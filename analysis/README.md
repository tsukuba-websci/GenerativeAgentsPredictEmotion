# Analyzing Output PANAS Scores

## Creating Line Charts
Visualizing the changes in PANAS scores from scene 1 to 5 for each situation.

### Preprocessing
Reads results stored in a neo4j database, calculates the average sum of positive and negative scores for each scene, and outputs the results in CSV format.

```
python analysis/panas_csv.py
```
To process results without norms, execute `python analysis/panas_csv.py --norm no_norm`.

### Generating Graphs
Generates line charts for each situation.

```
python analysis/make_linechart.py
```

## Creating Tables
### Preprocessing
Reads results stored in a neo4j database and calculates the average, maximum, and minimum scores for each scene, then writes the results in JSON format.

```
python analysis/panas_json.py
```
To process results without norms, execute `python analysis/panas_json.py --norm no_norm`.

Additionally, calculate default scores (scores without any given scenarios) and save them in JSON format.

```
python calc_default.py
```

### Outputting a Summary of Results in Markdown Format
Determines the difference from the default scores for each situation and outputs a summary of the results in markdown format.

```
python analysis/analysis.py
```
Whether to use norms and which of the scene averages, maximums, or minimums to use is determined via command line arguments. Details can be checked with `python analysis/analysis.py -h`.
