# GenerativeAgentsPredictEmotion

## Setup
### Create and activate the Conda environment
```
conda create -n emotion_understanding python=3.10
conda activate predict_emotion
```

### Install the requirements
```
pip install -r requirements.txt
```

### Create Neo4j Database
Create a Neo4j datbase to store the memories and norms in a graph. The database can be hosted locally or on a cloud service such as [AuraDB](https://neo4j.com/cloud/platform/aura-graph-database/).

### Env
A `.env` file must be placed in the root directory and should contain your OpenAI API key and your Neo4j database credentials.
```
OPENAI_API_KEY=XXXXXXXX
NEO4J_URI=XXXXXXXX
NEO4J_USER=XXXXXXXX
NEO4J_PASSWORD=XXXXXXXX
```


## Usage

### Run
```
nohup python -u text_model/main.py &> agent.out &
```

### Analysis
Results obtained from text model experiments can be analyzed at [analysis](/analysis/).
Please refer to the README of [analysis](/analysis/) for details.


## Access and Query the Neo4j Database
If your Neo4j database is hosted by AuraDB, you can access it through [the AuraDB console](https://neo4j.com/cloud/platform/aura-graph-database/). Log in with your credentials and open the instance. Navigate to the "Query" tab.

Neo4j graph databases use Cypher to run queries. The following are some useful queries for this project:

Return all nodes and relationships:
```
MATCH (n)
OPTIONAL MATCH (n)-[r]->(m)
RETURN n, r, m
```

Return all of the nodes and relationships for a specific run/experiment:
```
MATCH (n {experiment_id: "Alice"})
OPTIONAL MATCH (n)-[r]->(m {experiment_id: "Alice"})
RETURN n, r, m
```
In particular, the `experiment_id` property refers to the `CaptioningAgent` `name` property. By tracking and filtering the `experiment_id` property, we can analyse the results of different runs and store them as subgraphs in the same database.

If you want to delete nodes and edges from a run, you can use:

```
MATCH (n {experiment_id: "<name>"})
OPTIONAL MATCH (n)-[r]-()
DELETE n, r
```
