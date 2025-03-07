import json

from balrog.agents.utils.rag import parse_json

with open("local/data/processed_wiki.json", "r") as f:
    data = json.load(f)

data_sorted = sorted(data)

for item in data_sorted:
    print(item)
