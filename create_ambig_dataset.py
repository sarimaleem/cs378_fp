import json

f = open("difficult_labels.txt")
m = open("model500/eval_predictions.jsonl")
t = open("difficult_examples.jsonl", "a")
labels = []

for label in f:
    labels.append(int(label))

for (i, line) in enumerate(m):
    if(i in labels):
        example = json.loads(line)
        train_ex = {"premise" : example["premise"], "hypothesis" : example["hypothesis"], "label" : example["label"]}
        json_object = json.dumps(train_ex) 
        t.write(json_object + "\n")