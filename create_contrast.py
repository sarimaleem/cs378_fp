import json

in_file = open("./eval_output/eval_predictions.jsonl", "r")
output = open("./contrast_set.jsonl", "a")
for (i, line) in enumerate(in_file):
    if (i < 30):
        continue

    if (i > 50):
        break
    example = json.loads(line)
    print("------------------------------------------------------")
    print("premise:    ", example["premise"])
    print("hypothesis: ", example["hypothesis"])
    print("gold label: ", example["label"])

    new_premise  = input("enter new premise (nothing for no change): ").strip()
    if(new_premise == ""):
        new_premise = example["premise"]
    
    new_hypothesis  = input("enter new hypothesis (nothing for no change): ").strip()
    if(new_hypothesis == ""):
        new_hypothesis = example["hypothesis"]

    new_label = input("enter new label (nothing for no change): ").strip()
    if(new_label == ""):
        new_label = example["label"]
    else:
        new_label = int(new_label)

    new_example = {"premise": new_premise, "hypothesis": new_hypothesis, "label": new_label}
    json_object = json.dumps(new_example) 
    output.write("\n" +  json_object)
    
    print("------------------------------------------------------")
