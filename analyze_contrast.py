import json
import numpy as np
from prettytable import PrettyTable
from tabulate import tabulate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

f = open("./contrast_output/eval_predictions.jsonl")


# {"premise": "Two women are embracing while holding to go packages.", 
# "hypothesis": "The sisters are hugging goodbye while holding to go packages after just eating lunch.", 
# "label": 1, 
# "predicted_scores": [-1.8910499811172485, 3.4238548278808594, -1.99180006980896], 
# "predicted_label": 1}

mispredict = 0
i = -1

y_true = []
y_predict = []

# print out wrong lines for specific analysis
for (i, line) in enumerate(f):
    example = json.loads(line)
    label = example["label"]
    predict =  example["predicted_label"]
    y_true.append(label)
    y_predict.append(predict)
    if(label != predict):
        mispredict += 1
        print("line:", i)
        print("premise:", example["premise"])
        print("hypothesis:", example["hypothesis"])
        print("label:", example["label"])
        print("predicted:", example["predicted_label"])
        print()
        print("----------------------------------------")
        print()

print(mispredict, "/", i, "mispreditions")