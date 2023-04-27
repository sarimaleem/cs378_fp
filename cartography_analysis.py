import json
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

results500 = open("./model500/eval_predictions.jsonl", "r")
results1000 = open("./model1000/eval_predictions.jsonl", "r")
results1500 = open("./model1500/eval_predictions.jsonl", "r")
results2000 = open("./model2000/eval_predictions.jsonl", "r")
results2500 = open("./model2500/eval_predictions.jsonl", "r")
results3000 = open("./model3000/eval_predictions.jsonl", "r")
results= open("./model/eval_predictions.jsonl", "r")

results = [results500, results1000, results1500, results2000, results2500, results3000, results]
num_results = len(results)
num_lines = 50000

confidence_array = np.zeros(shape=(num_results, num_lines))
for i in range(num_results):
    res = results[i]
    for (j, line) in enumerate(res):
        if j >= num_lines:
            break
        example = json.loads(line)
        scores = np.array(example["predicted_scores"])
        label = example["label"]
        s_scores = softmax(scores)
        gold_prob = s_scores[label]
        confidence_array[i, j] = gold_prob

# print(confidence_array)
variance = np.std(confidence_array, axis=0)
mean = np.mean(confidence_array, axis=0)

df = pd.DataFrame({"confidence": mean, "variance": variance})
df["ambigious"] = (df["variance"] > 0.15) & (df["confidence"] > 0.4) & (df["confidence"] < 0.8)
df["difficult"] = (df["confidence"] < 0.5)

# sns.color_palette("flare", as_cmap=True)
# sns.scatterplot(df, x="variance", y="confidence", hue=df["difficult"])
# plt.savefig("paper/difficult.jpg")
# plt.show()

l = df.index[df["difficult"]].tolist()
f = open("difficult_labels.txt", "a")
for num in l:
    f.write(str(num) + "\n")