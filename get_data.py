from datasets import load_dataset
from hanziconv import HanziConv

dataset = load_dataset("cmrc2018", "cmrc2018")
train_dataset = dataset["train"]

for i in train_dataset:
    open("questions.txt", "a", encoding="utf-8").write(HanziConv.toTraditional(i["question"]).replace("\n", r"[\n]")+"\n")
    open("answers.txt", "a", encoding="utf-8").write(HanziConv.toTraditional(i["context"]).replace("\n", r"[\n]")+"\n")

