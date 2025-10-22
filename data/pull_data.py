from datasets import load_dataset

from src.PaperDataset import PaperDataset

dataset = PaperDataset()
stream_dataset = load_dataset("jackkuo/arXiv-metadata-oai-snapshot", streaming=True, split="train")
dataset = stream_dataset.take(1000)

print(dataset)

print("names: ", dataset.column_names)
for entry in dataset:
    print(entry)