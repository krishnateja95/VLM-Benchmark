
from datasets import load_dataset

ChartQA_dataset = load_dataset("HuggingFaceM4/ChartQA", split='test')
ChartQA_dataset = load_dataset("liuhaotian/LLaVA-Instruct-150K", split='test')

print(len(ChartQA_dataset))
example = ChartQA_dataset[0]

for j, input in enumerate(ChartQA_dataset):
    print(j, input['image'])
