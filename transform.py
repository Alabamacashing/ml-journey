import torch
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

result = classifier("This movie was not bad at all, actually pretty great!")
print(result)