from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

documents = [
    "The Titanic sank in 1912 after hitting an iceberg.",
    "Machine learning models learn patterns from data.",
    "Python is a popular programming language for data science.",
    "The California housing dataset includes median income and house age.",
]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(documents)

question = "What programming language is popular in data science?"
question_embedding = embedder.encode(question)

similarities = util.cos_sim(question_embedding, doc_embeddings)
print(similarities)

best_match_idx = similarities.argmax()
best_document = documents[best_match_idx]
print(best_document)

qa_model = pipeline("question-answering")

result = qa_model(question=question, context=best_document)
print(result)