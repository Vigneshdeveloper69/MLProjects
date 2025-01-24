from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample text data
documents = [
    "The economy is growing rapidly.",
    "The stock market is performing well.",
    "Sports events are drawing large crowds.",
    "The economy and stock market are closely related."
]

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(documents)

svd = TruncatedSVD(n_components=2)

topic_matrix = svd.fit_transform(tfidf_matrix)

print("Laten Topics :\n", topic_matrix)