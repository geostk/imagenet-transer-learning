import os
import re
import multiprocessing
from glob import glob

import sklearn.manifold
import gensim.models.word2vec as w2v
from nltk.tokenize import PunktSentenceTokenizer

model_path = "trained"
model_name = os.path.join(model_path, "thrones2vec.w2v")
book_names = glob("../datasets/data/*.txt")

corpus = ""
for book_name in book_names:
    with open(book_name) as f:
        print(book_name)
        corpus += f.read()
        print("Corpus is now {:,} words long".format(len(corpus)))

tokenizer = PunktSentenceTokenizer()
raw_sentences = tokenizer.tokenize(corpus)


def word_list(raw):
    """
    Converts raw sentences to list of words.
    :param raw: sentence to be cleaned up
    :return: list of words
    """
    clean_words = re.sub(r"[^a-zA-Z]", ' ', raw)
    return clean_words.split()


sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(word_list(raw_sentence))

token_count = sum([len(sentence) for sentence in sentences])
print("\nToken count = {:,}".format(token_count))

# Build Word2Vec model
num_features = 300
min_word_count = 3
num_workers = multiprocessing.cpu_count()
context_size = 7
thrones2vec = w2v.Word2Vec(
    sg=1,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    workers=num_workers
)
thrones2vec.build_vocab(sentences)

# # Train the model
# thrones2vec.train(sentences)
# if not os.path.exists(model_path):
#     os.mkdir(model_path)
# thrones2vec.save(model_name)

thrones2vec = w2v.Word2Vec.load(model_name)
man_sim = thrones2vec.most_similar(positive=['man', 'woman'], negative=['girl'], topn=1)
print(man_sim)

tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
