
import nltk
import os
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot

def main():
    sentence_file= open('./text_files/file_without_stopwords.txt','r').read()
    sentences=sentence_file.split('\n')
    sentences=[word_tokenize(sentence) for sentence in sentences]
    model = Word2Vec(sentences, size=100, window=1, workers=8, min_count=1)
    words = list(model.wv.vocab)
    print(words)
    if not os.path.exists("./Word_Embeddings_Model/"):
        os.makedirs("./Word_Embeddings_Model/")
    model.save('./Word_Embeddings_Model/model.bin')

if __name__ == "__main__":
    main()
