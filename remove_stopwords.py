import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import re

sentdetector = nltk.data.load('tokenizers/punkt/english.pickle')
stop_words_punc=stopwords.words('english')

def main():
    file_without_stopwords = open("./text_files/file_without_stopwords.txt",'w')
    count=0
    for file in os.listdir("./text_files/text_hotels_reviews/"):
        if(file.endswith(".txt")):
            count+=1
            if(count%100==0):
                print(str(count)+" files read\n")
            review_file=open(os.path.join("./text_files/text_hotels_reviews/", file),'r').read()
            sentences = sentdetector.tokenize(review_file.strip(), realign_boundaries=True)
            for sent in sentences:
                tokens=word_tokenize(sent)
                #POS Tagging of the words
                for tok,pos in nltk.pos_tag(tokens):
                    if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                        #If the token is a noun then checked for stopwords.
                        #if not a stopword then stored in the file_without_stopwords
                        if tok.lower() not in stop_words_punc:
                            file_without_stopwords.write(tok.lower()+" ")
                file_without_stopwords.write("\n")

if __name__ == "__main__":
    main()
