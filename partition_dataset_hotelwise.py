import pandas
import nltk
import re
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def main():
    stop_words = set(stopwords.words('english'))
    sentdetector = nltk.data.load('tokenizers/punkt/english.pickle')
    df = pandas.read_csv('./CSV_Files/text_mining_set.csv')
    filtered = df.filter(['Hotel','Negative','Positive'])
    #Getting unique hotels list
    hotels_list = filtered.Hotel.unique()
    count=0
    index=0
    for each_hotel in hotels_list:
        hotel_name=str(each_hotel)
        frame=filtered[(filtered.Hotel==hotel_name)]
        directory="./text_files/text_hotels_reviews/"
        #create a directory if not exists
        if not os.path.exists(directory):
            os.makedirs(directory)
        review_file = open(directory+hotel_name+".txt",'w')
        #copying the reviews to respective text files
        for row in frame.itertuples():
            if(row[2].strip() != "No Negative"):
                sentences = sentdetector.tokenize(row[2].strip(), realign_boundaries=True)
                for sentence in sentences:
                    if(len(sentence.split())>1):
                        review_file.write(sentence+"\n")
            if(row[3].strip() != "No Positive"):
                sentences = sentdetector.tokenize(row[3].strip(), realign_boundaries=True)
                for sentence in sentences:
                    if(len(sentence.split())>1):
                        review_file.write(sentence+"\n")
        if(count%10==0):
            print(str(count)+" hotels processed")
        count+=1
    review_file.close()

if __name__ == "__main__":
    main()
