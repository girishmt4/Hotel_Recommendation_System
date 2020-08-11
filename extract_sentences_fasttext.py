import os
import nltk
import enum
import pickle
from math import log
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from sner import Ner
import fastText

#Defining enum for the Hotel Features
class Features(enum.Enum):
    LOCATION=0
    VALUE = 1
    ROOM = 2
    SERVICE = 3
    FACILITY = 4
    MEAL = 5

def main():
    #loading the trained text classification model
    model = fastText.load_model('./fastText_classifier/sentiment_model.bin')
    sentdetector = nltk.data.load('tokenizers/punkt/english.pickle')
    Hotels_Features = [[0 for i in range(6)] for i in range(1490)]
    Hotels_List = {}
    #Using Stanford NER Server to run the NER Trained Tagger
    tagger = Ner(host='localhost',port=9199)
    count=0
    hotel_count=0
    for file in os.listdir("./text_files/text_hotels_reviews/"):
        if(file.endswith(".txt")):
            #Storing the hotel names in the list
            Hotels_List[hotel_count]=str(file[:-4])
            review_file=open(os.path.join("./text_files/text_hotels_reviews/", file),'r').read()
            #initializing the entity occurances in the entire reviews file of that particular hotel
            feature_occurances={'LOCATION':0, 'VALUE':0, 'ROOM':0, 'SERVICE':0, 'FACILITY':0, 'MEAL':0}
            sentences = review_file.strip().split('\n')
            for sentence in sentences:
                #predicting the class
                labels,probabilities = model.predict(sentence,k=2)
                #calculating the difference between 2 class probabilities
                #if the difference is less than 0.5 then it's termed as neutral
                #if the label is positive and diff is greater than 0.5 and less than 0.8 then slightly positive
                #if the label is positive and diff is greater than 0.8 then most positive
                #if the label is negative and diff is greater than 0.5 and less than 0.8 then slightly negative
                #if the label is negative and diff is greater than 0.8 then most negative
                prob_diff = probabilities[0]-probabilities[1]
                if(prob_diff < 0.5):
                    score = 3
                else:
                    if(labels[0] == '__label__2'):
                        if(prob_diff >= 0.8):
                            score = 5
                        else:
                            score = 4
                    if(labels[0] == '__label__1'):
                        if(prob_diff >= 0.8):
                            score = 1
                        else:
                            score = 2
                #Tagging named entities with Trained NE Tagger
                output=tagger.tag(sentence)
                for tok,tag in output:
                    if(tag is not 'O'):
                        #incrementing the score for each feature in the Hotels_Features Matrix
                        feature_occurances[tag] += 1
                        if(tag == "LOCATION"):
                            Hotels_Features[hotel_count][Features.LOCATION.value] += score
                        else:
                            if(tag == "ROOM"):
                                Hotels_Features[hotel_count][Features.ROOM.value] += score
                            else:
                                if(tag == "SERVICE"):
                                    Hotels_Features[hotel_count][Features.SERVICE.value] += score
                                else:
                                    if(tag == "FACILITY"):
                                        Hotels_Features[hotel_count][Features.FACILITY.value] += score
                                    else:
                                        if(tag == "MEAL"):
                                            Hotels_Features[hotel_count][Features.MEAL.value] += score
                                        else:
                                            if(tag == "VALUE"):
                                                Hotels_Features[hotel_count][Features.VALUE.value] += score
            #Normalizing the values to the scale of 5 by dividing the total score by total number of tagged entities occurances
            print(feature_occurances)
            print(Hotels_Features[hotel_count])

            if(feature_occurances['LOCATION'] != 0):
                Hotels_Features[hotel_count][0] = Hotels_Features[hotel_count][0]/feature_occurances['LOCATION']

            if(feature_occurances['VALUE'] != 0):
                Hotels_Features[hotel_count][1] = Hotels_Features[hotel_count][1]/feature_occurances['VALUE']

            if(feature_occurances['ROOM'] != 0):
                Hotels_Features[hotel_count][2] = Hotels_Features[hotel_count][2]/feature_occurances['ROOM']

            if(feature_occurances['SERVICE'] != 0):
                Hotels_Features[hotel_count][3] = Hotels_Features[hotel_count][3]/feature_occurances['SERVICE']

            if(feature_occurances['FACILITY'] != 0):
                Hotels_Features[hotel_count][4] = Hotels_Features[hotel_count][4]/feature_occurances['FACILITY']

            if(feature_occurances['MEAL'] != 0):
                Hotels_Features[hotel_count][5] = Hotels_Features[hotel_count][5]/feature_occurances['MEAL']

            print(Hotels_Features[hotel_count])
            count+=1
            hotel_count+=1
            print(str(count)+" of 1490 files processed......")
    #Storing the matrix and hotels list into a pickle file for future use
    if not os.path.exists("./pickle_files/"):
        os.makedirs("./pickle_files/")
    pickle.dump(Hotels_Features,open("./pickle_files/Hotels_Features_Matrix_FastText.pickle","wb"))
    pickle.dump(Hotels_List,open("./pickle_files/Hotels_List_FastText.pickle","wb"))

if __name__ == "__main__":
    main()
