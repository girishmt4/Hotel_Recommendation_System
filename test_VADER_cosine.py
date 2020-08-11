import pandas
import nltk
import pickle
import operator
import math
from sner import Ner

tagger = Ner(host='localhost',port=9199)
sentdetector = nltk.data.load('tokenizers/punkt/english.pickle')


def dot_prod(v1, v2):
    return sum(map(operator.mul, v1, v2))


def cosine_sim(v1, v2):
    product = dot_prod(v1, v2)
    length1 = math.sqrt(dot_prod(v1, v1))
    length2 = math.sqrt(dot_prod(v2, v2))
    return product / (length1 * length2)

def main():

    df = pandas.read_csv('./CSV_Files/final_test_file.csv')
    filtered = df.filter(['Hotel','Negative','Positive','Score'])
    Hotels_Features=pickle.load(open("./pickle_files/Hotels_Features_Matrix.pickle","rb"))
    Hotels_List=pickle.load(open("./pickle_files/Hotels_List.pickle","rb"))
    evaluate_recommender(filtered,Hotels_Features,Hotels_List)

def evaluate_recommender(filtered,Hotels_Features,Hotels_List):
    rating_10_accuracy_score=0
    rating_10_total_hotels=0
    rating_9_accuracy_score=0
    rating_9_total_hotels=0
    rating_8_accuracy_score=0
    rating_8_total_hotels=0
    rating_7_accuracy_score=0
    rating_7_total_hotels=0

    for row in filtered.itertuples():
        review_text=""
        preference=[]
        rating=row[4]
        #appending the review text together
        if(row[2].strip() != "No Negative"):
            review_text=review_text+" "+row[2].strip()
        if(row[3].strip() != "No Positive"):
            review_text=review_text+" "+row[3].strip()
        sentences = sentdetector.tokenize(review_text, realign_boundaries=True)
        for sentence in sentences:
            #tagging Named entities
            output=tagger.tag(sentence)
            #appending the correct feature value to the preference list
            for tok,tag in output:
                if(tag is not 'O'):
                    if(tag == 'LOCATION'):
                        preference.append(0)
                    else:
                        if(tag == 'ROOM'):
                            preference.append(2)
                        else:
                            if(tag == 'SERVICE'):
                                preference.append(3)
                            else:
                                if(tag == 'FACILITY'):
                                    preference.append(4)
                                else:
                                    if(tag == 'MEAL'):
                                        preference.append(5)
                                    else:
                                        if(tag == 'VALUE'):
                                            preference.append(1)
        preference=list(set(preference))
        #Defining the Similarity_Score as a dictionary of Hotels and Score w.r.t. current user
        Similarity_Score={}
        good_hotels=[]

        for hotel in range(1490):
            flag = 1
            #appended to good hotels if the hotel has score of 4 or more for the features in preference list
            for i in preference:
                if(Hotels_Features[hotel][i] < 4):
                    flag = 0
            if(flag == 1):
                good_hotels.append(hotel)

        best_hotel_score=0
        best_hotel=0
        #finding out the best hotel = hotel having best total score
        for hotel in good_hotels:
            hotel_score=0
            for i in range(6):
                hotel_score += Hotels_Features[hotel][i]
            if(hotel_score > best_hotel_score):
                best_hotel_score = hotel_score
                best_hotel = hotel

        #CALCUATE THE COSINE SIMILARITY OF EACH GOOD_HOTEL W.R.T. THE BEST HOTEL
        for hotel in good_hotels:
            Similarity_Score[Hotels_List[hotel]] = cosine_sim(Hotels_Features[best_hotel],Hotels_Features[hotel])

        #counting the toal number of entries for each rating score
        top_count=0
        if(rating == 10):
            rating_10_total_hotels += 1
        if(rating >= 9):
            rating_9_total_hotels += 1
        if(rating >= 8):
            rating_8_total_hotels += 1
        if(rating < 7):
            rating_7_total_hotels += 1

        for item in sorted(Similarity_Score.items(), key=operator.itemgetter(1),reverse=True):
            top_count+=1
            #incrementing respective accuracy score w.r.t. evaluation strategy
            #rating == 10, this hotel should be in top 20
            #rating > 9, this hotel should be in top 50
            #rating > 8, this hotel should be in top 100
            #rating < 7, this hotel should not be in top 200
            if(rating == 10):
                if(row[1].strip() == str(item[0].strip())):
                    if(top_count <= 20):
                        rating_10_accuracy_score += 1
            if(rating > 9):
                if(row[1].strip() == str(item[0].strip())):
                    if(top_count <= 50):
                        rating_9_accuracy_score += 1
            if(rating > 8):
                if(row[1].strip() == str(item[0].strip())):
                    if(top_count <= 100):
                        rating_8_accuracy_score += 1
            if(rating < 7):
                if(row[1].strip() == str(item[0].strip())):
                    rating_7_accuracy_score += 1
            if(top_count==200):
                break
    print("10 rating Accuracy(in top 20) is = "+str(rating_10_accuracy_score)+"/"+str(rating_10_total_hotels)+" = "+str(rating_10_accuracy_score/rating_10_total_hotels))
    print("9 rating Accuracy(in top 50) is = "+str(rating_9_accuracy_score)+"/"+str(rating_9_total_hotels)+" = "+str(rating_9_accuracy_score/rating_9_total_hotels))
    print("8 rating Accuracy(in top 100) is = "+str(rating_8_accuracy_score)+"/"+str(rating_8_total_hotels)+" = "+str(rating_8_accuracy_score/rating_8_total_hotels))
    print("7 rating Accuracy(should not be in top 200) is = "+str(rating_7_total_hotels-rating_7_accuracy_score)+"/"+str(rating_7_total_hotels)+" = "+str((rating_7_total_hotels-rating_7_accuracy_score)/rating_7_total_hotels))

if __name__ == "__main__":
    main()
