import pandas
import nltk
import pickle
import operator
from sner import Ner


tagger = Ner(host='localhost',port=9199)
sentdetector = nltk.data.load('tokenizers/punkt/english.pickle')


def main():

    df = pandas.read_csv('./CSV_Files/final_test_file.csv')
    filtered = df.filter(['Hotel','Negative','Positive','Score'])
    Hotels_Features=pickle.load(open("./pickle_files/Hotels_Features_Matrix_FastText.pickle","rb"))
    Hotels_List=pickle.load(open("./pickle_files/Hotels_List_FastText.pickle","rb"))
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
        #Defining the Score as a dictionary of Hotels and Score w.r.t. current user
        Score={}
        for hotel in range(1490):
            hotel_score=0
            for feature in preference:
                hotel_score += Hotels_Features[hotel][feature]
                #storing the scores in the dictionary
            Score[Hotels_List[hotel]] = hotel_score

        top_count=0
        #counting the toal number of entries for each rating score
        if(rating == 10):
            rating_10_total_hotels += 1
        if(rating >= 9):
            rating_9_total_hotels += 1
        if(rating >= 8):
            rating_8_total_hotels += 1
        if(rating < 7):
            rating_7_total_hotels += 1


        for item in sorted(Score.items(), key=operator.itemgetter(1),reverse=True):
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
