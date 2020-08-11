import pickle
import enum
import operator
import sys
# from sklearn.metrics.pairwise import cosine_similarity
import math

def dot_prod(v1, v2):
    return sum(map(operator.mul, v1, v2))


def cosine_sim(v1, v2):
    product = dot_prod(v1, v2)
    length1 = math.sqrt(dot_prod(v1, v1))
    length2 = math.sqrt(dot_prod(v2, v2))
    return product / (length1 * length2)

def usage():
    print("Arguments Needed : [list of preferences from below]")
    print("\nList of options: ")
    print("\t\t0 : LOCATION")
    print("\t\t1 : VALUE")
    print("\t\t2 : ROOM")
    print("\t\t3 : SERVICE")
    print("\t\t4 : FACILITY")
    print("\t\t5 : MEAL")
    print("\nNOTE : Only 6 arguments can be entered")

def main():
    if(len(sys.argv) == 1 or len(sys.argv) > 7):
        usage()
        exit(0)
    Hotels_Features=pickle.load(open("./pickle_files/Hotels_Features_Matrix_FastText.pickle","rb"))
    Hotels_List=pickle.load(open("./pickle_files/Hotels_List_FastText.pickle","rb"))
    preference = []
    #appending the preference list
    for i in range(1,len(sys.argv)):

        preference.append(int(sys.argv[i]))
    print(preference)
    recommend_hotels(Hotels_Features,Hotels_List,preference)

def recommend_hotels(Hotels_Features,Hotels_List,preference):
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
    top_count=0
    for item in sorted(Similarity_Score.items(), key=operator.itemgetter(1),reverse=True):
        top_count+=1
        print(str(item[0])+" : "+str(item[1]))
        if(top_count==10):
            break

if __name__ == "__main__":
    main()
