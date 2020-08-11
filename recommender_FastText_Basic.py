import pickle
import enum
import operator
import sys

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
    Score={}
    for hotel in range(1490):
        hotel_score=0
        #Calculating score for each hotel
        for feature in preference:
            hotel_score += Hotels_Features[hotel][feature]
        Score[Hotels_List[hotel]] = hotel_score

    top_count=0
    #sorting the hotels in descending order
    for item in sorted(Score.items(), key=operator.itemgetter(1),reverse=True):
        top_count+=1
        print(str(item[0])+" : "+str(item[1]))
        if(top_count==10):
            break


if __name__ == "__main__":
    main()
