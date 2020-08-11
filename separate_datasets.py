import csv

def main():
    with open('./CSV_Files/Hotel_Reviews.csv', newline='') as org_file:
        review_reader = csv.reader(org_file)
        count=0
        with open('./CSV_Files/final_test_file.csv', 'w', newline='') as final_test_file:
            test_writer = csv.writer(final_test_file, delimiter=',')
            with open('./CSV_Files/recognizer_training_set.csv', 'w', newline='') as recognizer_training_set:
                recognizer_training_set_writer = csv.writer(recognizer_training_set, delimiter=',')
                with open('./CSV_Files/text_mining_set.csv', 'w', newline='') as text_mining_set:
                    text_mining_set_writer = csv.writer(text_mining_set, delimiter=',')
                    for row in review_reader:
                        if(count == 0):
                            test_writer.writerow(row)
                            recognizer_training_set_writer.writerow(row)
                            text_mining_set_writer.writerow(row)
                        else:
                            #Storing each 5000th entry for recommender testing to have diverse test data
                            if(count % 5000 == 0):
                                test_writer.writerow(row)
                                print(str(count)+" reviews read")
                            else:
                                #storing each 1225th entry for training thhe NE recognizer to train the NE recognizer with diverse data
                                if(count % 1225 == 0):
                                    recognizer_training_set_writer.writerow(row)
                                else:
                                    text_mining_set_writer.writerow(row)
                        count+=1

if __name__ == "__main__":
    main()
