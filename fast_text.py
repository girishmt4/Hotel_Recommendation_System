import fastText

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

def train_model():
    #Training the classifier over training dataset
    model = fastText.train_supervised('./fastText_classifier/train.ft.txt')
    #saving the classifier for future use
    model.save_model("./fastText_classifier/sentiment_model.bin")
    # model = fastText.load_model("./fastText_classifier/sentiment_model.bin")
    #Testing the classifier on the preseparated test test data
    print_results(*model.test("./fastText_classifier/test.ft.txt"))

def main():
    train_model()

if __name__ == "__main__":
    main()
