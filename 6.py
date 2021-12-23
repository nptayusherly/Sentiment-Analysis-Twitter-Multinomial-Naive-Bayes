import re
import string
import nltk
import time
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(use_idf=True)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
from time import sleep
from nltk.corpus import stopwords
listStopword = set(nltk.corpus.stopwords.words('indonesian'))
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmerIndonesia = factory.create_stemmer()

def readDataset():
    global target, data
    df = pd.read_csv('preprocessing.csv', sep="\t")
    target = list(df["sentimen"])
    data = list(df["Tweet"]) 
    print((len(data)-1),"tweet loaded.")
    jumlahdata = int(input("Jumlah data digunakan: "))
    target = target[:jumlahdata]
    data = data[:jumlahdata]

def dataSplit(data, target):
    dataTest = float(input("Input size of Data Testing (0.0-1.0): "))
    data_train, data_test, labels_train, labels_test = train_test_split(data, target, test_size = dataTest, random_state = 0)
    print("\ndata_train, labels_train:",len(data_train),",",len(labels_train))
    print("data_test, labels_test:",len(data_test),",",len(labels_test))
    print("-----------------------------------------\n")
    print("Extraxtion Feature TD-IDF...")
    sleep(3)
    data_train_tfidf, data_test_tfidf = fiturExtract(data_train, data_test)
    print("-----------------------------------------\n")
    print("Predicting tweet sentiments...")
    sleep(3)
    data_test, predictions = classifierMNB(data_train_tfidf, data_test_tfidf, labels_train, data_test, labels_test)
    return data_test, labels_test, predictions

def fiturExtract(data_train, data_test):
    print("-----------------------------------------\n")
    print("Training the model...") 
    sleep(3)
    data_train_tfidf = tfidf_vectorizer.fit_transform(data_train)
    data_test_tfidf  = tfidf_vectorizer.transform(data_test)
    print("Model trained. Ready to predict!") 
    sleep(3)
    return data_train_tfidf, data_test_tfidf

def classifierMNB(data_train_tfidf, data_test_tfidf, labels_train, data_test, labels_test):
    print("-----------------------------------------\n")
    classifier.fit(data_train_tfidf, labels_train)    
    predictions = classifier.predict(data_test_tfidf)
    print("Confusion_matrix: \n", confusion_matrix(labels_test, predictions))
    print("\nClassification Report:\n")
    print(classification_report(labels_test, predictions))
    return data_test, predictions

def preProcessing(tweet):
    tweet = tweet.lower()  # Convert text to lowercase
    tweet = re.sub(r'\d+', '', tweet)  # Numbers removing
    tweet = re.sub(r'(.)\1+', r'\1\1', tweet) # funnnnny --> funny
    tweet = tweet.translate(str.maketrans("", "", string.punctuation))
    tweet = tweet.strip(r'\n')
    tweet = tweet.strip(' ')  # White spaces removal
    tweet = normalization(tweet)
    filtering = [i for i in list(tweet) if not i in listStopword]
    tweet = [term for term in filtering if (term != "" and term != "-")]
    tweet = " ".join(tweet)
    tweet = stemmerIndonesia.stem(tweet)
    return tweet

def normalization(tweet):
    df = pd.read_csv("colloquial-indonesian-lexicon.csv", sep=",")
    slang = list(df["slang"])
    formal = list(df["formal"])
    tweet = tweet.split()
    new_tweet = []
    for i in tweet:
        for index, j in enumerate(slang):
            if i == j:
                i = formal[index]
        new_tweet.append(i)
    return(new_tweet)

def main():
    start = time.time()
    print("Sentiment Analysis Using Machine Learning")
    print("-----------------------------------------\n")
    print("Reading the DataSet...")
    sleep(3)
    readDataset()
    print("-----------------------------------------\n")
    print("Split the DataSet into Data Testing & Da-\nta Training...")
    sleep(3)
    data_test, labels_test, predictions = dataSplit(data, target)
    datas = {'Tweets':data_test, 'Sentiment':labels_test, 'Prediction':predictions}
    
    print("-----------------------------------------\n")
    print("Writing results to Predicted.csv...")
    sleep(3)
    pd.DataFrame(datas).to_csv("Predicted01.csv")
    print("-----------------------------------------\n")
    end = time.time()
    print("Done in",(end-start), "seconds")
    print("-----------------------------------------\n")
    tweetss = input("Enter your tweet: ")
    while(tweetss != 'p'):
        tweets = preProcessing(tweetss)
        tweets = tfidf_vectorizer.transform([tweets])
        sentiments = classifier.predict(tweets)
        print(tweetss,":",sentiments)
        print("-----------------------------------------\n")
        tweetss = input("Enter your tweet: ")

if __name__ == "__main__":
    main()