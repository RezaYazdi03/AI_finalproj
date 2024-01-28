from template import NaiveBayesClassifier
import csv

pronoun =   ( 
            '', '.' ,',' ,';' ,':' ,'::'
            'I' ,'i' ,'am' ,"I'am" ,'my' ,
            'you' ,"you're" ,'are' ,'were' ,
            'he' ,'she' ,'is' ,'was' ,
            'them' ,'this' ,'that' ,
            'we' ,'We' ,"we're"
            'the' ,'a' ,'an' ,
            'of' ,'in' ,'to' ,'on' ,'not'
            'do', 'does'
            )

def preprocess(tweet_string: str):
    # clean the data and tokenize it
    features = []
    l = tweet_string.split()
    for s in l :
        i = ''.join(e for e in s if e.isalnum())
        if i not in pronoun:
            features.append(i)
    return features

def load_data(data_path):
    # load the csv file and return the data
    data = []
    with open(data_path, mode ='r')as file:
        d = csv.DictReader(file)
        for line in d:
            data.append([preprocess(line['text']),line['label_text']])
    return data


# train your model and report the duration time
train_data_path = 'train_data.csv'
classes = ['positive', 'negative', 'neutral']
nb_classifier = NaiveBayesClassifier(classes)
nb_classifier.train(load_data(train_data_path))

# test_string = "I love playing football"

# print(nb_classifier.classify(preprocess(test_string)))
