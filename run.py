from template import NaiveBayesClassifier
import csv
import time


pronoun = (
    '', 'i', 'am', "i'am",
    'you', "you're", 'are', 'were',
    'he', 'she', 'is', 'was',
    'them', 'this', 'that',
    'we', "we're"
    'the', 'a', 'an',
    'of', 'in', 'to', 'on', 'not'
    'do', 'does'
)


def process_word(word: str):
    word = word.strip(" .!?:;@#'")
    if (word in pronoun):
        return None
    if (word[0:8] == "https://"):
        return None
    if (word[0:7] == "http://"):
        return None
    if (word[0:4] == "www."):
        return None

    # if (word[-3:] == "ing"):
    #     word = word[:-3]
    # elif (word[-2:] == "ed"):
    #     word = word[:-2]
    # elif (word[-4:] == "ness"):
    #     word = word[:-4]
    # elif (word[-3:] == "ion"):
    #     word = word[:-3]

    # if (word[-1:] == "y"):
    #     word = word[:-1] + "i"
    # elif (word[-1:] in ("e", "t", "s")):
    #     word = word[:-1]

    if (len(word) == 0):
        return None
    return word


def preprocess(tweet_string: str):
    # clean the data and tokenize it
    features = []
    for word in tweet_string.lower().split():
        f = process_word(word)
        if (f):
            features.append(f)
    return features


def load_data(data_path):
    # load the csv file and return the data
    data = []
    with open(data_path, mode='r') as file:
        d = csv.DictReader(file)
        for line in d:
            data.append([preprocess(line['text']), line['label_text']])
    return data


def evaluate(data_path):
    total = 0
    correct = 0
    with open(data_path, mode='r') as file:
        d = csv.DictReader(file)
        for line in d:
            total += 1
            label = nb_classifier.classify(preprocess(line['text']))
            if (label == line['label_text']):
                correct += 1
    print(
        f"Classified {total} tweets from {data_path} with {100 * correct / total :.2f}% accuracy")


def test(data_path, result_path):
    with open(data_path, mode='r') as input_file, open(result_path, mode='w') as output_file:
        d = csv.DictReader(input_file)
        for line in d:
            label = nb_classifier.classify(preprocess(line['text']))
            output_file.write(label)


# train your model and report the duration time
train_data_path = 'train_data.csv'
eval_data_path = 'eval_data.csv'
test_data_path = 'test_data_nolabel.csv'
classes = ['positive', 'negative', 'neutral']
nb_classifier = NaiveBayesClassifier(classes)

train_timer_start = time.time()
nb_classifier.train(load_data(train_data_path))
train_timer_end = time.time()
print(f"Training took {1000 * (train_timer_end - train_timer_start) :.0f}ms")

evaluate(train_data_path)
evaluate(eval_data_path)
test(test_data_path, "result.txt")

# test_string = "I love playing football"
# print(nb_classifier.classify(preprocess(test_string)))
