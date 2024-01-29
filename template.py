# Naive Bayes 3-class Classifier 
# Authors: Baktash Ansari - Sina Zamani 

# complete each of the class methods  

import math


class NaiveBayesClassifier:

    def __init__(self, classes: list[str]):
        # initialization: 
        # inputs: classes(list) --> list of label names
        # class_word_counts --> frequency dictionary for each class
        # class_counts --> number of instances of each class
        # vocab --> all unique words  
        self.classes = classes
        self.class_word_counts: list[dict[str, int]] = [dict() for _ in range(len(self.classes))]
        self.class_counts = [0 for _ in range(len(classes))]
        self.vocab = []

    def train(self, data: list[tuple[list[str], str]]):
        # training process:
        # inputs: data(list) --> each item of list is a tuple 
        # the first index of the tuple is a list of words and the second index is the label(positive, negative, or neutral)

        for features, label in data:
            index = self.classes.index(label)
            self.class_counts[index] += 1
            for word in features:
                if word not in self.vocab:
                    self.vocab.append(word)
                    for index in range(len(self.classes)):
                        self.class_word_counts[index][word] = 0
                self.class_word_counts[index][word] += 1

    def calculate_prior(self, label):
        # calculate log prior
        # you can add some attributes to this method

        index = self.classes.index(label)
        return math.log(self.class_counts[index] / sum(self.class_counts))

    def calculate_likelihood(self, word, label):
        # calculate likelihhood: P(word | label)
        # return the corresponding value
        
        if word not in self.vocab:
            return 0
        index = self.classes.index(label)
        return math.log((self.class_word_counts[index][word] + 1) / (sum(self.class_word_counts[index].values()) + len(self.vocab)))

    def classify(self, features: list[str]):
        # predict the class
        # inputs: features(list) --> words of a tweet 
        best_class = None 
        best_class_likelihood = -float("inf")
        for label in self.classes:
            likelihood = self.calculate_prior(label)
            for word in features:
                likelihood += self.calculate_likelihood(word, label)
            if (likelihood > best_class_likelihood):
                best_class = label
        return best_class
    

# Good luck :)
