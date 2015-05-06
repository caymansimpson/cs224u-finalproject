
#!/usr/bin/env python
# Cayman Simpson (cayman@stanford.edu)
# CS224U, Updated: 3 May 2015
# file: NaiveBayes.py
# This is a simple Naive Bayes implementation I have from a while back.

import math
import itertools as it

class NaiveBayes:
    class ClassStats:
        def __init__(self):
            self.num_features = 0
            self.num_examples = 0
            self.counter = {}

        def log_likelihood(self, features, feature_set_size):
            prob = 0.0
            for feature in features:
                prob -= math.log(self.num_features + feature_set_size)
                prob += math.log(self.counter.get(feature, 0) + 1)
            return prob

    def __init__(self):
        """NaiveBayes initialization."""
        self.stats = {}
        self.feature_set = set()
        self.class_set = set()
        self.total_examples = 0

    def addExamples(self, features_list, labels):
        """Trains the model on some data. Simply call this with your labeled feature vectors (feature
        vectors and their corresponding labels must be in the same order) to train the model."""
        for features, label in it.izip(features_list, labels):
            self.addExample(label, features)

    def classifyWithOptions(self, features, classes):
        """Returns a label for a set of features."""
        total_examples = 0
        for klass in self.class_set:
            total_examples = total_examples + self.stats[klass].num_examples
        log_total_examples = math.log(total_examples)

        probs = {}
        max_prob = float("-inf")
        for klass in self.class_set:
            if klass not in classes:
                continue

            probs[klass] = (self.stats[klass].log_likelihood(features, len(self.feature_set)) +
                    math.log(self.stats[klass].num_examples) - log_total_examples)
            max_prob = max(max_prob, probs[klass])

        for idx, klass in enumerate(probs):
            if probs[klass] >= max_prob:
                return klass

        return "NO LABEL";

    def classify(self, features):
        """Returns a label for a set of features."""
        total_examples = 0
        for klass in self.class_set:
            total_examples = total_examples + self.stats[klass].num_examples
        log_total_examples = math.log(total_examples)

        probs = {}
        max_prob = float("-inf")
        for klass in self.class_set:
            probs[klass] = (self.stats[klass].log_likelihood(features, len(self.feature_set)) +
                    math.log(self.stats[klass].num_examples) - log_total_examples)
            max_prob = max(max_prob, probs[klass])

        for klass in self.class_set:
            if probs[klass] >= max_prob:
                return klass

        return "NO LABEL";
    
    def scoreData(self, features_list, labels):
        correct = 0
        for features, label in it.izip(features_list, labels):
            if self.classify(features) == label:
                correct += 1
        return float(correct) / len(labels)
    
    def addExample(self, klass, features):
        self.class_set.add(klass)
        if klass not in self.stats:
            self.stats[klass] = self.ClassStats()
        class_stat = self.stats[klass]
        
        class_stat.num_examples += 1
        for feature in features:
            self.feature_set.add(feature)
            if feature not in class_stat.counter:
                class_stat.counter[feature] = 0.0
            class_stat.counter[feature] += 1
            class_stat.num_features += 1


def main():
    """For testing if you'd like"""
    pass;

if __name__ == '__main__':
    main()
