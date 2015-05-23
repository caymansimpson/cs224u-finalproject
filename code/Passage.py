


#!/usr/bin/env python
# Cayman Simpson (cayman@stanford.edu), Harley Sugarman (harleys@stanford.edu), Angelica Perez (pereza77@stanford.edu)
# CS224U, Created: 3 May 2015
# file: Passage.py

from Question import *

class Passage:
    def __init__(self, filename):
        """Passage initialization."""
        self.text = "";
        self.questions = [];

        with open(filename, "rb") as f:
            arr = f.read().split("###");
            self.text = arr[0].strip();
            self.questions = map(lambda qtext: Question(qtext), arr[1:]);

    def __str__(self):
    	string = "";
    	for question in self.questions:
    		string += str(question);
        return "Text:\n" + self.text + "\n\nQuestions:\n" + string