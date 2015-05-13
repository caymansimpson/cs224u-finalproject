
#!/usr/bin/env python
# Cayman Simpson (cayman@stanford.edu), Harley Sugarman (harleys@stanford.edu), Angelica Perez (pereza77@stanford.edu)
# CS224U, Created: 3 May 2015
# file: Question.py

class Question:

    def __init__(self, string):
        """Question initialization."""
        arr = filter(lambda x: len(x) > 0, string.split("\n"));
        self.text = arr[0].strip();
        self.answers = map(lambda x: x.strip(), arr[1:-1]);
        self.correctAnswer = int(arr[-1]);

    def __str__(self):
        return "Question: " + self.text + "\n" + str(self.answers) + "\nCorrect Answer: " + str(self.correctAnswer)