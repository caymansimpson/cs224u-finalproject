
#!/usr/bin/env python
# Cayman Simpson (cayman@stanford.edu), Harley Sugarman (harleys@stanford.edu), Angelica Perez (pereza77@stanford.edu)
# CS224U, Created: 3 May 2015
# file: Question.py

class Question:

    def __init__(self, string):
        """Question initialization."""
        arr = string.split("\n");
        self.text = arr[0].trim();
        self.answers = map(lambda x: x.trim(), arr[1:-1]);
        self.correctAnswer = int(arr[-1]);