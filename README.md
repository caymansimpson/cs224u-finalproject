# cs224u-finalproject
NLU Final Project to answerthe Critical Reading section of the SAT

By Cayman Simpson, Harley Sugarman and Angelica Perez -- 04/17/15

Dependencies:
NLTK
	-Documentation: http://www.nltk.org/api/nltk.tag.html#module-nltk.tag.stanford
	-Installation: : sudo easy_install pip / sudo pip install -U numpy / sudo pip install -U nltk

scikit-learn
	-Documentation: http://scikit-learn.org/stable/
	-Installation : pip install -U numpy scipy scikit-learn 


COMMAND LINE REFERENCE
===========================================================================

An example call to this program would be:
	python testaker.py -f "../data" -v -o "../output.txt"

Use the -f flag to point to find the data:
	python testtaker.py -f "../data"


If you want the program to be verbose, use the -v flag:
	python testaker.py -f "../data" -v

If you want the program to output the results to a file instead of printing to the console, use the -o flag and specify the file you want to write:
	python testaker.py -f "../data" -o "../output.txt"


Observe the bottom of testtaker.py file for details on how the terminal calling interacts with the test-taking model itself.