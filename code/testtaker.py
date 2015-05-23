
#!/usr/bin/env python
# Cayman Simpson (cayman@stanford.edu), Harley Sugarman (harleys@stanford.edu), Angelica Perez (pereza77@stanford.edu)
# CS224U, Created: 3 May 2015
# file: testaker.py

# =====================================================================================================================================================
# =====================================================================================================================================================
# ====================================================================== IMPORTS ======================================================================
# =====================================================================================================================================================
# =====================================================================================================================================================


# =====================================================================================================================================================
# =====================================================================================================================================================
# ================================================================= UTILITY FUNCTIONS =================================================================
# =====================================================================================================================================================
# =====================================================================================================================================================

# A command line progress bar that accepts an integer from 1-100
def update_progress(progress):
    sys.stdout.write('\r');
    sys.stdout.write('[{0}>{1}] {2}%'.format('='*int(progress/10), ' '*(10 - int(progress/10)), progress));
    sys.stdout.flush();

# Reads a file and returns the text contents
def readFile(filename):
    with open(filename) as f: return f.read();

# Throws an error.
#   First param: String that contains error/notification
#   Second param: Whether to halt program execution or not.
def error(msg, shouldExit):
    print '\033[91m' + msg + '\033[0m';
    if(shouldExit): sys.exit();

# Prints a success (in green).
def printSuccess(message):
    print '\n\033[92m' + str(message) + '\033[0m\n';

# Returns a list of all filenames that are recursively found down a path.
#   First param: String of initial directory to start searching
#   Second param (optional): A filter function that filters the files found. Default returns all files.
def getRecursiveFiles(path, filter_fn=lambda x: True):
    paths = [path]
    files = [];
    try:
        while(len(paths) > 0):
            path = paths[0] if paths[0][-1] != "/" else paths[0][:-1];
            children = [f for f in listdir(paths[0])];
            for child in children:
                if not isfile(join(path,child)) and "." not in f: paths.append(join(path,child));
                elif isfile(join(path,child)): files.append(join(path,child));
                paths = paths[1:]; #remove te path we just looked at
        return filter(filter_fn, files);
    except:
        error(path + " is not a directory. Exiting...", True);

# =====================================================================================================================================================
# =====================================================================================================================================================
# ================================================================== MAIN CODE BASE ===================================================================
# =====================================================================================================================================================
# =====================================================================================================================================================

# Loads all passages in file.
def loadPassages(path):
    files = getRecursiveFiles(path, lambda x: x[x.rfind("/") + 1] != "." and ".txt" in x and x[-1] != '~');
    return [Passage(filename) for filename in files];

# Main method
def main(f, o, g, v):
    if(v): print "Loading passages...";
    passages = loadPassages(f);

    if(v): print "Loading glove vectors...";
    glove = Glove(g, delimiter=" ", header=False, quoting=csv.QUOTE_NONE);

    if(v): print "Finished loading all data!";

    print glove.getVec("and");


# =====================================================================================================================================================
# =====================================================================================================================================================
# =============================================================== COMMAND LINE REFERENCE ==============================================================
# =====================================================================================================================================================
# =====================================================================================================================================================

# Command Line Reference:
#   Example call: python testaker.py -f "../data/passages" -v -c column_num -o "../output.txt"
#   1) -v: if you want this program to be verbose
#   2) -o: if you want this program to output results to a file (defaults to printing to console)
#   3) -f: filename or path flag pointing to data (necessary)
#   4) -g: filename for glove vectors, default "../data/glove_vectors/glove.6B.50d.txt"
if __name__ == "__main__":

    # Preliminary loading to get arguments
    import sys
    import time

    start = time.time();
    args = sys.argv[1:];

    v = reduce(lambda a,d: a or d== "-v", args, False);
    if(v): print "\nImporting modules..."

    f = "";
    o = "";
    g = "../data/glove_vectors/glove.6B.50d.txt";

    # Get command lime arguments
    for i, arg in enumerate(args):
        if(arg == "-f"): # extract the filename argument
            f = args[i+1];
        elif(arg == "-o"): # extract the output filename argument
            o = args[i+1];
        elif(arg == "-g"):
            g = args[i+1];

    # Report error if called the wrong way
    if(f == ""):
        error("You must use the -f flag to specify where to find that data.\n   1) -v: if you want this program to be verbose\n   2) -o: if you want this program to output results to a file (defaults to printing to console)\n   3) -f: filename or path flag pointing to data (necessary)\n    4) -g: path to glove vector file (defaults to '../data/glove_vectors/glove.6B.50d.txt'", True)


    # Loading Modules
    from nltk.tag.stanford import POSTagger
    from distributedwordreps import *
    import NaiveBayes as nb
    import time
    from os import listdir
    from os.path import isfile, join
    from Passage import *
    from Question import *
    from Glove import *

    if(v): print "All modules successfully loaded in " + str(int(time.time() - start)) +  " seconds!"

    # Main Method
    main(f, o, g, v);

    # Finished Testaker execution
    if(v): printSuccess("Program successfully finished and exited in " + str(int(time.time() - start)) +  " seconds!");
    sys.exit();

# =====================================================================================================================================================
# =====================================================================================================================================================
# =================================================================== EXAMPLE CALLS ===================================================================
# =====================================================================================================================================================
# =====================================================================================================================================================


"""
Example call of POSTagger:
======================================================
tagger = POSTagger(
        'stanford-postagger/models/french.tagger', 
        'stanford-postagger/stanford-postagger.jar',
        'utf-8'
    );

tagger.tag_sents(array_of_string_sentences);


Example call of NaiveBayes:
======================================================
classifier = nb.NaiveBayes();
nb.addExamples(["good","good","ok","bad"],["pos","pos","pos","neg"]);
print nb.classify("good");

Example call of Passages and Questions
======================================================
Passage(filename) <= creates questions and stores them in member
passage.text = "passage text"
passage.questions = [Question Object, Question Object]

Question(text) <= constructor, created within Passage constructor, text automatically passed
Question.text = "question prompt text"
question.answers = ["answer #0", "answer #1"]
question.correctAnswer = int_of_correct_answer <= corresponds with index of answers

Example call of Glove Module
======================================================
glove = Glove(filename);
print glove.getVec("and"); # <= prints out glove vector for that word, or None if word not in vocab
print glove.getVocab(); # <= returns an array of all the words the glove vectors have
"""





