
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

# Outputs array for visualization in mathematica
def mathematicatize(array):
    return str(array).replace("[","{").replace("]","}").replace("(","{").replace(")","}");

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

def inform(msg):
    print '\033[93m' + str(msg) + '\033[0m';

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
# ================================================================= DISTANCE METRICS ==================================================================
# =====================================================================================================================================================
# =====================================================================================================================================================

def kldist(p,q):
    return reduce(lambda soFar,i: soFar + p[i]*np.log(p[i]/q[i]), xrange(len(p)), 0);

def jsd(p,q):
    p = map(lambda u: u/sum(p), p);
    q = map(lambda v: v/sum(q), q);
    m = .5*np.add(p,q);
    return np.sqrt(.5*kldist(p,m) + .5*kldist(q,m))

def L2(u,v):
    return reduce(lambda soFar,i: soFar + (u[i]-v[i])*(u[i]-v[i]), range(len(u)), 0);

def cosine(u, v):
    return scipy.spatial.distance.cosine(u, v)

# distributed reps has: cosine, L1 (euclidean), jaccard

# =====================================================================================================================================================
# =====================================================================================================================================================
# ================================================================== MAIN CODE BASE ===================================================================
# =====================================================================================================================================================
# =====================================================================================================================================================

def combineModels(m1, m2):
    w = [];
    for i, tup in enumerate(m1):
        if(m1[i][0] == m2[i][0]): # Agree on wrong answer
            w.append(tup);
    return w;

def percentWrong(arr):
    count = 0.0;
    right = 0.0;
    for tup in arr:
        if(tup[0] == -1 or tup[0] == None): continue;
        if(tup[0] != tup[1]): right += 1;
        count += 1;
    return right/count;

def rawScore(arr):
    score = 0.0;
    for tup in arr:
        if(tup[0] == -1 or tup[0] == None): continue;
        if(tup[0] == tup[1]): score += 1;
        else: score -= .25;
    return score;

# Loads all passages in file.
def loadPassages(path):
    files = getRecursiveFiles(path, lambda x: x[x.rfind("/") + 1] != "." and ".txt" in x and x[-1] != '~' and "norvig" not in x.lower());
    return [Passage(filename) for filename in files];

# Computes the sum of the glove vectors of all elements in words
def getSumVec(words, glove):
    targetvec = glove.getVec(words[0]);
    if(targetvec == None and v): error("Glove does not have \"" + words[0] + "\" in its vocabulary", False);

    for word in words[1:]:
        wordvec = glove.getVec(word);
        if(wordvec != None): targetvec = map(lambda i: targetvec[i] + wordvec[i], xrange(len(targetvec)));  
        else:
            if(v): error("Glove does not have \"" + word + "\" in its vocabulary", False);

    return targetvec

# Computes the average of the glove vectors of all elements in words
def getAverageVec(words, glove):
    start = 0;
    targetvec = glove.getVec(words[start]);
    while(targetvec == None):
        if(v): error("Glove does not have \"" + words[start] + "\" in its vocabulary", False);
        start += 1;
        targetvec = glove.getVec(words[start]);

    count = 0;
    for word in words[start:]:
        wordvec = glove.getVec(word);
        if(wordvec != None):
            count += 1;
            targetvec = map(lambda i: targetvec[i] + wordvec[i], xrange(len(targetvec)));
            
        else:
            if(v): error("Glove does not have \"" + word + "\" in its vocabulary", False);

    return map(lambda x: x/count, targetvec);

#returns a matrix of word-document frequencies
def createWordDocMatrix(passages, data_passages):
    allWords = set(); # to hold all answer words and words contained in all target senteces
    for passage in passages:
        for question in passage.questions:
            targetline = int(re.findall("[0-9]+", question.text)[0]) - 1; # Lines are 1 indexed

            sentence = passage.text.split("\n")[int(targetline)];
            sentence = re.split("[^A-Za-z0-9]", sentence);
            sentence = filter(lambda x: len(x) > 0, sentence);
            sentence = map(lambda x: x.strip().lower(), sentence);

            allWords |= set(sentence + question.answers);

    allWords = list(allWords);

    matrix = np.zeros((len(allWords), len(data_passages)));
    for i,dp in enumerate(data_passages):
        words = re.split("[^A-Za-z0-9]", dp.text);
        words = filter(lambda x: len(x) > 0, words);
        words = map(lambda x: x.strip().lower(), words);
        wordCounts = collections.Counter(words);

        for j,w in enumerate(allWords):
            if w in words: matrix[j][i] = wordCounts[w];
            
    return matrix, allWords

#returns a list of the top x words in sentence with the highest tfidf scores (defaults to 10)
def findTopX(sentence, tfidf, allWords, x=10):
    d = defaultdict(float);
    for word in sentence: d[word] = tfidf[allWords.index(word)];
    sorted_d = sorted(d.items(), key=operator.itemgetter(1));

    return [key for key,val in sorted_d][-x:];

# Returns a tfidf array with all the associated words
def computeTFIDFArray(passages, data_passages_file="../data/data_passages"):
    data_passages = loadPassages(data_passages_file);
    freqMatrix, allWords = createWordDocMatrix(passages, data_passages);

    tfidf_mat = tfidf(mat=freqMatrix)[0];
    tfidf_array = [];
    for i,row in enumerate(tfidf_mat):
        count, rowSum = 0.0, 0.0;
        for val in row:
            if val > 0.0:
                count += 1.0;
                rowSum += val;

        if count == 0.0: tfidf_array.append(0.0)
        else: tfidf_array.append(rowSum/count);

    return tfidf_array, allWords;

# Returns (unigram_dict, bigram_dict, trigram_dict)
def getGrams(filename="../data/data_passages/norvig.txt"):
    unigramCounts = collections.defaultdict(lambda: 1);
    bigramCounts = collections.defaultdict(lambda: []);
    trigramCounts = collections.defaultdict(lambda: []);

    sentences = readFile(filename).lower().split(".");
    sentences = map(lambda sentence: re.sub("[^A-Za-z\ \,\'\"]", "", sentence.replace("-"," ")).strip(), sentences);
    sentences = map(lambda sentence: filter(lambda word: len(word) > 0, re.split("[^A-Za-z]", sentence)), sentences);

    for sentence in sentences:
        for i, word in enumerate(sentence):
            unigramCounts[word] += 1;
            if(i + 1 < len(sentence)): bigramCounts[word] += [sentence[i+1]]
            if(i + 2 < len(sentence)): trigramCounts[(word, sentence[i+1])] += [sentence[i+2]];

    return unigramCounts, bigramCounts, trigramCounts

# Finds the best answer given a target vector, answers, a distance function and a threshold
# Returns -1 if none of the answers fall within the threshold
# Returns None if an answer has a word we don't understand (the question is illegible);
def findBestVector(targetvec, answers, glove, distfunc, threshold):
    ind, mindist = -1, 10e100;
    for i,answer in enumerate(answers):
        vec = glove.getVec(answer);

        # Two word answer, adding the vector
        if(" " in answer): vec = getSumVec(answer.split(" "), glove);

        # Glove does not have the answer in its vocabulary
        if(vec == None):
            if(v): error("Glove does not have the answer \"" + answer + "\" in its vocabulary", False);
            return None;

        if( distfunc(vec, targetvec) < mindist and distfunc(vec, targetvec) < threshold ):
            ind, mindist = i, distfunc(vec, targetvec);

    return answers[ind];

# Finds the worst answer given a target vector, answers, a distance function and a threshold
# Returns -1 if none of the answers fall within the threshold
# Returns None if an answer has a word we don't understand (the question is illegible);
def findWorstVector(targetvec, answers, glove, distfunc, threshold):
    ind, mindist = -1, 0;
    for i,answer in enumerate(answers):
        vec = glove.getVec(answer);

        # Two word answer, adding the vector
        if(" " in answer): vec = getSumVec(answer.split(" "), glove);

        # Glove does not have the answer in its vocabulary
        if(vec == None):
            if(v): error("Glove does not have the answer \"" + answer + "\" in its vocabulary", False);
            return None;

        if( distfunc(vec, targetvec) > mindist and distfunc(vec, targetvec) < threshold ):
            ind, mindist = i, distfunc(vec, targetvec);

    return answers[ind];

#returns lists of nouns, verbs, and adjectives of sentence
def getTargetVecs(sentence):
    nounVec = []
    verbVec = []
    adjVec = []
    for word in sentence:
        ss = wn.synsets(word)
        if len(ss) < 1 or word in stopwords.words('english'): continue
        pos = str(ss[0].pos())
        if pos == 'n':
            nounVec.append(word)
        elif pos == 'v':
            verbVec.append(word)
        elif pos == 'a':
            adjVec.append(word)
    return nounVec, verbVec, adjVec


# Returns cooccurrence counts within sentences
def cooccurrence(filename="../data/data_passages/norvig.txt"):
    cooccurCounts = collections.defaultdict(lambda: []);

    #for f in gutenberg.fileids():
    sentences = readFile(filename).lower().split(".");
    sentences = map(lambda sentence: re.sub("[^A-Za-z\ \,\'\"]", "", sentence.replace("-"," ")).strip(), sentences);
    sentences = map(lambda sentence: filter(lambda word: len(word) > 0, re.split("[^A-Za-z]", sentence)), sentences);

    for sentence in sentences:
        for w1, w2 in itertools.product(sentence, sentence):
            w1 = w1.lower()
            w2 = w2.lower()
            if w1 != w2:
                    cooccurCounts[w1] += [w2]

    return cooccurCounts

#####################################################################################################################
###################################################### MODELS #######################################################
#####################################################################################################################


# Returns answer word based on random chance, given the answers 
def randomModel(answers):
    return answers[random.randint(0,len(answers)) - 1];

# Returns answer word by taking the nearest neighbor of the ambiguous word in the question
# Returns None if the target (ambiguous) word or an answer is not in the glove vocab
# Returns -1 if no answers pass the confidence threshold
def nearestNeighborModel(targetword, answers, glove, distfunc=cosine, threshold=1):
    targetvec = glove.getVec(targetword);

    # Glove does not have the target word in its vocabulary
    if(targetvec == None):
        if(v): error("Glove does not have \"" + targetword + "\" in its vocabulary", False);
        return None;

    return findWorstVector(targetvec, answers, glove, distfunc, threshold)

# Sentence is an array of words
# Returns answer word by averaging the sentence passed in.
# Returns None if an answer doesn't exist in the glove vocab
# Returns -1 if no answers pass the confidence threshold
def sentenceModel(sentence, answers, glove, distfunc=cosine, threshold=1):
    targetvec = getAverageVec(sentence, glove);
    ind, mindist = -1, 10e100;

    return findWorstVector(targetvec, answers, glove, distfunc, threshold)


# Sentence is an array of words
# Returns a chosen answer based on pre-computed tfidf values
# Returns None if an answer isn't in the glove dictionary
# Returns -1 if an answer doesn't beat the threshold
def tfidfModel(sentence, answers, tfidf_array, allWords, glove, distfunc=cosine, threshold=1):
    topX = findTopX(sentence, tfidf_array, allWords, 15);
    targetvec = getSumVec(topX, glove);

    return findWorstVector(targetvec, answers, glove, distfunc, threshold)


# Sentence is an array of words
# Returns a chosen answer based on pre-computed tfidf values
# Returns None if an answer isn't in the glove dictionary
# Returns -1 if an answer doesn't beat the threshold
def gramModel(sentence, answers, targetword, unigrams, bigrams, trigrams, glove, distfunc=cosine, threshold=1):

    prediction = "";

    # Try trigrams
    index = -1;
    try: index = sentence.index(targetword);
    except: return -1; # Something went wrong

    if(index >= 2 and (sentence[index-2], sentence[index-1]) in trigrams):
        prediction = max(set(trigrams[(sentence[index-2], sentence[index-1])]), key=trigrams[(sentence[index-2], sentence[index-1])].count);

    # Try bigrams
    elif(index >= 1 and sentence[index-1] in bigrams):
        prediction = max(set(bigrams[sentence[index-1]]), key=bigrams[sentence[index-1]].count)

    # TODO: Integrate disambiguate function and use most common word
    else: # Right now: Sends signal that we don't know answer -- we don't answer the question
        return -1;

    targetvec = glove.getVec(prediction);
    return findWorstVector(targetvec, answers, glove, distfunc, threshold)

# Replace the target word with each synonym, then check bigram dictionary
# for commonality, guess accordingly.
def synonymModel(targetword, sentence, answers, bigrams, trigrams, glove, distfunc=cosine, threshold=1):
    guess = -1
    bestScore = 0
    for i, answer in enumerate(answers):
        if targetword in sentence:
            wordBeforeTarget = sentence[sentence.index(targetword) - 1]
            bigram_dict = dict(collections.Counter(bigrams[wordBeforeTarget]))
            score = 0
            if answer in bigram_dict:
                score = bigram_dict[answer]
            if score > bestScore:
                bestScore = score
                guess = i

    targetvec = glove.getVec(answers[guess])
    if(targetvec == None):
        if(v): error("Glove does not have \"" + targetword + "\" in its vocabulary", False)
        return None
    return findWorstVector(targetvec, answers, glove, distfunc, threshold)

def wordnetModel(targetword, sentence, answers, glove, distfunc=cosine, threshold=1):
    target_synonyms = list(set(synset.name()[:-5] for synset in wn.synsets(targetword)))
    target_synonyms.append(targetword)
    targetvec = glove.getVec(targetword)
    if len(target_synonyms) > 1:
        targetvec = getAverageVec(target_synonyms, glove)
    wordnet_vectors = []
    for i, answer in enumerate(answers):
        answer_synonyms = list(set(synset.name()[:-5] for synset in wn.synsets(answers[i])))
        answer_synonyms.append(answer)
        wn_syn_vector = glove.getVec(answer)
        if len (answer_synonyms) > 1:
            wn_syn_vector = getAverageVec(answer_synonyms, glove)
        wordnet_vectors.append(wn_syn_vector)

    if(targetvec == None):
        if(v): error("Glove does not have \"" + targetword + "\" in its vocabulary", False)
        return None

    ind, mindist = -1, 10e100;
    for i, wnv in enumerate(wordnet_vectors):
        if(wnv == None):
            continue
        if( distfunc(wnv, targetvec) < mindist and distfunc(wnv, targetvec) < threshold ):
            ind, mindist = i, distfunc(wnv, targetvec)
    return answers[ind];

#uses tfidf and coccurrence data to compute an answer vector
def cooccurrenceModel(targetword, sentence, answers, cooccurrences, glove, distfunc=cosine, threshold=1):
    guess = -1
    bestScore = 0
    target_dict = dict(collections.Counter(cooccurrences[targetword]))
    targetCount = sum(target_dict.values())
    n, v, a = getTargetVecs(sentence)
    topWords = n+v+a
    for i, answer in enumerate(answers):
        if targetword in sentence:
            answer_dict = dict(collections.Counter(cooccurrences[answer]))
            answerCount = sum(answer_dict.values())
            score = 0


            for w in topWords:
                if w in answer_dict:
                    score += (answer_dict[w]/answerCount)
            if score > bestScore:
                bestScore = score
                guess = i

    targetvec = glove.getVec(answers[guess])
    if(targetvec == None):
        if(v): error("Glove does not have \"" + targetword + "\" in its vocabulary", False)
        return None
    return findWorstVector(targetvec, answers, glove, distfunc, threshold)

def findTopAnalogy(targetvec, answervec, tlist, alist, glove):
    score = 0

    for w1, w2 in itertools.product(tlist, alist):
        vec1 = glove.getVec(w1)
        vec2 = glove.getVec(w2)
        if vec1 != None and vec2 != None:
            s = abs( cosine( np.subtract(vec1,vec2), np.subtract(targetvec,answervec) ) )

        if s > score:
            score = s

    return score


#uses the Glove vector analogy equation to compute a score for each answer
def analogyModel(targetword, sentence, answers, cooccurrences, glove, distfunc=cosine, threshold=1):
    guess = -1
    bestScore = 0
    target_dict = dict(collections.Counter(cooccurrences[targetword]))
    targetCount = sum(target_dict.values())
    targetvec = glove.getVec(targetword)
    n, v, a = getTargetVecs(sentence)
    for i, answer in enumerate(answers):
        answervec = glove.getVec(answer)
        if targetword in sentence and targetvec != None and answervec != None:
            answer_dict = dict(collections.Counter(cooccurrences[answer]))
            answerCount = sum(answer_dict.values())
            answer_set = list(set(cooccurrences[answer]))
            an, av, aa = getTargetVecs(answer_set)
            nscore = findTopAnalogy(targetvec, answervec, n, an, glove)
            vscore = findTopAnalogy(targetvec, answervec, v, av, glove)
            ascore = findTopAnalogy(targetvec, answervec, a, aa, glove)

            score = max(nscore, vscore, ascore)
            if score > bestScore:
                bestScore = score
                guess = i

    targetvec = glove.getVec(answers[guess])
    if(targetvec == None):
        if(v): error("Glove does not have \"" + targetword + "\" in its vocabulary", False)
        return None
    return findWorstVector(targetvec, answers, glove, distfunc, threshold)


# Main method
def main():
    if(v): print "Loading passages...";
    passages = loadPassages(f);

    # Initialize all the external data
    if(v): print "Loading all external data...";
    tfidf_array, allWords = computeTFIDFArray(passages);
    unigrams, bigrams, trigrams = getGrams();
    glove = Glove(g, delimiter=" ", header=False, quoting=csv.QUOTE_NONE);
    cooccurrences = cooccurrence()

    if(v): print "Running models..."
    # Initialize arrays to keep answers
    rand, nn, sent, tfidf, gram, syn, wdn, cc, an = [], [], [], [], [], [], [], [], [];
    
    # Loop through all the questions
    for passage in passages:
        for question in passage.questions:

            # Find relevant word
            targetword = re.findall("[\xe2\x80\x9c\u2019\"\']([A-Za-z\s]+)[\xe2\x80\x9c\u2019\"\']", question.text)[0].lower();

            # Tokenize relevant sentence
            sentence = passage.text.split("\n")[int(re.findall("[0-9]+", question.text)[0]) - 1];
            sentence = re.split("[^A-Za-z0-9]", sentence);
            sentence = filter(lambda x: len(x) > 0, sentence);
            sentence = map(lambda x: x.strip().lower(), sentence);

            # Get correct answer
            correctAnswer = question.answers[question.correctAnswer];


            # Get answers
            randAnswer = randomModel(question.answers);
            nnAnswer = nearestNeighborModel(targetword, question.answers, glove);
            sentAnswer = sentenceModel(sentence, question.answers, glove);
            tfidfAnswer = tfidfModel(sentence, question.answers, tfidf_array, allWords, glove);
            gramAnswer = gramModel(sentence, question.answers, targetword, unigrams, bigrams, trigrams, glove);
            synAnswer = synonymModel(targetword, sentence, question.answers, bigrams, trigrams, glove)
            wdnAnswer = wordnetModel(targetword, sentence, question.answers, glove, threshold=0.3)
            ccAnswer = cooccurrenceModel(targetword, sentence, question.answers,cooccurrences, glove)
            anAnswer = analogyModel(targetword, sentence, question.answers, cooccurrences, glove)


            # Guess the word if we can answer it
            rand.append( (randAnswer, correctAnswer) );
            nn.append( (nnAnswer, correctAnswer) );
            sent.append( (sentAnswer, correctAnswer) );
            tfidf.append( (tfidfAnswer, correctAnswer) );
            gram.append( (gramAnswer, correctAnswer) );
            syn.append( (synAnswer, correctAnswer) )
            wdn.append( (wdnAnswer, correctAnswer) )
            cc.append( (ccAnswer, correctAnswer) )
            an.append(  (anAnswer, correctAnswer) )

    print "NN: ", percentWrong(nn);
    print "Sent: ", percentWrong(sent);
    print "gram: ", percentWrong(gram);
    print "tfidf: ", percentWrong(tfidf);
    print "syn: ", percentWrong(syn);
    print "wdn: ", percentWrong(wdn);
    print "cc: ", percentWrong(cc);
    print "an: ", percentWrong(an);

    names = ["NN","sent","gram","tfidf","syn","wdn","cc","an"]
    for m1 in zip(names, [nn, sent, gram, tfidf, syn, wdn, cc, an]):
        for m2 in zip(names, [nn, sent, gram, tfidf, syn, wdn, cc, an]):
            print m1[0], m2[0], percentWrong(combineModels(m1[1], m2[1])), len(combineModels(m1[1], m2[1]));
    # score_model(rand, verbose=True, modelname="Random Model");
    # score_model(nn, verbose=True, modelname="Nearest Neighbor Model");
    # score_model(sent, verbose=True, modelname="Sentence-Based Model");
    # score_model(tfidf, verbose=True, modelname="TFIDF Model");
    # score_model(gram, verbose=True, modelname="Gram Model");
    # score_model(syn, verbose=True, modelname="Synonym Model")
    # score_model(wdn, verbose=True, modelname="WordNet Model")
    # score_model(cc, verbose=True, modelname="Cooccurrence Model")
    # score_model(an, verbose=True, modelname="Analogy Model")


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

    args = sys.argv[1:];
    start = time.time();

    v = reduce(lambda a,d: a or d== "-v", args, False);
    if(v): inform("\nImporting modules...")

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
        error("You must use the -f flag to specify where to find that data.\n" + 
            "   1) -v: if you want this program to be verbose\n" +
            "   2) -o: if you want this program to output results to a file (defaults to printing to console)\n" +
            "   3) -f: filename or path flag pointing to data (necessary)\n" + 
            "   4) -g: path to glove vector file (defaults to '../data/glove_vectors/glove.6B.50d.txt'", True)


    # Loading Modules
    import scipy
    from sklearn import svm
    from nltk.tag.stanford import POSTagger
    from nltk.corpus import wordnet as wn
    from nltk.corpus import stopwords
    from nltk.corpus import gutenberg
    from distributedwordreps import *
    import NaiveBayes as nb
    from os import listdir
    from os.path import isfile, join
    import random
    import collections
    import operator
    import re
    from Passage import *
    from Question import *
    from Glove import *
    from scoring import score_model
    import numpy as np
    if(v): print "All modules successfully loaded in " + str(int(time.time() - start)) +  " seconds!"

    # Main Method
    main();

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





