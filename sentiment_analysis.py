import re, math, collections, itertools, os
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.corpus import stopwords

DIR = os.path.join('testdata', 'training')
#RT_POLARITY_POS_FILE = os.path.join(DIR, 'rt-polarity-pos.txt')
RT_POLARITY_POS_FILE = os.path.join(DIR, 'PositiveFile.txt')
#RT_POLARITY_NEG_FILE = os.path.join(DIR, 'rt-polarity-neg.txt')
RT_POLARITY_NEG_FILE = os.path.join(DIR, 'NegativeFile.txt')
RT_POLARITY_POS_FILE_review = os.path.join(DIR, 'rt-polarity-pos-review.txt')
RT_POLARITY_NEG_FILE_review = os.path.join(DIR, 'rt-polarity-neg-review.txt')

#this function takes a feature selection mechanism and returns its performance in a variety of metrics
def evaluate_features(feature_select):
	posFeatures = []
	negFeatures = []
	#http://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation
	#breaks up the sentences into lists of individual words (as selected by the input mechanism) and appends 'pos' or 'neg' after each list
	with open(RT_POLARITY_POS_FILE, 'r') as posSentences:
		for i in posSentences:
			posWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
			#posWords = [w for w in posWords if not w in stopwords.words('english')]
			#print "posWords ",posWords
			posWords = [feature_select(posWords), 'pos']
			posFeatures.append(posWords)
	with open(RT_POLARITY_NEG_FILE, 'r') as negSentences:
		for i in negSentences:
			negWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
			#negWords = [w for w in negWords if not w in stopwords.words('english')]
			negWords = [feature_select(negWords), 'neg']
			negFeatures.append(negWords)

	
	#selects 3/4 of the features to be used for training and 1/4 to be used for testing
	posCutoff = int(math.floor(len(posFeatures)*3/4))
	negCutoff = int(math.floor(len(negFeatures)*3/4))
	trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
	testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]
        #testFeatures="Very Good Movie" + " the next best thing" 
	#trains a Naive Bayes Classifier
	classifier = NaiveBayesClassifier.train(trainFeatures)	

	#initiates referenceSets and testSets
	referenceSets = collections.defaultdict(set)
	testSets = collections.defaultdict(set)	

	#puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
	for i, (features, label) in enumerate(testFeatures):
		referenceSets[label].add(i)
		#print "features ",features
		predicted = classifier.classify(features)
		#print " predicted classifications is ",predicted
		#print predicted
		testSets[predicted].add(i)	

	#prints metrics to show how well the feature selection did
	print 'train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures))
	#print 'accuracy:', nltk.classify.util.accuracy(classifier, testFeatures)
	#print 'pos precision:', nltk.metrics.precision(referenceSets['pos'], testSets['pos'])
	#print 'pos recall:', nltk.metrics.recall(referenceSets['pos'], testSets['pos'])
	#print 'neg precision:', nltk.metrics.precision(referenceSets['neg'], testSets['neg'])
	#print 'neg recall:', nltk.metrics.recall(referenceSets['neg'], testSets['neg'])
	classifier.show_most_informative_features(10)
	
	#print "labels ",classifier.labels()
	#print classifier.classify(testSets)
	

#creates a feature selection mechanism that uses all words
def make_full_dict(words):
	return dict([(word, True) for word in words])

#tries using all words as the feature selection mechanism
print "-----------------Welcome to company review system-----------------------"
print "Enter the pros of company"
posreview=raw_input()
print "Enter the cons of company"
negreview=raw_input()
evaluate_features(make_full_dict)

def create_word_scores(posreview,negreview):
	print "review",posreview
	print "negative review",negreview
	rposWords=[]
        rposWord=re.findall(r"[\w']+|[.,!?;]",posreview.rstrip())
        rposWord = [w for w in rposWord if not w in stopwords.words('english')]
        print "rposWords ",rposWord
        #print "RohitWords ",rposWord
        rposWords.append(rposWord)
        rposWords=list(itertools.chain(*rposWords))

        rnegWords=[]
        rnegWord=re.findall(r"[\w']+|[.,!?;]",negreview.rstrip())
        rnegWord = [w for w in rnegWord if not w in stopwords.words('english')]
        rnegWords.append(rnegWord)
        rnegWords=list(itertools.chain(*rnegWords))
       	word_fd = FreqDist()
	cond_word_fd = ConditionalFreqDist()
	for word in rposWords:
                word_fd.inc(word.lower())
		cond_word_fd['pos'].inc(word.lower())
	for word in rnegWords:
		word_fd.inc(word.lower())
		cond_word_fd['neg'].inc(word.lower())

	#finds the number of positive and negative words, as well as the total number of words
	pos_word_count = cond_word_fd['pos'].N()
	#print "Rohit positive wrd count is ",pos_word_count
	neg_word_count = cond_word_fd['neg'].N()
	#print "Negative word count is ",neg_word_count
	total_word_count = pos_word_count + neg_word_count

	#builds dictionary of word scores based on chi-squared test
	word_scores = {}
	for word, freq in word_fd.iteritems():
		pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
		neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
		#print " score for word ",word ," score ",pos_score+neg_score
		
		word_scores[word] = pos_score + neg_score

	return word_scores


#posreview="Good company to work for.Amazing work envionment, nice culture,Great for freshsers."
#negreview="micro managment,silly team leads, should not be joined, I hate this company, worst place to work, worst, bad, nothing good about it, hell"
word_scores = create_word_scores(posreview,negreview)


def find_best_words(word_scores, number):
        print "number ",number
	best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
	best_words = set([w for w, s in best_vals])
	return best_words


def best_word_features(words):
	return dict([(word, True) for word in words if word in best_words])

#numbers of features to select
#numbers_to_test = [10, 100, 1000, 10000, 15000]
numbers_to_test = [40]
for num in numbers_to_test:
	print 'evaluating best %d word features' % (num)
	best_words = find_best_words(word_scores, num)
	evaluate_features(best_word_features)
        
