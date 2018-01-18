from textblob.classifiers import NaiveBayesClassifier
cl = NaiveBayesClassifier(train)

'''
JSON format
[
    {"text": "I love this sandwich.", "label": "pos"},
    {"text": "This is an amazing place!", "label": "pos"},
    {"text": "I do not like this restaurant", "label": "neg"}
]

'''

with open('train.json', 'r') as fp:
	cl = NaiveBayesClassifier(fp, format="json")

cl.classify("This is an amazing library!")

#updating classifier

new_data = [('She is my best friend.', 'pos'),("I'm happy to have a new friend.", 'pos'),("Stay thirsty, my friend.", 'pos'),("He ain't from around here.", 'neg')]
cl.update(new_data)

cl.accuracy(test)


def end_word_extractor(document):
     tokens = document.split()
     first_word, last_word = tokens[0], tokens[-1]
     feats = {}
     feats["first({0})".format(first_word)] = True
     feats["last({0})".format(last_word)] = False
     return feats
features = end_word_extractor("I feel happy")
assert features == {'last(happy)': False, 'first(I)': True}

cl2 = NaiveBayesClassifier(test, feature_extractor=end_word_extractor)
blob = TextBlob("I'm excited to try my new classifier.", classifier=cl2)
blob.classify()

