#!/usr/bin/env python
# coding: utf-8

# !pip install markcrowe


import data_analytics.github as github
github.display_jupyter_notebook_header("markcrowe-com", "machine-learning", 
                                       "nlp-text-analytics/nlp-text-analytics-example.ipynb")


# # Text Analysis Example

# ### Setup

import matplotlib.pyplot
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import nltk


# These commands are required to run once to download text files to support the nltk package.  Without them some of Python code will have exceptions.

nltk.download('averaged_perceptron_tagger')
nltk.download('names')
nltk.download('punkt')
nltk.download('stopwords') # required for from nltk.corpus import stopwords
nltk.download('wordnet')


# ## Tokenization Example
# 
# ### Sentence Tokenization
# Taking the text "Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome. 
# The sky is pinkish-blue. You shouldn't eat cardboard".  The functions 'sent_tokenize' will break up the text into it's sentences using punctuation marks such as \[. ? ! \].

text = """Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome. 
The sky is pinkish-blue. You shouldn't eat cardboard"""
sentences = sent_tokenize(text)
print("sentences type is:", type(sentences))
print(sentences)
print("There are ", len(sentences), "items in sentences")


# ### Word Tokenization
# Taking the text "Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome. 
# The sky is pinkish-blue. You shouldn't eat cardboard".  The function 'word_tokenize' will break up the text into it's word using spaces and other punctuation marks such as \[. ? ! \].

text_tokens = word_tokenize(text)

print("text_tokens type is:", type(text_tokens))
print(text_tokens)
print("There are ", len(text_tokens), "items in text_tokens")


# # Frequency Distribution
# How many times do the words occur?

frequency_distribution = FreqDist(text_tokens)

print(frequency_distribution)
print(len(text_tokens))


# So what does that mean.  There are 25 unique words and all the words are used a total of 30 times.

TOP_COMMON_COUNT = 5
frequency_distribution.most_common(TOP_COMMON_COUNT)


# If we increased 'TOP_COMMON_COUNT' to 6 or 10 every word after ('Hello', 1) would also have a value of 1.

# ### Frequency Distribution Plot

frequency_distribution.plot(30, cumulative = False)
pyplot.show()


# Natural Language Processing with PythonNatural language processing (nlp) is a research field that presents many challenges such as natural language understanding.  
# 
# Text may contain stop words like 'the', 'is', 'are'. Stop words can be filtered from the text to be processed. There is no universal list of stop words in nlp research, however the nltk module contains a list of stop words.  
# 
# For more information c.f. <https://pythonspot.com/nltk-stop-words/>

stop_words = set(stopwords.words("english"))

print("stop_words type is:", type(text_tokens))
print(stop_words)
print("There are ", len(stop_words), "items in stop_words")


# ## Removing Stop Words

def filter_stop_words(word_list: list, stop_words: list) -> list:
    words_of_interest = []
    for word in word_list:
        if word not in stop_words:
            words_of_interest.append(word)
    return words_of_interest


words_of_interest = filter_stop_words(text_tokens, stop_words)

# Display the tokenize and filtered sentences
print("Tokenized Sentences:", text_tokens)
print("Filterd Sentences:", words_of_interest)


# ## Stemming
# What is stemming?
# Stemming is the process of reducing inflection in words to their root forms such as mapping a group of words to the same stem even if the stem itself is not a valid word in the Language.  
# c.f. <https://www.datacamp.com/community/tutorials/stemming-lemmatization-python>

# #### Using the 'PorterStemmer'

# Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

# Create an initialise an object ps by calling a method PorterStemmer()
porter_stemmer = PorterStemmer()

# Initialise an array 'stemmed_words'
stemmed_words = []

# Store all the words into an array 'stemmed_words'

for word in words_of_interest:
    stemmed_words.append(porter_stemmer.stem(word))

# Display the stemmed_words
print("words of interest:", words_of_interest)
print()
print("Stemmed words:", stemmed_words)


# ### Word Net Lemmatizer
# what is Lemmatizer?

# Lexicon Normalization
# Performing stemming and Lemmatization

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

# Create an initialise an object 'lem' by calling a method WordNetLemmatizer()
lemmatizer = WordNetLemmatizer()

# Create an initialise an object 'stem' by calling a method PorterStemmer()
porter_stemmer = PorterStemmer()

# Store the word 'flying' into string 'word'
word = "flying"

# Display the Lemmatized and stemmed words
print("Word:", word)
print("Lemmatized Word:", lemmatizer.lemmatize(word, "v"))
print("Stemmed Word:", porter_stemmer.stem(word))


# Store a sentence into an array 'sent'
sent = "Albert Einstein was born in Ulm, Germany in 1879."


# Loading NLTK
import nltk
from nltk.tokenize import sent_tokenize

# Store the 'sent' into an array 'tokens'
tokens = nltk.word_tokenize(sent)

# Display tokens
print(tokens)


# Display the parts of speech (pos) for the words in the sentence
nltk.pos_tag(tokens)


# # Topic Modelling

from sklearn.datasets import fetch_20newsgroups

# Declare the categories
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

# Fetch the 20 news group by calling a method fetch_20newsgroups()
groups = fetch_20newsgroups(subset='all', categories=categories)

# Create and declare a function is_letter_only()
def is_letter_only(word):
    for char in word:
        if not char.isalpha():
            return False
    return True


from nltk.corpus import names
all_names = set(names.words())
all_names


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Declare and create an empty array
data_cleaned = []

# Create a loop for all documents
for doc in groups.data:
    doc = doc.lower()
    doc_cleaned = ' '.join(lemmatizer.lemmatize(word) for word in doc.split() if is_letter_only(word) and word not in all_names)
    data_cleaned.append(doc_cleaned)


from sklearn.feature_extraction.text import CountVectorizer

# Declare and create an array (count_vector) by calling a method CountVectorizer()
count_vector = CountVectorizer(stop_words = "english", max_features = None, max_df = 0.5, min_df = 2)
data = count_vector.fit_transform(data_cleaned)


# Discovering underlying topics in newsgroups
# A topic model is a type of statistical model for discovering the probability distributions of words linked to the topic. The topic in topic modeling does not exactly match the dictionary definition, but corresponds to a nebulous statistical concept, an abstraction occurs in a collection of documents.
# 
# When we read a document, we expect certain words appearing in the title or the body of the text to capture the semantic context of the document. An article about Python programming will have words such as class and function, while a story about snakes will have words such as eggs and afraid. Documents usually have multiple topics; for instance, this recipe is about three things, topic modeling, non-negative matrix factorization, and latent Dirichlet allocation, which we will discuss shortly. We can therefore define an additive model for topics by assigning different weights to topics.
# 
# Topic modeling is widely used for mining hidden semantic structures in given text data. There are two popular topic modeling algorithms—non-negative matrix factorization, and latent Dirichlet allocation. We will go through both of these in the next two sections.

# # Topic modeling using LDA
# Let's explore another popular topic modeling algorithm, latent Dirichlet allocation (LDA). LDA is a generative probabilistic graphical model that explains each input document by means of a mixture of topics with certain probabilities. Again, topic in topic modeling means a collection of words with a certain connection. In other words, LDA basically deals with two probability values, P(term | topic) and P(topic | document). This can be difficult to understand at the beginning. So, let's start from the bottom, the end result of an LDA model.
# 
# Let's take a look at the following set of documents:

# * <b>Document 1: This restaurant is famous for fish and chips.</b>
# * <b>Document 2: I had fish and rice for lunch.</b>
# * <b>Document 3: My sister bought me a cute kitten.</b>
# * <b>Document 4: Some research shows eating too much rice is bad.</b>
# * <b>Document 5: I always forget to feed fish to my cat.</b>

# * Topic 1: 30% fish, 20% chip, 30% rice, 10% lunch, 10% restaurant (which we can interpret Topic 1 to be food related)
# * Topic 2: 40% cute, 40% cat, 10% fish, 10% feed (which we can interpret Topic 1 to be about pet)

# * Documents 1: 85% Topic 1, 15% Topic 2
# * Documents 2: 88% Topic 1, 12% Topic 2
# * Documents 3: 100% Topic 2
# * Documents 4: 100% Topic 1
# * Documents 5: 33% Topic 1, 67% Topic 2

# LDA is trained in a generative manner, where it tries to abstract from the documents a set of hidden topics that are likely to generate a certain collection of words.
# 
# With all this in mind, let's see LDA in action. The LDA model is also included in scikit-learn:

from sklearn.decomposition import LatentDirichletAllocation

# Declare and initialise a variable t
t = 20

# Declare and initialise an object 'lda' by calling a method LatentDirichletAllocation()
lda = LatentDirichletAllocation(n_components = t, learning_method = 'batch', random_state = 42)

# Train the model
lda.fit(data)

# Print all lda components
print(lda.components_)

# Get all feature names
terms = count_vector.get_feature_names()


# Again, we specify 20 topics (n_components). The key parameters of the model are included in the following table:

from IPython.display import Image
Image(filename =r'Im1.png', width = 520, height = 350)


# For the input data to LDA, remember that LDA only takes in term counts as it is a probabilistic graphical model. This is unlike NMF, which can work with both the term count matrix and the tf-idf matrix as long as they are non-negative data. Again, we use the term matrix defined previously as input to the lda model:

for topic_idx, topic in enumerate(lda.components_):
        print("Topic {}:" .format(topic_idx))
        print(" ".join([terms[i] for i in topic.argsort()[-10:]]))


# ## Reference:
# * Chapter 8, Python Machine Learning, Sebastian Raschka, Packt Publishing, 2015.
