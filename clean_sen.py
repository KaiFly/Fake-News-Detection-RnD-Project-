# # This funct use for predict_news.py
# Clean the text
import re
import string
import nltk
from nltk.stem import SnowballStemmer, PorterStemmer
from nltk.tokenize import word_tokenize
eng_stemmer2 = PorterStemmer()
eng_stemmer = SnowballStemmer('english')
#nltk.corpus.stopwords.words('english').remove('not')
stopwords = set(nltk.corpus.stopwords.words('english'))
stopwords.remove('not')

def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed
#tokens = data.iloc[3, 0]
def process_data(tokens):
    tokens = re.sub(r"what’s", "what is ", tokens)
    tokens = re.sub(r"\’s", " ", tokens)
    tokens = re.sub(r"\'ve", " have ", tokens)
    tokens = re.sub(r"can’t", "cannot ", tokens)
    tokens = re.sub(r"don’t", "do not ", tokens)
    tokens = re.sub(r"doesn’t", "do not ", tokens)
    tokens = re.sub(r"n’t", " not ", tokens)
    tokens = re.sub(r"i’m", "i am ", tokens)
    tokens = re.sub(r"\'re", " are ", tokens)
    tokens = re.sub(r"\’d", " would ", tokens)
    tokens = re.sub(r"\’ll", " will ", tokens)
    tokens = re.sub(r"what's", "what is ", tokens)
    tokens = re.sub(r"\'s", " ", tokens)
    tokens = re.sub(r"\'ve", " have ", tokens)
    tokens = re.sub(r"can't", "cannot ", tokens)
    tokens = re.sub(r"don't", "do not ", tokens)
    tokens = re.sub(r"doesn't", "do not ", tokens)
    tokens = re.sub(r"n't", " not ", tokens)
    tokens = re.sub(r"i'm", "i am ", tokens)
    tokens = re.sub(r"\'re", " are ", tokens)
    tokens = re.sub(r"\'d", " would ", tokens)
    tokens = re.sub(r"\'ll", " will ", tokens)
    
    tokens = re.sub(r'[\n\t]', ' ', tokens)
    tokens = re.sub(r"\0s", "0", tokens)
    tokens = re.sub(r"e - mail", "email", tokens)
    
    tokens = re.sub('[' + string.punctuation  + ']', ' ', tokens)
    tokens = re.sub(r'[0-9]+', ' ', tokens)
    
    tokens = re.sub("—", " ", tokens)
    tokens = re.sub("’", " ", tokens)
    tokens = re.sub("„", " ", tokens)
    tokens = re.sub("“", " ", tokens)
    tokens = re.sub("”", " ", tokens)
    tokens = tokens.lower()
    tokens = tokens.split()
    tokens = [w for w in tokens if w not in stopwords and len(w) >= 3 ]

    tokens_stemmed = tokens
    tokens_stemmed = stem_tokens(tokens, eng_stemmer)
    tokens_stemmed = ' '.join(tokens_stemmed)
    return tokens_stemmed