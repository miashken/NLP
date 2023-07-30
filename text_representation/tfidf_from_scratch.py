"""
@author: Michal Ashkenazi
"""
import math
import numpy          as np
import nltk
from nltk.corpus      import stopwords
from nltk.stem        import PorterStemmer
from nltk.tokenize    import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')


class TfidfVectorizer:
    """
    Term Frequency - Inverse Document Frequency
    """
    def __init__(self):
        pass
    
    def calc_tf(self, term, document):
        """
        calculate the term frequency of term in a document.
        
        :parm term:       term (string)
        :param document:  document (list of words)
        :return tf of term in doc (float)
        """
        return document.count(term) / len(document)

    def calc_idf(self, term, documents):
        """
        calculate the inverse document frequency of a term in a collection of documents.
        
        :param term:      term (string)
        :param documents: list of documents (list)
        :return idf of term in docs (float)
        """
        doc_frequency = sum(1 for doc in documents if term in doc)
        return math.log((1 + len(documents)) / (1 + doc_frequency))

    def fit(self, documents):
        """
        calculate IDF scores for all the terms in the documents.
        
        :param documents: list of documents (list)
        """
        self.tfidf_model          = {}
        self.tfidf_model['terms'] = list(set(term for doc in documents for term in doc))
        self.tfidf_model['idf']   = {}

        for term in self.tfidf_model['terms']:
            self.tfidf_model['idf'][term] = self.calc_idf(term, documents)

    def transform(self, doc):
        """
        calculate the TF-IDF scores for all terms in a document.
        
        :param doc:  document to transform (list of words)
        :return tf-idf vector for a document (np array)
        """
        tfidf_vector = []

        for term in self.tfidf_model['terms']:
            tf  = self.calc_tf(term, doc)
            idf = self.tfidf_model['idf'][term]
            tfidf_vector.append(tf * idf)

        return np.asarray(tfidf_vector)
    
def preprocess(document):
    # Tokenize the document into individual words
    tokens = word_tokenize(document.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    # Apply stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    
    # Return the preprocessed document as a string
    preprocessed_document = ' '.join(stemmed_tokens)
    return preprocessed_document

if __name__ == "__main__":
    documents = [ "The sun is shining today.",
              "I enjoy going for a walk in the sun.",
              "It's a beautiful day outside.",
              "I prefer to stay indoors and read a book.",
              "Reading is my favorite hobby.",
              "I don't like the heat of the sun." ]

    # Preprocess the documents
    preprocessed_documents = [preprocess(doc) for doc in documents]

    # Define and Train the Model
    tfidf_vect = TfidfVectorizer()
    tfidf_vect.fit(preprocessed_documents)

    # Calculate TF-IDF vector for a new document
    doc           = "I like to go swimming in the pool."
    processed_doc = preprocess(doc)

    print(tfidf_vect.transform(processed_doc))

    #[0.06593489 0.02433958 0.02945346 0.         0.08918925 0.
    # 0.         0.06593489 0.         0.01622639 0.         0.
    # 0.         0.00811319 0.06593489 0.         0.         0.
    # 0.         0.         0.         0.         0.         0.        ]
