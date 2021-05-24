import pandas as pd
import numpy as np
import nltk
#nltk.download('all')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import pickle
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import average_precision_score
#from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import time

# Set Random seed
np.random.seed(500)
start = time.time()
# Add the Data using pandas

Corpus = pd.read_csv(r"C:\Users\lasya\OneDrive\Desktop\major project\Major code\Sql-Injection\Sqli_dataset1.csv", encoding='latin-1')
#Corpus.SentimentText=Corpus.SentimentText.astype(str)

# Step - 1: Data Pre-processing - This will help in getting better results through the classification algorithms

# Step - 1a : Remove blank rows if any.
Corpus['Sentence'].dropna(inplace=True)

# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV


def text_preprocessing(text):
    # Step - 1b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    text = text.lower()

    # Step - 1c : Tokenization : In this each entry in the corpus will be broken into set of words
    text_words_list = word_tokenize(text)
    return str(text_words_list)
    # Step - 1d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
    # eclaring Empty List to store the words that follow the rules for this step
    """inal_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(text_words_list):
        # Below condition is to cheCock for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
    return str(Final_words)"""

Corpus.Sentence=Corpus.Sentence.astype(str)
Corpus['text_final'] = Corpus['Sentence'].map(text_preprocessing)

# Step - 2: Split the model into Train and Test Data set
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'], Corpus['label'],
                                                                    test_size=0.3)

# Step - 3: Label encode the target variable  - This is done to transform Categorical data of string type in the data set into numerical values
Encoder = LabelEncoder()
Encoder.fit(Train_Y)
Train_Y = Encoder.transform(Train_Y)
Test_Y = Encoder.transform(Test_Y)

# Step - 4: Vectorize the words by using TF-IDF Vectorizer - This is done to find how important a word in document is in comaprison to the corpus
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['text_final'])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

# Step - 5: Now we can run different algorithms to classify out data check for accuracy

# Classifier - Algorithm - Naive Bayes
# fit the training dataset on the classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf, Train_Y)

# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)
average_precision = average_precision_score(Test_Y, predictions_NB)

# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, Test_Y) * 100)
print('Naive Bayes Average precision-recall score:-> {0:0.2f}'.format(
      average_precision))
disp = plot_precision_recall_curve(Naive, Test_X_Tfidf, Test_Y)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))
print('Classification report:')
print(classification_report(Test_Y, predictions_NB))
# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf, Train_Y)

# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
average_precision = average_precision_score(Test_Y, predictions_SVM)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y) * 100)
print('SVM Average precision-recall score:-> {0:0.2f}'.format(
      average_precision))



GB = GradientBoostingClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(GB, Train_X_Tfidf, Train_Y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
GB = GradientBoostingClassifier()
GB.fit(Train_X_Tfidf, Train_Y)
predictions_GB = GB.predict(Test_X_Tfidf)
average_precision = average_precision_score(Test_Y, predictions_GB)
print('GB Accuracy Score-> %f' % (mean(n_scores)*100))
print('GB Average precision-recall score:-> {0:0.2f}'.format(
      average_precision))
# fit the GB on the whole dataset


# Saving Encdoer, TFIDF Vectorizer and the trained model for future infrerencing/prediction

# saving encoder to disk
filename = 'labelencoder_fitted.pkl'
pickle.dump(Encoder, open(filename, 'wb'))

# saving TFIDF Vectorizer to disk
filename = 'Tfidf_vect_fitted.pkl'
pickle.dump(Tfidf_vect, open(filename, 'wb'))

# saving the both GBs to disk
filename = 'svm_trained_model.sav'
pickle.dump(SVM, open(filename, 'wb'))

filename = 'nb_trained_model.sav'
pickle.dump(Naive, open(filename, 'wb'))
filename = 'gb_trained_model.sav'
pickle.dump(GB, open(filename, 'wb'))
print("Files saved to disk! Proceed to inference.py")
