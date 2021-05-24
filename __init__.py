from flask import Flask, render_template, request, redirect, url_for, flash
from flask_mysqldb import MySQL
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
import pickle
import ast
from flask import request, jsonify

import nltk
#nltk.download('all')

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

app = Flask(__name__)
# Loading Label encoderr
labelencode = pickle.load(open(r'C:\Users\lasya\OneDrive\Desktop\major project\Major code\Webapp\Imp\sqli model\labelencoder_fitted.pkl', 'rb'),encoding="latin1")
labelencode1 = pickle.load(open(r'C:\Users\lasya\OneDrive\Desktop\major project\Major code\Webapp\Imp\xss model\labelencoder_fitted.pkl', 'rb'),encoding="latin1")
# Loading TF-IDF Vectorizer-
Tfidf_vect = pickle.load(open(r'C:\Users\lasya\OneDrive\Desktop\major project\Major code\Webapp\Imp\sqli model\Tfidf_vect_fitted.pkl', 'rb'),encoding="latin1")
Tfidf_vect1 = pickle.load(open(r'C:\Users\lasya\OneDrive\Desktop\major project\Major code\Webapp\Imp\xss model\Tfidf_vect_fitted.pkl', 'rb'),encoding="latin1")
# Loading models
SVM = pickle.load(open(r'C:\Users\lasya\OneDrive\Desktop\major project\Major code\Webapp\Imp\sqli model\svm_trained_model.sav', 'rb'),encoding="latin1")
Naive = pickle.load(open(r'C:\Users\lasya\OneDrive\Desktop\major project\Major code\Webapp\Imp\sqli model\nb_trained_model.sav', 'rb'),encoding="latin1")
GB = pickle.load(open(r'C:\Users\lasya\OneDrive\Desktop\major project\Major code\Webapp\Imp\sqli model\gb_trained_model.sav', 'rb'),encoding="latin1")
SVM1 = pickle.load(open(r'C:\Users\lasya\OneDrive\Desktop\major project\Major code\Webapp\Imp\xss model\svm_trained_modelxss.sav', 'rb'),encoding="latin1")
Naive1 = pickle.load(open(r'C:\Users\lasya\OneDrive\Desktop\major project\Major code\Webapp\Imp\xss model\nb_trained_modelxss.sav', 'rb'),encoding="latin1")
GB1 = pickle.load(open(r'C:\Users\lasya\OneDrive\Desktop\major project\Major code\Webapp\Imp\xss model\gb_trained_modelxss.sav', 'rb'),encoding="latin1")
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV


app.secret_key = 'many random bytes'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'crud'

mysql = MySQL(app)



@app.route('/')
def Index():
    cur = mysql.connection.cursor()
    cur.execute("SELECT  * FROM students")
    data = cur.fetchall()
    cur.close()




    return render_template('index2.html', students=data )



@app.route('/insert', methods = ['POST'])
def insert():

    if request.method == "POST":
        
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        sample_text_processed1 = text_preprocessing(name)
        sample_text_processed2 = text_preprocessing(email)
        sample_text_processed3 = text_preprocessing(phone)
        vect1= Tfidf_vect.transform([sample_text_processed1])
        vect2= Tfidf_vect.transform([sample_text_processed2])
        vect3= Tfidf_vect.transform([sample_text_processed3])
        my_prediction1 = SVM.predict(vect1)
        my_prediction2 = SVM.predict(vect2)
        my_prediction3 = SVM.predict(vect3)
        vect1= Tfidf_vect1.transform([sample_text_processed1])
        vect2= Tfidf_vect1.transform([sample_text_processed2])
        vect3= Tfidf_vect1.transform([sample_text_processed3])
        my_prediction4 = SVM1.predict(vect1)  
        my_prediction5 = SVM1.predict(vect2)
        my_prediction6= SVM1.predict(vect3)
        if my_prediction1==1 or my_prediction2==1 or my_prediction3==1 :
            return render_template('result.html', prediction=1)
        elif my_prediction4==1 or my_prediction5==1 or my_prediction6==1:
             return render_template('result.html', prediction=0)
        else:
            flash("Data Inserted Successfully")
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO students (name, email, phone) VALUES (%s, %s, %s)", (name, email, phone))
            mysql.connection.commit()
            return redirect(url_for('Index'))
def text_preprocessing(text):
    # Step - 1b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    text = text.lower()

    # Step - 1c : Tokenization : In this each entry in the corpus will be broken into set of words
    text_words_list = word_tokenize(text)

    # Step - 1d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    #return str(word_Lemmatized) # pos_tag funct)ion below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(text_words_list):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
    return str(Final_words)




@app.route('/delete/<string:id_data>', methods = ['GET'])
def delete(id_data):
    flash("Record Has Been Deleted Successfully")
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM students WHERE id=%s", (id_data,))
    mysql.connection.commit()
    return redirect(url_for('Index'))





@app.route('/update',methods=['POST', 'GET'])
def update():
    if request.method == 'POST':
        id_data = request.form['id']
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        #pickle.dump(name)
        #return render_template('result.html', prediction=name)
        sample_text_processed1 = text_preprocessing(name)
        sample_text_processed2 = text_preprocessing(email)
        sample_text_processed3 = text_preprocessing(phone)
        vect1= Tfidf_vect.transform([sample_text_processed1])
        vect2= Tfidf_vect.transform([sample_text_processed2])
        vect3= Tfidf_vect.transform([sample_text_processed3])
        my_prediction1 = SVM.predict(vect1)
        my_prediction2 = SVM.predict(vect2)
        my_prediction3 = SVM.predict(vect3)
        vect1= Tfidf_vect1.transform([sample_text_processed1])
        vect2= Tfidf_vect1.transform([sample_text_processed2])
        vect3= Tfidf_vect1.transform([sample_text_processed3])
        my_prediction4 = SVM1.predict(vect1)  
        my_prediction5 = SVM1.predict(vect2)
        my_prediction6= SVM1.predict(vect3)
        #return render_template('result.html', prediction=my_prediction1)
        if my_prediction1==1 or my_prediction2==1 or my_prediction3==1:
            return render_template('result.html', prediction=1)
        elif my_prediction4==1 or my_prediction5==1 or my_prediction6==1:
            return render_template('result.html', prediction=0)
        else:
            flash("Data Updated Successfully")
            cur = mysql.connection.cursor()
            cur.execute("UPDATE students SET name=%s, email=%s, phone=%s WHERE id=%s", (name, email, phone, id_data))
            mysql.connection.commit()
            return redirect(url_for('Index'))
def text_preprocessing(text):
    # Step - 1b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    text = text.lower()

    # Step - 1c : Tokenization : In this each entry in the corpus will be broken into set of words
    text_words_list = word_tokenize(text)

    # Step - 1d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    #return str(word_Lemmatized) # pos_tag funct)ion below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(text_words_list):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
    return str(Final_words)
    
@app.route("/get_my_ip", methods=["POST"])
def get_my_ip():
    ip=request.environ['REMOTE_ADDR']
    return render_template('result.html', prediction=ip)






if __name__ == "__main__":
    app.run(debug=True)
