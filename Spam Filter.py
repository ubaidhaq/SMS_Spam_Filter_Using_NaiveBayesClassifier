import nltk
from nltk.corpus import stopwords
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix

plt.style.use('ggplot')
sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['patch.force_edgecolor'] = True

#Data Processing 
data = pd.read_csv('C://Users//Admin//Desktop//SMS Spam Filter//spam.csv', encoding='latin-1')
data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
data = data.rename(columns={'v1': 'class','v2': 'text'})
data.head()

data["class"].value_counts().plot(kind = 'pie', explode = [0, 0.1], figsize = (6, 6), autopct = '%1.1f%%', shadow = True)
plt.ylabel("Spam vs Ham")
plt.legend(["Ham", "Spam"])
plt.show()

data.groupby('class').describe()

#Plotting the length of text of spam & ham messages  

data['length'] = data['text'].apply(len)
data.hist(column='length',by='class',bins=50, figsize=(15,6))

#Function of cleaning the data
def clean_text(text):
    
    punc = [char for char in text if char not in string.punctuation]
    punc = ''.join(punc)
    
    wordss = [word for word in punc.split() if word.lower() not in stopwords.words('english')]
    
    return wordss

data['text'].apply(process_text).head()
data.head()

#Splitting into training and testing data. Training data 70%
x_train, x_test, y_train, y_test = train_test_split(data['text'],data['class'],test_size=0.3)

#Creating the Model
pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=clean_text)),
    ('tfidf',TfidfTransformer()), 
    ('classifier',MultinomialNB()) # training on TF-IDF vectors with Naive Bayes classifier
])

#Training the model
pipeline.fit(x_train,y_train)

#Testing
predictions = pipeline.predict(x_test)
print(classification_report(y_test,predictions))

#Confusion Matrix 
sns.heatmap(confusion_matrix(y_test,predictions),annot=True)