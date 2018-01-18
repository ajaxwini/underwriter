
# coding: utf-8

# # Importing Module
# - pandas for formatting or provides table
# - seabourn is wrapper over matplotlib

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[4]:

df = pd.read_csv("under.csv")
df["Contents"][0]


# In[27]:

df


# In[7]:

df['Offer']


# In[8]:

#sns.countplot(x='Is_Response',data=df)


# # Preprocessing 

# In[9]:

from nltk.corpus import stopwords,wordnet as wn
from nltk.tokenize import wordpunct_tokenize,sent_tokenize
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import re


# In[18]:

def rem_punt(doc):
    ans = re.sub('"|\\n|\(|\)|,|\.|[$!--+@#:]',' ',doc)
    ans = re.sub(' +',' ',ans)
    ans = ans.lower()
    return ans


# In[19]:

df['RmNoise'] = df['Contents'].apply(rem_punt)


# In[20]:

stop_word = set(stopwords.words('english'))


# In[21]:

def tokenize(document): 
    lemmy = []
    for sent in sent_tokenize(document):
        for token, tag in pos_tag(wordpunct_tokenize(sent)):
            #print(token,tag)
            if token in stop_word:
                 continue
            lemma = lemmatize(token, tag)
            lemmy.append(lemma)
    return lemmy

def lemmatize(token, tag):
    tag = {
          'N': wn.NOUN,
          'V': wn.VERB,
          'R': wn.ADV,
          'J': wn.ADJ
    }.get(tag[0], wn.NOUN)
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(token, tag)


# In[22]:

df['Lemmitize'] = df['RmNoise'].apply(tokenize)


# In[23]:

df['Lemmitize'][0]


# In[24]:

df.to_csv('Script2data.csv',index=False)


# In[25]:

df = pd.read_csv('Script2data.csv')


# # Statistical Modeling 

# In[26]:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer,LabelEncoder
from sklearn.metrics import accuracy_score,classification_report


# In[28]:

X = df.iloc[: ,-1]
y = df.iloc[:,-2]
lab_y = LabelEncoder()
y = lab_y.fit_transform(y)


# In[29]:

X_train,X_text,y_train,y_text = train_test_split(X,y,test_size = 0.1)


# In[30]:

vect = TfidfVectorizer(max_df=1.0, max_features=15000, min_df=0.02, use_idf=True , ngram_range=(1,3))


# In[7]:

#svd = TruncatedSVD(200,n_iter=7, random_state=42)
#lsa = make_pipeline(vect,svd, Normalizer(copy=False))


# # For test data

# In[21]:

df_test = pd.read_csv('Script2test.csv')
Xt = df_test.iloc[: ,[0,1,5]]


# # Xg boost

# In[31]:

from xgboost.sklearn import XGBClassifier
model = XGBClassifier(nthread=4,n_estimators=1000)


# # Navie Bayes

# In[23]:

#from sklearn.naive_bayes import GaussianNB,MultinomialNB
#model = GaussianNB()


# # ExtraTree Classifier

# In[24]:

#from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
#model = RandomForestClassifier(n_estimators=600,n_jobs=-1)


# # SVM Classifier
# 

# In[25]:

#from sklearn.svm import SVC
#model = SVC()


# # Logistic Regression 

# In[26]:

#from sklearn.linear_model import LinearRegression,SGDClassifier,LogisticRegression
#model = LogisticRegression()


# # Model Fitting

# In[33]:

clf = make_pipeline(vect,model)


# In[34]:

clf.fit(X_train,y_train)


# In[40]:

pd = clf.predict(X_test)

