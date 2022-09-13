import re
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow import keras
import requests
import urllib
from requests_html import HTML
from requests_html import HTMLSession
from urllib.request import urlopen
from bs4 import BeautifulSoup

training_round = 5

with open('C:/Users/taige/Desktop/Research/summer2022/week8/django/mysite/classifier/resource_data/version.txt') as f:
    output = [line.strip() for line in f.readlines()]

awards_ver = output[2]
edu_ver = output[5]
interest_ver = output[8]

amodel_add = output[0] + awards_ver + '.h5'
adata_add = output[1] + awards_ver + '.csv'
emodel_add = output[3] + awards_ver + '.h5'
edata_add = output[4] + awards_ver + '.csv'
imodel_add = output[6] + awards_ver + '.h5'
idata_add = output[7] + awards_ver + '.csv'

awards_model = keras.models.load_model(amodel_add, custom_objects={'KerasLayer':hub.KerasLayer})
edu_model =  keras.models.load_model(emodel_add, custom_objects={'KerasLayer':hub.KerasLayer})
interest_model = keras.models.load_model(imodel_add, custom_objects={'KerasLayer':hub.KerasLayer})

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|Prof)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu)"


def split_sen(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def process_paragraph(text):
    final_dict = {}
    background = []
    interest = []
    awards = []

    sens = split_sen(text)
    for sen in sens:
        if edu_model.predict([sen])[0][0] > 0.5:
            background.append(sen)
        
        if interest_model.predict([sen])[0][0] > 0.5:
            interest.append(sen)
        
        if awards_model.predict([sen])[0][0] > 0.5:
            awards.append(sen)

    final_dict["background"] = background
    final_dict["interest"] = interest
    final_dict["awards"] = awards

    return final_dict, len(sens)

def train_awd(new_sentences):
    global awards_ver, output
    df = pd.read_csv(adata_add, encoding = "ISO-8859-1", engine='python')

    for sen in new_sentences:
        df.loc[len(df.index)] = [1, sen] 

    df_related = df[df["Awards"] == 1]
    df_unrelated = df[df["Awards"] == 0]
    df_down = df_unrelated.sample(df_related.shape[0])
    df_balanced = pd.concat([df_down, df_related])
    X_train, X_test, y_train, y_test = train_test_split(df_balanced['Text'], df_balanced['Awards'], stratify=df_balanced['Awards'])
    
    awards_model.save('C:/Users/taige/Desktop/Research/summer2022/week8/django/mysite/classifier/resource_data/backups/amodel_' + awards_ver + '.h5')
    awards_model.fit(X_train, y_train, epochs=training_round)
    awards_ver = (str)((int)(awards_ver) + 1)
    awards_model.save('C:/Users/taige/Desktop/Research/summer2022/week8/django/mysite/classifier/resource_data/amodel_' + awards_ver + '.h5')

    df.to_csv('C:/Users/taige/Desktop/Research/summer2022/week8/django/mysite/classifier/resource_data/adata_' + awards_ver + '.csv')
    
    output[2] = awards_ver
    with open('C:/Users/taige/Desktop/Research/summer2022/week8/django/mysite/classifier/resource_data/version.txt', "w") as f:
        for item in output:
            f.write(str(item) + '\n')


def train_edu(new_sentences):
    global edu_ver, output
    df = pd.read_csv(edata_add, encoding = "ISO-8859-1", engine='python')

    for sen in new_sentences:
        df.loc[len(df.index)] = [1, sen] 

    df_related = df[df["Ed"] == 1]
    df_unrelated = df[df["Ed"] == 0] 
    df_down = df_unrelated.sample(df_related.shape[0])
    df_balanced = pd.concat([df_down, df_related])
    X_train, X_test, y_train, y_test = train_test_split(df_balanced['Text'], df_balanced['Ed'], stratify=df_balanced['Ed'])
    
    edu_model.save('C:/Users/taige/Desktop/Research/summer2022/week8/django/mysite/classifier/resource_data/backups/emodel_' + edu_ver + '.h5')
    edu_model.fit(X_train, y_train, epochs=training_round)
    edu_ver = (str)((int)(edu_ver) + 1)
    edu_model.save('C:/Users/taige/Desktop/Research/summer2022/week8/django/mysite/classifier/resource_data/emodel_' + edu_ver + '.h5')

    df.to_csv('C:/Users/taige/Desktop/Research/summer2022/week8/django/mysite/classifier/resource_data/edata_' + edu_ver + '.csv')
    
    output[5] = edu_ver
    with open('C:/Users/taige/Desktop/Research/summer2022/week8/django/mysite/classifier/resource_data/version.txt', "w") as f:
        for item in output:
            f.write(str(item) + '\n')


def train_int(new_sentences):
    global interest_ver, output
    df = pd.read_csv(adata_add, encoding = "ISO-8859-1", engine='python')

    for sen in new_sentences:
        df.loc[len(df.index)] = [1, sen] 

    df_related = df[df['Interest'] == 1]
    df_unrelated = df[df['Interest'] == 0]
    df_down = df_unrelated.sample(df_related.shape[0])
    df_balanced = pd.concat([df_down, df_related])
    X_train, X_test, y_train, y_test = train_test_split(df_balanced['Text'], df_balanced['Interest'], stratify=df_balanced['Interest'])
    
    interest_model.save('C:/Users/taige/Desktop/Research/summer2022/week8/django/mysite/classifier/resource_data/backups/imodel_' + interest_ver + '.h5')
    interest_model.fit(X_train, y_train, epochs=training_round)
    interest_ver = (str)((int)(interest_ver) + 1)
    interest_model.save('C:/Users/taige/Desktop/Research/summer2022/week8/django/mysite/classifier/resource_data/imodel_' + interest_ver + '.h5')

    df.to_csv('C:/Users/taige/Desktop/Research/summer2022/week8/django/mysite/classifier/resource_data/idata_' + interest_ver + '.csv')
    
    output[8] = interest_ver
    with open('C:/Users/taige/Desktop/Research/summer2022/week8/django/mysite/classifier/resource_data/version.txt', "w") as f:
        for item in output:
            f.write(str(item) + '\n')

def get_csv():
    df_a = pd.read_csv(adata_add, encoding = "ISO-8859-1", engine='python')
    df_e = pd.read_csv(edata_add, encoding = "ISO-8859-1", engine='python')
    df_i = pd.read_csv(adata_add, encoding = "ISO-8859-1", engine='python')
    return df_a, df_e, df_i

