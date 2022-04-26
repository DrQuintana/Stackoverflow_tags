import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import nltk
import spacy
#import en_core_web_sm

import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
#from langdetect import detect
#from deep_translator import GoogleTranslator
import nltk
from nltk.corpus import stopwords
import spacy

def remove_pos(nlp, x, pos_list):

    # Test of language detection
    #lang = detect(x)
    #if(lang != "en"):
        # Deep translate if not in English
     #   x = GoogleTranslator(source='auto', target='en').translate(x)
    
    doc = nlp(x)
    list_text_row = []
    for token in doc:
        if(token.pos_ in pos_list):
            list_text_row.append(token.text)
    join_text_row = " ".join(list_text_row)
    join_text_row = join_text_row.lower().replace("c #", "c#")
    return join_text_row
def text_cleaner(x, nlp, pos_list, lang="english"):

    # Remove POS not in "NOUN", "PROPN"
    x = remove_pos(nlp, x, pos_list)
    # Case normalization
    x = x.lower()
    # Remove unicode characters
    x = x.encode("ascii", "ignore").decode()
    # Remove English contractions
    x = re.sub("\'\w+", '', x)
    # Remove ponctuation but not # (for C# for example)
    x = re.sub('[^\\w\\s#]', '', x)
    # Remove links
    x = re.sub(r'http*\S+', '', x)
    # Remove numbers
    x = re.sub(r'\w*\d+\w*', '', x)
    # Remove extra spaces
    x = re.sub('\s+', ' ', x)
        
    # Tokenization
    x = nltk.tokenize.word_tokenize(x)
    # List of stop words in select language from NLTK
    stop_words = stopwords.words(lang)
    # Remove stop words
    x = [word for word in x if word not in stop_words 
         and len(word)>2]
    # Lemmatizer
    wn = nltk.WordNetLemmatizer()
    x = [wn.lemmatize(word) for word in x]
    
    # Return cleaned text
    return x

def predict_tags(article):

    
    #load models
    multibin_model = pickle.load(open('multilabel_binarizer.pkl', 'rb'))
    model = pickle.load(open('my_model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))



    #clean article

    #nlp = en_core_web_sm.load(exclude=['tok2vec', 'ner', 'parser', 'attribute_ruler', 'lemmatizer'])
    nlp = spacy.load('en_core_web_sm')
    pos_list = ["NOUN","PROPN"]
    rawtext = article
    cleaned_question =text_cleaner(rawtext, nlp, pos_list, "english")
    #vektorize
    print(cleaned_question)
    X_tfidf = vectorizer.transform(cleaned_question)
    
    predict = model.predict(X_tfidf)
    predict_probas = model.predict_proba(X_tfidf)[0]

    tags_predict = multibin_model.inverse_transform(predict)

    # DataFrame of probas
    df_predict_probas = pd.DataFrame(columns=['Tags', 'Probas'])
    df_predict_probas['Tags'] = multibin_model.classes_
    df_predict_probas['Probas'] = predict_probas.reshape(-1)
    # Select probas > 33%
    df_predict_probas = df_predict_probas[df_predict_probas['Probas']>=0.33].sort_values('Probas', ascending=False)
          
    # Results
    results = {}
    results['Predicted_Tags'] = tags_predict
    results['Predicted_Tags_Probabilities'] = df_predict_probas.set_index('Tags')['Probas'].to_dict()
    
    return results
    


#predict_tags(article)