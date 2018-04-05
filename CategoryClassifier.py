#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.base import TransformerMixin
from TextNormalizer import TextNormalizer
from sklearn.pipeline import Pipeline
from nltk.tokenize import TreebankWordTokenizer
from sklearn.linear_model import RidgeClassifierCV
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from polyglot.detect import Detector
import numpy as np

class LanguageNotAvailableError(Exception):
    pass

class LanguageNotRecognisedError(Exception):
    pass 

class KeyWordClassifier(TransformerMixin):
    def __init__(self, categories, keyword_folder = ''):
        self.__categories = categories
        self.__keyword_folder = keyword_folder
        self.__freqs = {}
        for cat in categories:
            self.__freqs[cat] = pd.read_excel('keywords/' + keyword_folder + cat + '_topwords.xlsx')
        
        
    def predict(self, X, y=None, **fit_params):
        res = []
        def check4word(w, freqs):
            if w in freqs.index:
                return freqs.loc[w].tolist()[0]
            else:
                return 0
        tokenizer = TreebankWordTokenizer()
        classes2nums = dict(zip(self.__categories,range(len(self.__categories))))    
        for text in X:
            tokens    = tokenizer.tokenize(text)
            cat_freqs = pd.DataFrame(columns = tokens)
            for w in tokens:
                for cat in self.__categories:
                    cat_freqs.loc[cat,w] = check4word(w,self.__freqs[cat])        
            if cat_freqs.apply(sum).sum()==0:
                res.append(0)
            else:
                res.append(classes2nums[cat_freqs.apply(sum,axis = 1).idxmax()])
        return res

    def fit_predict(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.predict(X)

    def fit(self, X, y=None, **fit_params):
        return self
    
    
class CategoryClassifier():
    
    def __init__(self, model = 'keywords'):
        
        def prepare_train_data():
            classes_nums = {
                'Balance':1,
                'Graphics':2,
                'Bug':3,
                'Advertising':4,
                'Monetization':5,
                'Other':0
            }
            labeled4 = pd.read_excel('temp data/for_labeling 4.xlsx').loc[:,['Review', 'Label']]
            labeled1 = pd.read_excel('temp data/for_labeling 1.xlsx').loc[:,['Review', 'Label']]
            labeled2 = pd.read_excel('temp data/for_labeling 2.xlsx').loc[:,['Review', 'Label']]
            labeled2 = labeled2[(labeled2.Label!='?')&(labeled2.Label!='-')]
            labeled1['label_num'] = labeled1.Label.map(classes_nums)
            labeled4['label_num'] = labeled4.Label.map(classes_nums)
            labeled2['label_num'] = labeled2.Label
            labeled = pd.concat([labeled4, labeled2, labeled1], axis = 0)
            labeled = labeled.dropna(axis = 0)
            labeled.label_num = labeled.label_num.apply(int)
            feats = labeled.Review
            labels = labeled.label_num
            return feats,labels
        
        self.__tn = TextNormalizer()
        if model == 'keywords':
            categories = [
                    'other',
                    'balance',
                    'graphics',
                    'bug',
                    'ads',
                    'money'
                    ]
            self.__model = Pipeline([('text_cleaner', self.__tn), ('classifier', KeyWordClassifier(categories))])
            self.__model.fit(X = [])
        elif model == 'ridge_new':
            self.__model = Pipeline([('text_cleaner', self.__tn), 
                                     ('vectorizer',CountVectorizer()),
                                     ('classifier', RidgeClassifierCV())])
            self.__model.set_params(vectorizer__ngram_range = (1,3),
                                    vectorizer__analyzer = 'word',
                                    vectorizer__stop_words = 'english',
                                    vectorizer__max_features = 5000,
                                    vectorizer__min_df = 2,
                                    vectorizer__max_df = 0.95,
                                    #vectorizer__vocabulary = vocab,
                                    classifier__class_weight = 'balanced')
            feats,labels = prepare_train_data()
            self.__model = self.__model.fit(feats, labels)
        elif model == 'ridge_load':
            with open('ridge_new.pkl', 'rb') as f:
                self.__model = pickle.load(f)
        elif model == 'logit_load':
            with open('logit.pkl', 'rb') as f:
                self.__model = pickle.load(f)            
        subcategories = [
                'other',
                'combat',
                'gameplay',
                'matchmaking'
                ]
            
        self.__balance_model = Pipeline([('text_cleaner', self.__tn), ('classifier', KeyWordClassifier(keyword_folder = 'subcats/balance/', categories = subcategories))])
        self.__balance_model.fit(X = [])            
    def __lang_detect(self, text, threshold = 0.9):
        detector = Detector(text,quiet = True)
        if detector.language.confidence>threshold:
            return detector.language.code
        else:
            raise LanguageNotRecognisedError('Could not recognize the language')    
    def predict(self, comments):
        balance_subcats = {
                1:'Combat Balance',
                2:'Gameplay Balance',
                3:'Matchmaking',
                0:'Undefined'
                }
        res = []
        for comment in comments:
            #res.append(self.__model.predict_proba(comment)[0])
            classes_labels = {
                1:'Balance',
                2:'Graphics',
                3:'Bug',
                4:'Advertising',
                5:'Monetization',
                0:'Irrelevant/Other'
                }
            subclasses_labels = {
                    1: balance_subcats[self.__balance_model.predict(comment)[0]],
                    2: 'not trained',
                    3: 'not trained',
                    4: 'not trained',
                    5: 'not trained',
                    0: 'not trained'
                    }
            predicted_probs = self.__model.predict_proba(comment)[0]
            prediction_root = classes_labels[np.argmax(predicted_probs)]
            prediction_sub = subclasses_labels[np.argmax(predicted_probs)]
            output = {
                    'comment':comment,
                    'lang':self.__lang_detect(comment[0]),
                    'root_category': prediction_root,
                    'probability':np.max(predicted_probs),
                    'subcategory': prediction_sub
                    }
            res.append(output)
            #print(classes_labels[self.__model.predict(comment)[0]])
        return res[0]
if __name__ == '__main__':
    cat_classifier = CategoryClassifier(model = 'logit_load')
    while True:
        review = input()
        print('\n')
        print(cat_classifier.predict([[review]]))