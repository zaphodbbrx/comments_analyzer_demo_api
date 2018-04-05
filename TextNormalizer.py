#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import enchant
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import nltk
from nltk.corpus import wordnet
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

class TextNormalizer(BaseEstimator,TransformerMixin):
    def __init__(self, vocab_filename = None):
        if vocab_filename == None:
            vocab_filename = 'pw.txt'
        self.__vocab_filename = vocab_filename
        self.__enc = enchant.Dict("en_UK")
        
    
    def __clean_comment(self, text):

        def get_wordnet_pos(treebank_tag):
        
            if treebank_tag.startswith('J'):
                return wordnet.ADJ
            elif treebank_tag.startswith('V'):
                return wordnet.VERB
            elif treebank_tag.startswith('N'):
                return wordnet.NOUN
            elif treebank_tag.startswith('R'):
                return wordnet.ADV
            else:
                return ''        
        def check_spelling(text):
            if not self.__enc.check(text):
                suggestions = list(set(self.__enc.suggest(text)).intersection(set(self.__spelling_vocab)))
                if len(suggestions)>0:
                    res = suggestions[0]
                elif len(self.__enc.suggest(text))>0:
                    res = self.__enc.suggest(text)[0]
                else:
                    res = text
            else:
                res = text
            return res    
        
        wnl = WordNetLemmatizer()
        deacc = re.sub(r'\!',' exclamation_point ', text)
        tokens = word_tokenize(deacc)
        tags = nltk.pos_tag(tokens)
        processed = []
        for (word, tag) in tags:
            wn_tag = get_wordnet_pos(tag)
            if wn_tag!='':
                processed.append(wnl.lemmatize(check_spelling(word),wn_tag))
            else:
                processed.append(wnl.lemmatize(check_spelling(word)))
        res = ' '.join(processed)
        return res
    
    def transform(self, X, y=None, **fit_params):
        res = []
        for line in X:
            res.append(self.__clean_comment(line))
        return res

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X=None, y=None, **fit_params):
        with open(self.__vocab_filename) as f:
            self.__spelling_vocab = f.readlines()
        self.__spelling_vocab = [x.strip() for x in self.__spelling_vocab]
        return self
    
if __name__ == '__main__':
    tn = TextNormalizer()
    print(tn.fit_transform(['tihs game is assome']))