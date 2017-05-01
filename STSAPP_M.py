# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:56:41 2017

@author: tilakputta
"""
from __future__ import division
from Tkinter import *
from ttk import Notebook
import tkFileDialog as fd
from ScrolledText import ScrolledText
import nltk
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.metrics import jaccard_distance
from nltk.metrics import edit_distance
from nltk.util import ngrams
from ete2 import Tree
from nltk.corpus import brown
from sklearn import svm
import os
import subprocess
import time
import math
from sklearn import ensemble
import csv
import pandas as pd
import numpy as np
import sys
from nltk.stem import WordNetLemmatizer
from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
lemmatizer = WordNetLemmatizer()
global directory,filename

global out_file
global status
def browseDestDir():
    directory.set(fd.askdirectory(title="Ouput Directory:")+"/"+filename.get())
    status.set("")
def lcs(X, Y):
    mat = []
    for i in range(0,len(X)):
        row = []
        for j in range(0,len(Y)):
            if X[i] == Y[j]:
                if i == 0 or j == 0:
                    row.append(1)
                else:
                    val = 1 + int( mat[i-1][j-1] )
                    row.append(val)
            else:
                row.append(0)
        mat.append(row)
    new_mat = []
    for r in  mat:
        r.sort()
        r.reverse()
        new_mat.append(r)
    lcs = 0
    for r in new_mat:
        if lcs < r[0]:
            lcs = r[0]
    return lcs
ALPHA = 0.2
BETA = 0.45
ETA = 0.4
PHI = 0.2
DELTA = 0.85
sent_pair=[]

brown_freqs = dict()
N = 0
def get_best_synset_pair(word_1, word_2):
    """ 
    Choose the pair with highest path similarity among all pairs. 
    Mimics pattern-seeking behavior of humans.
    """
    max_sim = -1.0
    synsets_1 = wn.synsets(word_1)
    synsets_2 = wn.synsets(word_2)
    if len(synsets_1) == 0 or len(synsets_2) == 0:
        return None, None
    else:
        max_sim = -1.0
        best_pair = None, None
        for synset_1 in synsets_1:
            for synset_2 in synsets_2:
               sim = wn.path_similarity(synset_1, synset_2)
               if sim != None:
                   if sim > max_sim:
                       max_sim = sim
                       best_pair = synset_1, synset_2
        return best_pair

def length_dist(synset_1, synset_2):
    """
    Return a measure of the length of the shortest path in the semantic 
    ontology (Wordnet in our case as well as the paper's) between two 
    synsets.
    """
    l_dist = sys.maxsize
    if synset_1 is None or synset_2 is None: 
        return 0.0
    if synset_1 == synset_2:
        # if synset_1 and synset_2 are the same synset return 0
        l_dist = 0.0
    else:
        wset_1 = set([str(x.name()) for x in synset_1.lemmas()])        
        wset_2 = set([str(x.name()) for x in synset_2.lemmas()])
        if len(wset_1.intersection(wset_2)) > 0:
            # if synset_1 != synset_2 but there is word overlap, return 1.0
            l_dist = 1.0
        else:
            # just compute the shortest path between the two
            l_dist = synset_1.shortest_path_distance(synset_2)
            if l_dist is None:
                l_dist = 0.0
    # normalize path length to the range [0,1]
    return math.exp(-ALPHA * l_dist)

def hierarchy_dist(synset_1, synset_2):
    """
    Return a measure of depth in the ontology to model the fact that 
    nodes closer to the root are broader and have less semantic similarity
    than nodes further away from the root.
    """
    h_dist = sys.maxsize
    if synset_1 is None or synset_2 is None: 
        return h_dist
    if synset_1 == synset_2:
        # return the depth of one of synset_1 or synset_2
        h_dist = max([x[1] for x in synset_1.hypernym_distances()])
    else:
        # find the max depth of least common subsumer
        hypernyms_1 = {x[0]:x[1] for x in synset_1.hypernym_distances()}
        hypernyms_2 = {x[0]:x[1] for x in synset_2.hypernym_distances()}
        lcs_candidates = set(hypernyms_1.keys()).intersection(
            set(hypernyms_2.keys()))
        if len(lcs_candidates) > 0:
            lcs_dists = []
            for lcs_candidate in lcs_candidates:
                lcs_d1 = 0
                if lcs_candidate in hypernyms_1:
                    lcs_d1 = hypernyms_1[lcs_candidate]
                lcs_d2 = 0
                if lcs_candidate in hypernyms_2:
                    lcs_d2 = hypernyms_2[lcs_candidate]
                lcs_dists.append(max([lcs_d1, lcs_d2]))
            h_dist = max(lcs_dists)
        else:
            h_dist = 0
    return ((math.exp(BETA * h_dist) - math.exp(-BETA * h_dist)) / 
        (math.exp(BETA * h_dist) + math.exp(-BETA * h_dist)))
    
def word_similarity(word_1, word_2):
    synset_pair = get_best_synset_pair(word_1, word_2)
    return (length_dist(synset_pair[0], synset_pair[1]) * 
        hierarchy_dist(synset_pair[0], synset_pair[1]))
def most_similar_word(word, word_set):
    """
    Find the word in the joint word set that is most similar to the word
    passed in. We use the algorithm above to compute word similarity between
    the word and each word in the joint word set, and return the most similar
    word and the actual similarity value.
    """
    max_sim = -1.0
    sim_word = ""
    for ref_word in word_set:
      sim = word_similarity(word, ref_word)
      if sim > max_sim:
          max_sim = sim
          sim_word = ref_word
    return sim_word, max_sim

def info_content(lookup_word):
    """
    Uses the Brown corpus available in NLTK to calculate a Laplace
    smoothed frequency distribution of words, then uses this information
    to compute the information content of the lookup_word.
    """
    global N
    if N == 0:
        # poor man's lazy evaluation
        for sent in brown.sents():
            for word in sent:
                word = word.lower()
                if not word in brown_freqs:
                    brown_freqs[word] = 0
                brown_freqs[word] = brown_freqs[word] + 1
                N = N + 1
    lookup_word = lookup_word.lower()
    n = 0 if not lookup_word in brown_freqs else brown_freqs[lookup_word]
    return 1.0 - (math.log(n + 1) / math.log(N + 1))

def semantic_vector(words, joint_words, info_content_norm):
    """
    Computes the semantic vector of a sentence. The sentence is passed in as
    a collection of words. The size of the semantic vector is the same as the
    size of the joint word set. The elements are 1 if a word in the sentence
    already exists in the joint word set, or the similarity of the word to the
    most similar word in the joint word set if it doesn't. Both values are 
    further normalized by the word's (and similar word's) information content
    if info_content_norm is True.
    """
    sent_set = set(words)
    semvec = np.zeros(len(joint_words))
    i = 0
    for joint_word in joint_words:
        if joint_word in sent_set:
            # if word in union exists in the sentence, s(i) = 1 (unnormalized)
            semvec[i] = 1.0
            if info_content_norm:
                semvec[i] = semvec[i] * math.pow(info_content(joint_word), 2)
        else:
            # find the most similar word in the joint set and set the sim value
            sim_word, max_sim = most_similar_word(joint_word, sent_set)
            semvec[i] = PHI if max_sim > PHI else 0.0
            if info_content_norm:
                semvec[i] = semvec[i] * info_content(joint_word) * info_content(sim_word)
        i = i + 1
    return semvec

def semantic_similarity(sentence_1, sentence_2, info_content_norm):
    """
    Computes the semantic similarity between two sentences as the cosine
    similarity between the semantic vectors computed for each sentence.
    """
    
    words_1 = nltk.word_tokenize(sentence_1)
    words_2 = nltk.word_tokenize(sentence_2)
    joint_words = set(words_1).union(set(words_2))
    vec_1 = semantic_vector(words_1, joint_words, info_content_norm)
    vec_2 = semantic_vector(words_2, joint_words, info_content_norm)
    return np.dot(vec_1, vec_2.T) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))
def word_order_vector(words, joint_words, windex):
    """
    Computes the word order vector for a sentence. The sentence is passed
    in as a collection of words. The size of the word order vector is the
    same as the size of the joint word set. The elements of the word order
    vector are the position mapping (from the windex dictionary) of the 
    word in the joint set if the word exists in the sentence. If the word
    does not exist in the sentence, then the value of the element is the 
    position of the most similar word in the sentence as long as the similarity
    is above the threshold ETA.
    """
    wovec = np.zeros(len(joint_words))
    i = 0
    wordset = set(words)
    
    for joint_word in joint_words:
        if joint_word in words:
            # word in joint_words found in sentence, just populate the index
            wovec[i] = windex[joint_word]+1
        else:
            # word not in joint_words, find most similar word and populate
            # word_vector with the thresholded similarity
            
            sim_word, max_sim = most_similar_word(joint_word, wordset)
            
            if max_sim > ETA:
                wovec[i] = windex[sim_word]+1
            else:
                wovec[i] = 0
        i = i + 1
    #print(wovec)
    return wovec
def word_order_similarity(sentence_1, sentence_2):
    """
    Computes the word-order similarity between two sentences as the normalized
    difference of word order between the two sentences.
    """
    words_1 = nltk.word_tokenize(sentence_1)
    words_2 = nltk.word_tokenize(sentence_2)
    joint_words = list(set(words_1).union(set(words_2)))
    windex = {x[1]: x[0] for x in enumerate(words_1)}
    r1 = word_order_vector(words_1, joint_words, windex)
    windex = {x[1]: x[0] for x in enumerate(words_2)}
    r2 = word_order_vector(words_2, joint_words, windex)
    print("%.3f" % (1.0 - (np.linalg.norm(abs(r1 - r2)) / np.linalg.norm(r1 + r2))))
    return 1.0 - (np.linalg.norm(abs(r1 - r2)) / np.linalg.norm(r1 + r2))
def nGrams(d,inputfile):
    text.configure(state="normal")
    text.delete('1.0',END)
    e = inputfile+"/"
    fi = open(d+inputfile+".txt",'r')
    d += e
    fn = open(d+"ngrams.txt",'w')
    fi.readline()
    fn.write("char2\tchar3\tchar4\tword1\tword2\tword3\tlemma1\tlemma2\tlemma3\tposgm1\tposgm2\tposgm3\n")
    for line in fi.readlines():
        sents = line.split("\t")
        words_1 = sents[0].split()
        words_2 = sents[1].split()
        if len(words_1) < 3:
            words_1.append(".")
        if len(words_2) < 3:
            words_2.append(".")
        char1_2 = set(ngrams(sents[0],2))
        char1_3 = set(ngrams(sents[0],3))
        char1_4 = set(ngrams(sents[0],4))
        char2_2 = set(ngrams(sents[1],2))
        char2_3 = set(ngrams(sents[1],3))
        char2_4 = set(ngrams(sents[1],4))
        word1_1 = set(ngrams(words_1,1))
        word1_2 = set(ngrams(words_1,2))
        word1_3 = set(ngrams(words_1,3))
        word2_1 = set(ngrams(words_2,1))
        word2_2 = set(ngrams(words_2,2))
        word2_3 = set(ngrams(words_2,3))
        sent1 = nltk.pos_tag(words_1)
        sent2 = nltk.pos_tag(words_2)
        nouns = ['NN','NNS','NNP','NNPS']
        adj = ['JJ','JJR','JJS']
        adv = ['RB','RBR','RBS']
        verbs = ['VB','VBG','VBN','VBZ','VBP','VBD']
        all_pos = ['NN','NNS','NNP','NNPS','JJ','JJR','JJS','RB','RBR','RBS','VB','VBG','VBN','VBZ','VBP','VBD']
        posgm1 = []
        posgm2 = []
        
        for s in sent1:
            posgm1.append(s[1])
        for s in sent2:
            posgm2.append(s[1])
        
        pos = []
        for s in sent1:
            if s[1] not in all_pos:
                pos.append((s[0],'v'))
            if s[1] in nouns:
                pos.append((s[0],'n'))
            if s[1] in adj:
                pos.append((s[0],'a'))
            if s[1] in adv:
                pos.append((s[0],'r'))
            if s[1] in verbs:
                pos.append((s[0],'v'))
        sent1 = pos
        pos = []
        for s in sent2:
            if s[1] not in all_pos:
                pos.append((s[0],'v'))
            if s[1] in nouns:
                pos.append((s[0],'n'))
            if s[1] in adj:
                pos.append((s[0],'a'))
            if s[1] in adv:
                pos.append((s[0],'r'))
            if s[1] in verbs:
                pos.append((s[0],'v'))
        sent2 = pos
        lemma1 = []
        for s in sent1:
            lemma1.append(lemmatizer.lemmatize(s[0],s[1]))
        lemma2 = []
        for s in sent2:
            lemma2.append(lemmatizer.lemmatize(s[0],s[1]))
        
        print sents[0],sents[1]
        lemma1_1 = set(ngrams(lemma1,1))
        lemma1_2 = set(ngrams(lemma1,2))
        lemma1_3 = set(ngrams(lemma1,3))
        lemma2_1 = set(ngrams(lemma2,1))
        lemma2_2 = set(ngrams(lemma2,2))
        lemma2_3 = set(ngrams(lemma2,3))
        posgm1_1 = set(ngrams(posgm1,1))
        posgm1_2 = set(ngrams(posgm1,2))
        posgm1_3 = set(ngrams(posgm1,3))
        posgm2_1 = set(ngrams(posgm2,1))
        posgm2_2 = set(ngrams(posgm2,2))
        posgm2_3 = set(ngrams(posgm2,3))
        char2_jd = 1.0 - jaccard_distance(char1_2,char2_2)
        char3_jd = 1.0 - jaccard_distance(char1_3,char2_3)
        char4_jd = 1.0 - jaccard_distance(char1_4,char2_4)
        word1_jd = 1.0 - jaccard_distance(word1_1,word2_1)
        word2_jd = 1.0 - jaccard_distance(word1_2,word2_2)
        word3_jd = 1.0 - jaccard_distance(word1_3,word2_3)
        lemma1_jd = 1.0 - jaccard_distance(lemma1_1,lemma2_1)
        lemma2_jd = 1.0 - jaccard_distance(lemma1_2,lemma2_2)
        lemma3_jd = 1.0 - jaccard_distance(lemma1_3,lemma2_3)
        posgm1_jd = 1.0 - jaccard_distance(posgm1_1,posgm2_1)
        posgm2_jd = 1.0 - jaccard_distance(posgm1_2,posgm2_2)
        posgm3_jd = 1.0 - jaccard_distance(posgm1_3,posgm2_3)
        text.insert(INSERT,"%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t\n" % (sents[0],sents[1],char2_jd,char3_jd,char4_jd,word1_jd,word2_jd,word3_jd,lemma1_jd,lemma2_jd,lemma3_jd,posgm1_jd,posgm2_jd,posgm3_jd))
        fn.write("%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n" % (char2_jd,char3_jd,char4_jd,word1_jd,word2_jd,word3_jd,lemma1_jd,lemma2_jd,lemma3_jd,posgm1_jd,posgm2_jd,posgm3_jd))
        #print "LCS of ",sent1,sent2,"is: ",lcs(sent1,sent2)/(1.0 * min_len)
    fi.close()
    fn.close()
    text.configure(state="disabled")
def lengthFeatures(d,inputfile):
    text.configure(state="normal")
    text.delete('1.0',END)
    e = inputfile+"/"
    fi = open(d+inputfile+".txt",'r')
    d += e
    fn = open(d+"len.txt",'w')
    fi.readline()
    fn.write("A\tB\tAsB\tBsA\tAuB\tAiB\tAsBdB\tBsAdA\n")
    for line in fi.readlines():
        sents = line.split("\t")
        A = set(sents[0].split())
        B = set(sents[1].split())
        text.insert(INSERT,"%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%.3f\t%.3f\t\n" % (sents[0],sents[1],len(A),len(B),len(A-B),len(B-A),len(A.union(B)),len(A.intersection(B)),(len(A-B)/(1.0 * len(B))),(len(B-A)/(1.0 * len(A)))))
        fn.write("%d\t%d\t%d\t%d\t%d\t%d\t%.3f\t%.3f\n" % (len(A),len(B),len(A-B),len(B-A),len(A.union(B)),len(A.intersection(B)),(len(A-B)/(1.0 * len(B))),(len(B-A)/(1.0 * len(A)))))
        #print "LCS of ",sent1,sent2,"is: ",lcs(sent1,sent2)/(1.0 * min_len)
    fi.close()
    fn.close()
    text.configure(state="disabled")
def tfidfFeature(d,inputfile):
    text.configure(state="normal")
    text.delete('1.0',END)
    e = inputfile+"/"
    fi = open(d+inputfile+".txt",'r')
    d += e
    fn = open(d+"tfidf.txt",'w')
    fi.readline()
    fn.write("tfidf\n")
    for line in fi.readlines():
        sents = line.split("\t")
        cos_sim = 0.0
        sent1 = list(sents[0].split())
        sent2 = list(sents[1].split())
        print sents[0],sents[1]
        #sent1 = [word for word in sent1 if word.lower() not in stopwords.words('english')]
        #sent2 = [word for word in sent2 if word.lower() not in stopwords.words('english')]
        if len(sent1) != 0 or len(sent2) != 0:
            
            sentences = (" ".join(sent1)," ".join(sent2))
            vect = TfidfVectorizer(lowercase=True)
            print(vect)
            tfidf_matrix = vect.fit_transform(sentences).toarray()
            print tfidf_matrix
            cos_sim = cosine_similarity(tfidf_matrix[0],tfidf_matrix[1])
        print cos_sim
        text.insert(INSERT,"%s\t%s\t%.3f\t\n" % (sents[0],sents[1],cos_sim))
        fn.write("%.3f\n" % (cos_sim))
        #print "LCS of ",sent1,sent2,"is: ",lcs(sent1,sent2)/(1.0 * min_len)
    fi.close()
    fn.close()
    text.configure(state="disabled")
def nounphrase(d,inputfile):
    text.configure(state="normal")
    text.delete('1.0',END)
    e = inputfile+"/"
    fi = open(d+inputfile+".txt",'r')
    d += e
    fn = open(d+"nounphrase.txt",'w')
    fi.readline()
    fn.write("nounphrase\n")
    for line in fi.readlines():
        print line
        sents = line.split("\t")
        sent1 = nltk.pos_tag(sents[0].split())
        sent2 = nltk.pos_tag(sents[1].split())
        actual = sents[2][:-1]
        nounphr = "NP: {<DT>?<JJ.*>*<NN.*>}"
        cp = nltk.RegexpParser(nounphr)
        res = cp.parse(sent1)
        res1=cp.parse(sent2)
        #print(res)
        
        np1 = []
        for subtree in res.subtrees():
            if subtree.label() == 'NP':
                np1.append(subtree[0:])
        print(np1)
        res = cp.parse(sent2)
        #print(res)
        np2 = []
        for subtree in res.subtrees():
            if subtree.label() == 'NP':
                np2.append(subtree[0:])
        print(np2)
        np1_len = len(np1)
        np2_len = len(np2)
        #print(np1_len,np2_len)
        mat = []
        nps1 = []
        for nnp1 in np1:
            snt = []
            for pos1 in nnp1:
                snt.append(pos1[0])
            nps1.append(" ".join(snt))
        nps2 = []
        for nnp2 in np2:
            snt = []
            for pos2 in nnp2:
                snt.append(pos2[0])
            nps2.append(" ".join(snt))
        np1 = nps1
        np2 = nps2
        for nnp1 in np1:
            row = []
            for nnp2 in np2:
                similarity = semantic_similarity(nnp1,nnp2, True)
                #print(similarity)
                #print(nnp1,nnp2,similarity)
                row.append(similarity)
            mat.append(row)
        print(mat)
        np_similarity = 0.0
        if(len(mat)>0):
            avg_n = max(len(mat),len(mat[0]))
        else:
            avg_n = 1
        order = []
        while(len(mat)>0):
            if (len(mat[0])==0):
                break
            maximum = 0.0
            m = 0
            maxm = 0
            maxn = 0
            for row in mat:
                n = 0
                for col in row:
                    if maximum < col:
                        maximum = col
                        maxm = m
                        maxn = n
                    n += 1
                m += 1
            print(maximum,maxm,maxn)
            del mat[maxm]
            for row in mat:
                del row[maxn]
            np_similarity += maximum
            
        print(np_similarity)
        np_similarity /= avg_n
        print(np_similarity)
        text.insert(INSERT, "%s\t%s\t%.3f\t\n" % (sents[0],sents[1],np_similarity))
        fn.write("%.3f\n" % (np_similarity))
    fi.close()
    fn.close()
    text.configure(state="disabled")
def lcsFeature(d,inputfile):
    text.configure(state="normal")
    text.delete('1.0',END)
    e = inputfile+"/"
    fi = open(d+inputfile+".txt",'r')
    d += e
    fn = open(d+"lcs.txt",'w+')
    fi.readline()
    fn.write("lcs\n")
    for line in fi.readlines():
        sents = line.split("\t")
        sent1 = sents[0].split()
        sent2 = sents[1].split()
        min_len = min(len(sent1),len(sent2))
        #print "LCS of ",sent1,sent2,"is: ",lcs(sent1,sent2)/(1.0 * min_len)
        LCS = lcs(sent1,sent2)/(1.0 * min_len)
        text.insert(INSERT,"%s\t%s\t%.3f\t\n" % (sents[0],sents[1],LCS))
        fn.write("%.3f\n" % (LCS))
    fi.close()
    fn.close()
    text.configure(state="disabled")
def wordorderFeature(d,inputfile):
    text.configure(state="normal")
    text.delete('1.0',END)
    e = inputfile+"/"
    fi = open(d+inputfile+".txt",'r')
    d += e
    fn = open(d+"wordorder.txt",'w+')
    fi.readline()
    fn.write("wordorder\n")
    for sentence_pairs in fi.readlines():
        sent_pair=sentence_pairs.split("\t")
        sentence_1 = sent_pair[0].lower()
        sentence_2 = sent_pair[1].lower()
        wos = word_order_similarity(sentence_1, sentence_2)
        text.insert(INSERT,"%s\t%s\t%.3f\t\n"%(sent_pair[0],sent_pair[1],wos))
        fn.write("%.3f\n"%(wos))
    fi.close()
    fn.close()
    text.configure(state="disabled")
def semanticFeature(d,inputfile):
    text.configure(state="normal")
    text.delete('1.0',END)
    e = inputfile+"/"
    fi = open(d+inputfile+".txt",'r')
    d += e
    fn = open(d+"semantic.txt",'w+')
    fi.readline()
    fn.write("semantic\n")
    for sentence_pairs in fi.readlines():
        sent_pair=sentence_pairs.split("\t")
        sentence_1 = sent_pair[0].lower()
        sentence_2 = sent_pair[1].lower()
        semanT = semantic_similarity(sentence_1, sentence_2, True)
        text.insert(INSERT,"%s\t%s\t%.3f\t\n"%(sent_pair[0],sent_pair[1],semanT))
        fn.write("%.3f\n"%(semanT))
    fi.close()
    fn.close()
    text.configure(state="disabled")
def oneHotCoding(d,inputfile):
    text.configure(state="normal")
    text.delete('1.0',END)
    e = inputfile+"/"
    fi = open(d+inputfile+".txt",'r')
    d += e
    fn = open(d+"onehot.txt",'w+')
    fi.readline()
    fn.write("onehot\n")
    for sentence_pairs in fi.readlines():
        sent_pair=sentence_pairs.split("\t")
        sentence_1 = sent_pair[0].lower()
        sentence_2 = sent_pair[1].lower()
        words_1 = set(nltk.word_tokenize(sentence_1))
        words_2 = set(nltk.word_tokenize(sentence_2))
        j_words = words_1.union(words_2)
        w1 = []
        w2 = []
        for word in j_words:
            if word in words_1:
                w1.append(1)
            else:
                w1.append(0)
            if word in words_2:
                w2.append(1)
            else:
                w2.append(0)
        cos_sim = cosine_similarity(w1,w2)
        print cos_sim
        text.insert(INSERT,"%s\t%s\t%.3f\t\n"%(sent_pair[0],sent_pair[1],cos_sim))
        fn.write("%.3f\n" % (cos_sim))
    fi.close()
    fn.close()
    text.configure(state="disabled")
def combineAll(d,flag):
    d += "/"
    if flag:
        act = open(d+"actual.txt","r")
    fn = open(d+"combined.txt","w")
    leng = open(d+"len.txt","r")
    oneh = open(d+"onehot.txt","r")
    lcs = open(d+"lcs.txt","r")
    wo = open(d+"wordorder.txt","r")
    sem = open(d+"semantic.txt","r")
    ng = open(d+"ngrams.txt","r")
    np = open(d+"nounphrase.txt","r")
    tfidf = open(d+"tfidf.txt","r")
    for npe in np.readlines():
        lst = []
        if flag:
            lst.append((act.readline())[:-1])
        lst.append((leng.readline())[:-1])
        lst.append((lcs.readline())[:-1])
        lst.append((wo.readline())[:-1])
        lst.append((sem.readline())[:-1])
        lst.append(npe[:-1])
        lst.append((ng.readline())[:-1])
        lst.append((oneh.readline())[:-1])
        lst.append((tfidf.readline())[:-1])
        line = "\t".join(lst)
        line += "\n"
        fn.write(line)
    fn.close()
    txt_file = r"combined.txt"
    csv_file = r"combined.csv"
    in_txt = csv.reader(open(d+txt_file, "rb"), delimiter = '\t')
    out_csv = csv.writer(open(d+csv_file, 'wb'))

    out_csv.writerows(in_txt)
def modelGen(d,f):
    text.configure(state="normal")
    text.delete('1.0',END)
    df = d+"/"+f+"/"
    d += "/"
    train = pd.read_csv(df+"combined.csv")
    print d
    cols = ['A', 'B', 'AsB', 'BsA','AuB','AiB','AsBdB','BsAdA','lcs','wordorder','semantic','nounphrase','tfidf','onehot','char2','char3','char4','word1','word2','word3','lemma1','lemma2','lemma3','posgm1','posgm2','posgm3']
    colsRes = ['actual']
    trainArr = train.as_matrix(cols) #training array
    trainRes = train.as_matrix(colsRes) # training results
    print train,trainArr,trainRes
    model = RandomForestRegressor() # initialize
    model.fit(trainArr, trainRes)
    filename = d+'models/model.sav'
    joblib.dump(model, filename)
    text.insert(INSERT,"MODEL GENERATED.")
    text.configure(state="disabled")
def findSim(d):
    d += "/"
    text.configure(state="normal")
    text.delete('1.0',END)
    s1 = sent1.get()
    s2 = sent2.get()
    fn = open(d+"input.txt","w")
    fn.write("sentence1\tsentence2\n")
    fn.write("%s\t%s\t\n" % (s1,s2))
    fn.close()
    f = "input"
    nGrams(d,f)
    tfidfFeature(d,f)
    lengthFeatures(d,f)
    nounphrase(d,f)
    lcsFeature(d,f)
    wordorderFeature(d,f)
    semanticFeature(d,f)
    oneHotCoding(d,f)
    combineAll(d+"/"+f,0)
    cols = ['A', 'B', 'AsB', 'BsA','AuB','AiB','AsBdB','BsAdA','lcs','wordorder','semantic','nounphrase','tfidf','onehot','char2','char3','char4','word1','word2','word3','lemma1','lemma2','lemma3','posgm1','posgm2','posgm3']
    test = pd.read_csv(d+f+"/combined.csv")
    filename = d+'models/model.sav'
    model = joblib.load(filename)
    testArr = test.as_matrix(cols)
    results = model.predict(testArr)
    test['predictions'] = results
    text.configure(state="normal")
    text.delete('1.0',END)
    text.insert(INSERT,results)
    text.configure(state="disabled")
root = Tk()
root.configure(background='teal')
root.title("Syntactic Textual Similarity")
root.resizable(width=False, height=False)
note = Notebook(root)

frame = Frame(root)

queryvar = StringVar()
ql = StringVar()
frame.grid(row=4,column=0,rowspan=12,columnspan=3,pady=20)
text = ScrolledText(frame,wrap="word")
text.grid(row=4)
text.config(font=("Courier", 12))
text.configure(background='black',foreground='green')
directory = StringVar()
filename = StringVar()
sent1 = StringVar()
sent2 = StringVar()
status = StringVar()
modelnum = StringVar()
status.set("")
filename.set("")
Label(root,textvariable=status,bg='teal',foreground='white').grid(row=0,column=1,padx=(20),pady=(20),columnspan=6)

Label(root,text="Directory:",bg='teal',foreground='white').grid(row=2,column=0,padx=(10),pady=(10))
Entry(root,text=directory,bd=2,bg='#cfacaf',width='75').grid(row=2,column=1)

Button(root,command=browseDestDir,text='Browse').grid(row=2,column=2,padx=20)

Label(root,bg='teal',text="File name:",foreground='white').grid(row=3,column=0)
Entry(root,textvariable=filename,bd=2,bg='#cfacaf',width='75').grid(row=3,column=1)
b = Button(root,text='Length Features',command=lambda:lengthFeatures(directory.get(),filename.get()))
b.grid(row=4,column=3,padx=20)
b = Button(root,text='N-Grams',command=lambda:nGrams(directory.get(),filename.get()))
b.grid(row=4,column=4,padx=20)
b = Button(root,text='Tf-Idf',command=lambda:tfidfFeature(directory.get(),filename.get()))
b.grid(row=4,column=5,padx=20)
b = Button(root,text='Noun Phrase',command=lambda:nounphrase(directory.get(),filename.get()))
b.grid(row=5,column=3,padx=20)
b = Button(root,text='LCS Feature',command=lambda:lcsFeature(directory.get(),filename.get()))
b.grid(row=5,column=4,padx=20)
b = Button(root,text='Word Order',command=lambda:wordorderFeature(directory.get(),filename.get()))
b.grid(row=5,column=5,padx=20)
b = Button(root,text='Semanti With Corpus',command=lambda:semanticFeature(directory.get(),filename.get()))
b.grid(row=7,column=3,padx=20)

b = Button(root,text='One Hot Coding',command=lambda:oneHotCoding(directory.get(),filename.get()))
b.grid(row=8,column=3,padx=20)
b = Button(root,text='Combine All',command=lambda:combineAll(directory.get()+"/"+filename.get(),1))
b.grid(row=9,column=3,padx=20)
b = Button(root,text='Generate Model',command=lambda:modelGen(directory.get(),filename.get()))
b.grid(row=14,column=3,padx=20)

Label(root,text="Sentence 1:",bg='teal',foreground='white').grid(row=16,column=0,padx=(10),pady=(10))
Entry(root,text=sent1,bd=2,bg='#cfacaf',width='75').grid(row=16,column=1)

Label(root,text="Sentence 2:",bg='teal',foreground='white').grid(row=18,column=0,padx=(10),pady=(10))
Entry(root,text=sent2,bd=2,bg='#cfacaf',width='75').grid(row=18,column=1)
b = Button(root,text='Find degree of equivalence',command=lambda:findSim("D:/Major Project"))
b.grid(row=19,column=1,padx=20,pady=20)

root.mainloop()