# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 01:10:48 2018

@author: gayatri
        
"""

import sys
from nltk import word_tokenize
from nltk.util import bigrams
import re
import pandas as pd


class BigramModel:
    word_count_dict = {}
    bigram_count_sent = {}
    
    #Unigram counts of all words in corpus and bigram counts of pair of words in given sentences
    def wordCount(self, file, sent1_bigram_list, sent2_bigram_list, sentence1):
        self.file = file
        textData = ". " + file.read()   #Appending '. ' at the beginning of file to mark beginning of a sentence
        sent1 = word_tokenize(sentence1)
        #Storing a=tokens and bigram tuples in lower case
        for token in word_tokenize(textData):
            if token.lower() in BigramModel.word_count_dict.keys():
                BigramModel.word_count_dict[token.lower()] = BigramModel.word_count_dict[token.lower()] + 1
            else :
                BigramModel.word_count_dict.update({token.lower():1})
        count = 0
        for bigram_tuple in sent1_bigram_list:
            if bigram_tuple[1] != ".":
                bigram = "\s".join(str(word.lower()) for word in bigram_tuple)
            else:
                bigram = "".join(str(word.lower()) for word in bigram_tuple)
            
            count = len(re.findall(bigram, textData.lower()))
            #print(bigram, count)
                        
            BigramModel.bigram_count_sent.update({bigram_tuple:count})
          
        if (not file.closed):
            file.close()
        
    #Find probability of bigrams of given sentence. If isSmoothing = True, Add-one smoothing. isSmoothing = False, Without smoothing         
    def calculateProbabilty(self, sentence, isSmoothing): 
        # Without smoothing - conditional bigram count/unigram count
        # With smoothing - (conditional bigram count + 1)/(unigram count + vocabulary count)
        
        sentence_list = word_tokenize(sentence)
        probability_table = [[0 for x in range(len(sentence_list))] for y in range(len(sentence_list))]
        
        if isSmoothing:
            count_table = [[1 for x in range(0,len(sentence_list))] for y in range(0,len(sentence_list))]
            vocabulary_count = len(BigramModel.word_count_dict)
            for x in range(0,len(sentence_list)):
                for y in range(0,len(sentence_list)):
                    probability_table[x][y] = (count_table[x][y])/(BigramModel.word_count_dict.get(sentence_list[x].lower(),0) + vocabulary_count)
            
        else:
            count_table = [[0 for x in range(0,len(sentence_list))] for y in range(0,len(sentence_list))]
            for x in range(0,len(sentence_list)):
                for y in range(0,len(sentence_list)):
                    probability_table[x][y] = count_table[x][y]/BigramModel.word_count_dict.get(sentence_list[x].lower()) 
                    
        
        for x in range(0,len(sentence_list)):
            for y in range(0,len(sentence_list)):
                if (sentence_list[x].lower(),sentence_list[y].lower()) in BigramModel.bigram_count_sent.keys():
                    if isSmoothing:
                        count_table[x][y] = BigramModel.bigram_count_sent.get((sentence_list[x].lower(),sentence_list[y].lower())) + 1
                        probability_table[x][y] = (count_table[x][y])/(BigramModel.word_count_dict.get(sentence_list[x].lower(),0) + vocabulary_count)
                    else:
                        count_table[x][y] = BigramModel.bigram_count_sent.get((sentence_list[x].lower(),sentence_list[y].lower()))
                        probability_table[x][y] = count_table[x][y]/BigramModel.word_count_dict.get(sentence_list[x].lower()) 
                    
        return sentence_list, count_table, probability_table
        
    # Sentence probability
    def calculateSentenceProbability(self, sentence_list, probability_table):
        probability = 1
        for x in range(0,len(sentence_list) - 2):
            probability = probability * probability_table[x][x+1]
            
        return probability
        
    #Draw output in tabular form    
    def drawTable(self, sentence_list, table):
        print(pd.DataFrame(table,index=sentence_list,columns = sentence_list))
        
    #Create bigrams     
    def createBigrams(self, sentence):
        bigram_list = list(bigrams(word_tokenize(sentence.lower())))
        return bigram_list
    

#Main
        
modelObj = BigramModel() 
    
file = open (sys.argv[1],"r", encoding="ISO-8859-1")

sent1 = modelObj.createBigrams(sys.argv[2])
sent2 = modelObj.createBigrams(sys.argv[3])

modelObj.wordCount(file, sent1, sent2, sys.argv[2])

print("\nBigrams Without Smoothing: ")
print("----------------------------")
sentence1_list, count_table1, probability1 = modelObj.calculateProbabilty(sys.argv[2], False)
sentence2_list, count_table2, probability2 = modelObj.calculateProbabilty(sys.argv[3], False)

print("Bigram Count Table of Sentence 1 : ", sys.argv[2], "\n")
modelObj.drawTable(sentence1_list, count_table1)

print("\nProbability Table of Sentence 1 : ", sys.argv[2], "\n")
modelObj.drawTable(sentence1_list, probability1)

prob1 = modelObj.calculateSentenceProbability(sentence1_list, probability1)
print("\nProbability: ", prob1)

print("\nBigram Count Table of Sentence 2 : ", sys.argv[3], "\n")
modelObj.drawTable(sentence2_list, count_table2)

print("\nProbability Table of Sentence 2 : ", sys.argv[3], "\n")
modelObj.drawTable(sentence2_list, probability2)

prob2 = modelObj.calculateSentenceProbability(sentence2_list, probability2)
print("\nProbability: ", prob2)

print("\n\nBigrams With Smoothing: ")
print("-------------------------")
sentence1_list, count_table3, probability3 = modelObj.calculateProbabilty(sys.argv[2], True)
sentence2_list, count_table4, probability4 = modelObj.calculateProbabilty(sys.argv[3], True)

print("Bigram Count Table of Sentence 1 : ", sys.argv[2], "\n")
modelObj.drawTable(sentence1_list, count_table3)

print("\nProbability Table of Sentence 1 : ", sys.argv[2], "\n")
modelObj.drawTable(sentence1_list, probability3)

prob3 = modelObj.calculateSentenceProbability(sentence1_list, probability3)
print("\nProbability: ", prob3)

print("\nBigram Count Table of Sentence 2 : ", sys.argv[3], "\n")
modelObj.drawTable(sentence2_list, count_table4)

print("\nProbability Table of Sentence 2 : ", sys.argv[3], "\n")
modelObj.drawTable(sentence2_list, probability4)

prob4 = modelObj.calculateSentenceProbability(sentence2_list, probability4)
print("\nProbability: ", prob4)

if (not file.closed):
    file.close()
