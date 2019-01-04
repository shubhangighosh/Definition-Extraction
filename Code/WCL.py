#Header files
import os
import sys
import re
import numpy as np
from nltk.corpus import stopwords
from string import digits
import random
import math


#Global Variables
sentences=[]
tagged_sentences=[]
targets=[]
stop_words = set (stopwords.words( 'english' ))
sentences_bad = []
tagged_sentences_bad = []
targets_bad = []
augmentFlag = True
f= open("errors","w+")


#Getting sentences and tagged sentences fron files
# Use ukwac testfile as arg 1
with open(str(sys.argv[1]),'r') as fp:
    lines=fp.readlines()
    lines = [line.replace('\n','') for line in lines]
    # print('\n'.join(lines))
    
    for index,line in enumerate(lines):
        if index%2==0:
            sentences.append(line[2:])
        else:
            # parts=line.split(':')
            # targets.append(parts[0])
            tagged_sentences.append(line)

#Train Sentence to tagged sentence dictionary
sent_tagging_dict = {}    
for i in range(len(sentences)):
    sent_tagging_dict[sentences[i]] = tagged_sentences[i]  

#Train sentence to target word dictionary
# sent_target_dict = {}    
# for i in range(len(sentences)):
#     sent_target_dict[sentences[i]] = targets[i]      

# Function to get most frequent words set
def frequent_set(train_sentences,k):
    if k=='default':
        freq = set (stopwords.words( 'english' ))
    else:
        freq_dict = {}
        for sent in train_sentences:
            words = sent.split()
            for word in words:
                if word.lower() in freq_dict:
                    freq_dict[word.lower()]+=1
                else:
                    freq_dict[word.lower()]=1
        sorted_dict = sorted(freq_dict, key=freq_dict.get)
        total_size = len(sorted_dict)
        freq = set(sorted_dict[total_size-k:])
    return freq


#################################PREPORCESSING FUNCTIONS#######################################
#To find string between two substrings
def find_between(start,end,str_in):
    return (str_in.split(start))[1].split(end)[0]
#Find POS given token
def POS(tagged_word):
    tokens = tagged_word.split('_')
    
    if(len(tokens)!=3):
        
        raise AttributeError()
    else: 
        return tokens[1]    

#Find phrase given token belongs to
def phrase(tagged_word):
    tokens = tagged_word.split('_')
    
    if(len(tokens)!=3):
        
        raise AttributeError()
    else: 
        return tokens[0]            

#Find word given token
def word(tagged_word):
    tokens = tagged_word.split('_')
    
    if(len(tokens)!=3):
        raise AttributeError()
    else: 
        return tokens[2]  
  
#Splitting sentence into phrases
def TaggedSentToTargetPhrase(tagged_sentence):
    return find_between('<RGET>','<',tagged_sentence)[1:]

def TaggedSentToVerbPhrase(tagged_sentence):
    return find_between('<VERB>','<',tagged_sentence)[1:]    

def TaggedSentToGenusPhrase(tagged_sentence):
    genus_phrase = find_between('<GENUS>','<REST>',tagged_sentence)[1:]
    genus_phrase = genus_phrase.replace('<HYPER>','')
    genus_phrase = genus_phrase.replace('</HYPER>','')
    return genus_phrase

def TaggedSentToHypernym(tagged_sentence):
    hyp_phrase = find_between('<HYPER>','</HYPER>',tagged_sentence)[1:]
    
    hyp = ''
    for x in hyp_phrase.strip().split():
        x = x.split('_')[-1]
        hyp = hyp+x+' '
    return hyp[:-1]  

def TaggedSentToTaggedHypernym(tagged_sentence):
    return find_between('<HYPER>','</HYPER>',tagged_sentence)[1:]

def TaggedSentenceToTargetWord(tagged_sentence):
    parts=tagged_sentence.split(':') 
    return parts[0]  
    


#Stripping Phrase tags
def strip_phrase_tags(tagged_sentence):
    parts=tagged_sentence.split(':')
    taggged_sentence = tagged_sentence[len(parts[0])+1:]
    tagged_sentence = tagged_sentence.replace('<RGET>',' ')
    tagged_sentence = tagged_sentence.replace('<VERB>',' ') 
    tagged_sentence = tagged_sentence.replace('<GENUS>',' ')
    tagged_sentence = tagged_sentence.replace('<HYPER>',' ')
    tagged_sentence = tagged_sentence.replace('</HYPER>',' ') 
    if tagged_sentence.find('<REST>') < tagged_sentence.find('SENT_') and taggged_sentence.find('<REST>') != -1: 
        rest = find_between('<REST>','SENT_',tagged_sentence)
        tagged_sentence = tagged_sentence.replace(rest,' ')
    tagged_sentence = tagged_sentence.replace('<REST>',' ')
    tagged_sentence = tagged_sentence.replace('SENT_',' ')
    tagged_sentence = tagged_sentence.replace('\t',' ')[1:]

    return tagged_sentence
#Checking if bracket has been opened in tagged sentence
def bracket_open(tagged_word):
    tokens = tagged_word.split('_')
    
    bracket_token = False
    if tokens[0] == '(':
        bracket_token = True 
    if len(tokens)>2:
        if tokens[2]=='(' and tokens[1]=='(':
            bracket_token=True
    return bracket_token   
#Checking if bracket has been closed in tagged sentence    
def bracket_close(tagged_word):
    tokens = tagged_word.split('_')
    
    bracket_token = False
    if tokens[0] == ')':
        bracket_token = True 
    if len(tokens)>2:

        if tokens[2]==')' and tokens[1]==')':

            bracket_token=True 
    return bracket_token        

#Generalised phrase from tagged sentnce
def generalise_phrase(tagged_phrase):
    tokens =  tagged_phrase.strip().split()
    
    gen_sent = ''
    current_phrase = ''
    in_bracket = False
    words_till_prev_np  = 0
    prev_phrase = 'start'
    prev_POS = 'start'
    
    #Checking if inside bracket
    for token in tokens:
        
        if bracket_open(token):
            in_bracket = not in_bracket and bracket_open(token)
            if '(_(' in token:
                token='(_('
            
        if bracket_close(token): 
            in_bracket = not(in_bracket and bracket_close(token))
            if ')_)' in token:
                
                token=')_)'

            
        
        try:
            if not in_bracket:
                #to remove noun phrases before brackets which contain TARGET as redundant expansion
                if (prev_phrase != 'start' and prev_phrase != 'NP') and phrase(token) == 'NP':
                    words_till_prev_np = 0
                else:
                    words_till_prev_np += 1

                #replacing sequence of nouns with NP
                if prev_POS == 'NP' and POS(token) == 'NP' and word(token) != 'TARGET':
                    
                    continue
                elif word(token).lower() in stop_words:
                    
                    gen_sent = gen_sent + word(token).lower() + ' '
                elif word(token) == 'TARGET':
                    
                    gen_sent = gen_sent + word(token) + ' '    
                else:
                    
                    gen_sent = gen_sent + POS(token) + ' '  

                prev_POS = phrase(token)   
            #ignoring brackets apart from the ones which contain TARGET     
            else:
                prev_POS = 'bracket'
                if word(token) != 'TARGET':
                    continue
                elif word(token) == 'TARGET':
                    
                    gen_sent = ' ' + gen_sent
                    gen_sent = gen_sent.rsplit(' ', (words_till_prev_np+1))[0]

                    if gen_sent != '' and gen_sent[0] == ' ':
                        gen_sent = gen_sent[1:]
                    
                    gen_sent = gen_sent + word(token) + ' '


        except AttributeError:
            pass
    return gen_sent[:-1]                  

#Star Pattern from generalised phrase
def star_pattern(generalised_phrase):
    sp = ''
    tokens =  generalised_phrase.strip().split()
    for token in tokens:
        if token in stop_words or token == 'TARGET':
            sp = sp + token + ' '
        else:
            sp = sp + '* '   
    return sp[:-1]

#Finding the next key in wcl graph given POS of key
def find_key(li,pos):
    key1 = None
    for el in li:
        if el.translate(None, digits) == pos:
            key1 = el
    return key1   

#Finding the next key in graph which maintains continuity of POS in current snetence
def find_key_continuity(wcl_graph, token, token_next, best_key):
    key = find_key(wcl_graph[best_key], token)
    if key !=  None:
        next_key =  find_key(wcl_graph[key],token_next)
        new_li = [x for x in wcl_graph[best_key] if x != key]
        if next_key == None:
            while next_key == None:
                
                key = find_key(new_li,token ) 
                if key == None:
                    break 
                next_key = find_key(wcl_graph[key],token_next)
                new_li = new_li.remove(key)
    return key        


####################################FUNTIONS FOR GENERATING GRAPH#####################################
#Backtracking dp table
def backtrack(dp_indices, i,j):
    if i==0 or j==0:
        return [np.array([i,j])]
    return backtrack(dp_indices, dp_indices[i][j][0],dp_indices[i][j][1]) + [np.array([i,j])]

#DP Match function
def match(sentence_1,sentence_2):
    array_1=sentence_1.strip().split()
    array_2=sentence_2.strip().split()
    dp_memory = np.zeros((len(array_1)+1,len(array_2)+1))
    dp_indices = np.zeros((len(array_1)+1,len(array_2)+1,2), dtype=int)
    dp_path = []
    for i in range(1,len(array_1)+1):
        for j in range(1,len(array_2)+1):
            #cost for substitution is lower than insertion and deletion
            dp_memory[i,j]=max(dp_memory[i-1,j-1]+int(array_1[i-1]==array_2[j-1]), dp_memory[i-1,j]*0.8,dp_memory[i,j-1]*0.8)
            
            
            if dp_memory[i,j]==dp_memory[i,j-1]:
                dp_indices[i,j][0]=i
                dp_indices[i,j][1]=j-1
               
            elif dp_memory[i,j]==dp_memory[i-1,j]:
                dp_indices[i,j][0]=i-1
                dp_indices[i,j][1]=j
                

            elif dp_memory[i,j]==dp_memory[i-1,j-1]+int(array_1[i-1]==array_2[j-1]):
                dp_indices[i,j][0]=i-1
                dp_indices[i,j][1]=j-1
                
    dp_path=backtrack(dp_indices,len(array_1),len(array_2))
   
    dp_path = np.array(dp_path)    
    

    return dp_memory[len(array_1),len(array_2)],dp_path

#Converting generalised sentences/phrases in a star pattern cluster to a wcl graph
def StarPatternToGraph(tagged_sentences,StarPatternDict,StarPatterns):
    #generating dictionary of wcl graphs indexed by Star Patterns
    wclGraphDict = {}

    #For each Star Pattern
    for StarPattern in StarPatterns:
        wcl_graph = {}
        graph_pos_count = 0
        tokens = StarPatternDict[StarPattern][0].strip().split()
        #Generating graph acc to first sentence in star pattern cluster
        if len(tokens) != 0:
            wcl_graph['<s>'] = [tokens[0]+str(graph_pos_count)]
            
            key  = tokens[0]+str(graph_pos_count)
            for i in range(len(tokens)-1):
                graph_pos_count+=1
                wcl_graph[key] = [tokens[i+1]+str(graph_pos_count)]
                key = tokens[i+1]+str(graph_pos_count)
                
            wcl_graph[key]  = ['</s>'] 
            #Modifying graph to appened new sentences
            for i in np.arange(len(StarPatternDict[StarPattern][1:]))+1:
                best_score = -np.inf
                gen_sen = StarPatternDict[StarPattern][i]
                #Finding best fit sentence
                for j in range(i):
                    
                    prev_sen = StarPatternDict[StarPattern][j]
                    score,path = match(gen_sen,prev_sen)
                    
                    if score > best_score:
                        best_score = score
                        best_sen = prev_sen
                        best_path = path

                #Tokens of current sentence       
                tokens_gen = gen_sen.strip().split()
                #Tokens of best fit sentence which is already in graph
                tokens_best = best_sen.strip().split() 
                tokens_gen = ['<s>'] + tokens_gen + ['</s>']
                tokens_best = ['<s>'] + tokens_best + ['</s>'] 
                
                gen_key = '<s>'
                best_key = '<s>'
                prev_eq = True
                

                #Comparing mismatch of tokens and appending edge accordingly
                for i in range(len(tokens_gen))[1:]:
                    curr_eq = (tokens_gen[i] == tokens_best[i])
                    
                    if tokens_best[i] != '</s>':    
                        key  = find_key_continuity(wcl_graph, tokens_best[i], tokens_best[i+1], best_key)     
                        
                    if curr_eq == False and prev_eq == True:
                        graph_pos_count +=1
                        value = tokens_gen[i] + str(graph_pos_count)
                        if tokens_gen[i] not in map(lambda x: x.translate(None, digits),wcl_graph[best_key]):
                            wcl_graph[best_key].append(value)
                            gen_key = value
                        else:
                            gen_key = find_key(wcl_graph[best_key],tokens_gen[i])
                        if gen_key not in wcl_graph:
                            wcl_graph[gen_key] = []
                        
                    elif curr_eq == False and prev_eq == False:
                        graph_pos_count +=1
                        value = tokens_gen[i] + str(graph_pos_count) 
                        
                        wcl_graph[gen_key].append(value)
                        if value not in wcl_graph:
                            wcl_graph[value] = []
                        gen_key = value
                        
                    elif curr_eq == True and prev_eq == False:
                        value = find_key(wcl_graph[best_key],tokens_best[i])
                        wcl_graph[gen_key].append(value)

                    if best_key != '</s>':    
                        best_key = key    
                    prev_eq = curr_eq
        wclGraphDict[StarPattern] = wcl_graph
    return wclGraphDict    

# #WCL-1
# def wcl1(tagged_sentences):
#     #clustering sentences acc to Star Patterns
#     StarPatterns = []
#     StarPatternDict = {}
#     for sent in tagged_sentences:
#         strip_sent = strip_phrase_tags(sent)
#         gen_sen = generalise_phrase(strip_sent)
#         StarPattern = star_pattern(gen_sen)
#         if StarPattern not in StarPatterns and StarPattern != '':
#             StarPatterns.append(StarPattern)
#             StarPatternDict[StarPattern] = [gen_sen]
#         elif StarPattern in StarPatterns and gen_sen not in StarPatternDict[StarPattern]:
#             StarPatternDict[StarPattern].append(gen_sen) 
#     #Generating WCL graph for each star pattern and appending to dictionary
#     wclGraphDict = StarPatternToGraph(tagged_sentences,StarPatternDict,StarPatterns)
                    

#     return wclGraphDict 

#WCL-1
def wcl1(tagged_sentences):
    #clustering sentences acc to Star Patterns
    StarPatterns = []
    StarPatternDict = {}
    for sent in tagged_sentences:
        strip_sent = strip_phrase_tags(sent)
        gen_sen = generalise_phrase(strip_sent)
        if augmentFlag == True:
            augmented = augment(gen_sen)
            augmented.append(gen_sen)
        else:
            augmented = gen_sen    
        for gen_sen in augmented:
            StarPattern = star_pattern(gen_sen)
            if StarPattern not in StarPatterns and StarPattern != '':
                StarPatterns.append(StarPattern)
                StarPatternDict[StarPattern] = [gen_sen]
            elif StarPattern in StarPatterns and gen_sen not in StarPatternDict[StarPattern]:
                StarPatternDict[StarPattern].append(gen_sen) 
    #Generating WCL graph for each star pattern and appending to dictionary
    wclGraphDict = StarPatternToGraph(tagged_sentences,StarPatternDict,StarPatterns)
                    

    return wclGraphDict,StarPatternDict

#Checking if a given tagged sentence is a definition acc to WCL-1 graph
# def check_definition_wcl1(tagged_sentence, wcl1_graph_dict):
#     #Initializing
#     isDefinition = False
#     #Finding Star pattern to which given sentence belongs
#     strip_sent = strip_phrase_tags(tagged_sentence)
#     gen_sen = generalise_phrase(strip_sent)
#     tokens_gen = gen_sen.strip().split()
#     StarPattern = star_pattern(gen_sen)
#     key = '<s>'
#     idx = 0
#     #following the graph for found Star Pattern
#     if StarPattern in wcl1_graph_dict:
#         graph = wcl1_graph_dict[StarPattern]
#         while (key != None) and (key != '</s>') and (idx < len(tokens_gen)):
            
#             if idx != (len(tokens_gen)-1):
#                 key  = find_key_continuity(graph, tokens_gen[idx], tokens_gen[idx+1], key)
#             elif idx == (len(tokens_gen)-1): 
#                 key = find_key(graph[key], tokens_gen[idx])
#             idx += 1
#             if key == None:
#                 break
        
#         if key !=  None: 
             
#             if '</s>' in graph[key]:
#                 isDefinition = True 

#     if isDefinition == True:
#         print 'The Sentence is classified as a True Definition according to WCL-1'
#         try:
#             print 'Target Word:', TaggedSentenceToTargetWord(tagged_sentence)
#             print 'Hypernym:' , TaggedSentToHypernym(tagged_sentence) 
#         except IndexError:
#             print "Hypernym not found"     


#     return isDefinition
    
#Checking if a given tagged sentence is a definition acc to WCL-1 graph
def check_definition_wcl1(tagged_sentence, wcl1_graph_dict,StarPatternDict):
    #Initializing
    isDefinition = False
    #Finding Star pattern to which given sentence belongs
    strip_sent = strip_phrase_tags(tagged_sentence)
    gen_sen = generalise_phrase(strip_sent)
    if augmentFlag == True:
        gen_sen=gen_sen.replace('NN NN','NN')
        gen_sen=gen_sen.replace('JJ JJ','JJ')
        gen_sen=gen_sen.replace('JJ JJ','JJ')
        gen_sen=gen_sen.replace('NN NN','NN')
        gen_sen=gen_sen.replace('JJ NN','NN')
        gen_sen=gen_sen.replace('RB VV','VV')
    tokens_gen = gen_sen.strip().split()
    StarPattern = star_pattern(gen_sen)
    key = '<s>'
    idx = 0
    #following the graph for found Star Pattern
    if StarPattern in wcl1_graph_dict:
        graph = wcl1_graph_dict[StarPattern]
        while (key != None) and (key != '</s>') and (idx < len(tokens_gen)):
            
            if idx != (len(tokens_gen)-1):
                key  = find_key_continuity(graph, tokens_gen[idx], tokens_gen[idx+1], key)
            elif idx == (len(tokens_gen)-1): 
                key = find_key(graph[key], tokens_gen[idx])
            idx += 1
            if key == None:
                break
        
        if key !=  None: 
             
            if '</s>' in graph[key]:
                isDefinition = True 

    if isDefinition == True:
        print 'The Sentence is classified as a True Definition according to WCL-1'
        try:
            print 'Target Word:', TaggedSentenceToTargetWord(tagged_sentence)
            print 'Hypernym:' , TaggedSentToHypernym(tagged_sentence) 
        except IndexError:
            print "Hypernym not found"     
    else:
        max_score=0
        if StarPattern in wcl1_graph_dict:
            for pattern_sent in StarPatternDict[StarPattern]:
                score, path = match(gen_sen,pattern_sent)
                if score > max_score:
                    max_score = score
        if max_score>0.25*len(tokens_gen):
            isDefinition=True

    return isDefinition


def check_wcl1_coverage(tagged_sentence,StarPatternDict):
    strip_sent = strip_phrase_tags(tagged_sentence)
    gen_sen = generalise_phrase(strip_sent)
    if augmentFlag == True:
        gen_sen=gen_sen.replace('NN NN','NN')
        gen_sen=gen_sen.replace('JJ JJ','JJ')
        gen_sen=gen_sen.replace('JJ JJ','JJ')
        gen_sen=gen_sen.replace('NN NN','NN')
        gen_sen=gen_sen.replace('JJ NN','NN')
        gen_sen=gen_sen.replace('RB VV','VV')
    tokens_gen = gen_sen.strip().split()
    StarPattern = star_pattern(gen_sen)
    max_score = 0.0
    for DictPattern in StarPatternDict:
        for gen_phrase in StarPatternDict[DictPattern]:
            if gen_phrase in gen_sen:
                coverage = float(len(DictPattern.strip().split()))/float(len(tokens_gen))
                log_support = np.log(len(StarPatternDict[DictPattern]))
                score = coverage*log_support
                max_score = max(score,max_score)
    return max_score            


#WCL-3
def wcl3(tagged_sentences):
    tagged_sentences_target = []
    tagged_sentences_verb = []
    tagged_sentences_genus = []
    

    for tagged_sentence in tagged_sentences:
        tagged_sentences_target.append(TaggedSentToTargetPhrase(tagged_sentence))
        tagged_sentences_verb.append(TaggedSentToVerbPhrase(tagged_sentence))
        tagged_sentences_genus.append(TaggedSentToGenusPhrase(tagged_sentence))

    #Star patterns for Target Phrases
    StarPatterns_target = []
    StarPatternDict_target = {}
    for sent in tagged_sentences_target:
        gen_sen = generalise_phrase(sent)
        if augmentFlag == True:
            augmented = augment(gen_sen)
            augmented.append(gen_sen)
        else:
            augmented = [gen_sen]
        for gen_sen in augmented:
            StarPattern = star_pattern(gen_sen)
            if StarPattern not in StarPatterns_target and StarPattern != '':
                StarPatterns_target.append(StarPattern)
                StarPatternDict_target[StarPattern] = [gen_sen]
            elif StarPattern in StarPatterns_target and gen_sen not in StarPatternDict_target[StarPattern]:
                StarPatternDict_target[StarPattern].append(gen_sen) 

    #Star patterns for Verb Phrases
    StarPatterns_verb = []
    StarPatternDict_verb = {}
    for sent in tagged_sentences_verb:
        gen_sen = generalise_phrase(sent)
        if augmentFlag == True:
            augmented = augment(gen_sen)
            augmented.append(gen_sen)
        else:
            augmented = [gen_sen]
        for gen_sen in augmented:
            StarPattern = star_pattern(gen_sen)
            if StarPattern not in StarPatterns_verb and StarPattern != '':
                StarPatterns_verb.append(StarPattern)
                StarPatternDict_verb[StarPattern] = [gen_sen]
            elif StarPattern in StarPatterns_verb and gen_sen not in StarPatternDict_verb[StarPattern]:
                StarPatternDict_verb[StarPattern].append(gen_sen)


    #Star patterns for Genus Phrases
    StarPatterns_genus = []
    StarPatternDict_genus = {}
    for sent in tagged_sentences_genus:
        gen_sen = generalise_phrase(sent)
        if augmentFlag == True:
            augmented = augment(gen_sen)
            augmented.append(gen_sen)
        else:
            augmented = [gen_sen]

        for gen_sen in augmented:
            StarPattern = star_pattern(gen_sen)
            if StarPattern not in StarPatterns_genus and StarPattern != '':
                StarPatterns_genus.append(StarPattern)
                StarPatternDict_genus[StarPattern] = [gen_sen]
            elif StarPattern in StarPatterns_genus and gen_sen not in StarPatternDict_genus[StarPattern]:
                StarPatternDict_genus[StarPattern].append(gen_sen)   


    #Generating graph dictionaries for all phrases
    wclGraphDict_target = StarPatternToGraph(tagged_sentences_target,StarPatternDict_target,StarPatterns_target)
    wclGraphDict_verb = StarPatternToGraph(tagged_sentences_verb,StarPatternDict_verb,StarPatterns_verb)
    wclGraphDict_genus = StarPatternToGraph(tagged_sentences_genus,StarPatternDict_genus,StarPatterns_genus)                      



    
    return wclGraphDict_target,wclGraphDict_verb,wclGraphDict_genus, StarPatternDict_target, StarPatternDict_verb, StarPatternDict_genus



def check_definition_wcl3_coverage(tagged_sentence, spTargetDict, spVerbDict, spGenusDict):
    #Intializing
    isDefinition_target = False
    isDefinition_verb = False
    isDefinition_genus = False

    strip_sent = strip_phrase_tags(tagged_sentence)
    gen_sen = generalise_phrase(strip_sent)
    if augmentFlag == True:
        gen_sen=gen_sen.replace('NN NN','NN')
        gen_sen=gen_sen.replace('JJ JJ','JJ')
        gen_sen=gen_sen.replace('JJ JJ','JJ')
        gen_sen=gen_sen.replace('NN NN','NN')
        gen_sen=gen_sen.replace('JJ NN','NN')
        gen_sen=gen_sen.replace('RB VV','VV')
    tokens_gen = gen_sen.strip().split()
    StarPattern = star_pattern(gen_sen)
    
    
    max_score = 0.0
    for DictPatternTgt in spTargetDict:
        for gen_phrase_tgt in spTargetDict[DictPatternTgt]:
            for DictPatternVerb in spVerbDict:
                for gen_phrase_verb in spVerbDict[DictPatternVerb]:
                    for DictPatternGenus in spGenusDict:
                        for gen_phrase_genus in spGenusDict[DictPatternGenus]:
                            if gen_phrase_tgt in gen_sen and gen_phrase_verb in gen_sen and gen_phrase_genus in gen_sen:
                                coverage = float(len(DictPatternTgt.strip().split())+len(DictPatternVerb.strip().split())+len(DictPatternGenus.strip().split()))/float(len(tokens_gen))
                                log_support = np.log(len(spTargetDict[DictPatternTgt]))+np.log(len(spVerbDict[DictPatternVerb]))+np.log(len(spGenusDict[DictPatternGenus]))
                                score = coverage*log_support
                                max_score = max(score,max_score)



    return max_score   

def check_definition_wcl3_multipleCoverage(tagged_sentence, spTargetDict, spVerbDict, spGenusDict):
    #Intializing
    isDefinition_target = False
    isDefinition_verb = False
    isDefinition_genus = False

    strip_sent = strip_phrase_tags(tagged_sentence)
    gen_sen = generalise_phrase(strip_sent)
    if augmentFlag == True:
        gen_sen=gen_sen.replace('NN NN','NN')
        gen_sen=gen_sen.replace('JJ JJ','JJ')
        gen_sen=gen_sen.replace('JJ JJ','JJ')
        gen_sen=gen_sen.replace('NN NN','NN')
        gen_sen=gen_sen.replace('JJ NN','NN')
        gen_sen=gen_sen.replace('RB VV','VV')
    tokens_gen = gen_sen.strip().split()
    StarPattern = star_pattern(gen_sen)
    key = '<s>'
    idx = 0
    max_score_target = 0
    
    max_len = 0.0 
    for DictPattern in spTargetDict:
        present =  0
       
        coverage_target = float(len(DictPattern.strip().split()))/float(len(tokens_gen))
        log_support = np.log(len(spTargetDict[DictPattern]))

        for gen_phrase in spTargetDict[DictPattern]:
            if gen_phrase in gen_sen:
                print gen_phrase
                present =  1
                score_target = coverage_target*log_support

                max_score_target = max(score_target, max_score_target)
              
    max_score_verb = 0
    max_len = 0.0
    for DictPattern in spVerbDict:
        present =  0
        

        coverage_verb = float(len(DictPattern.strip().split()))/float(len(tokens_gen))
        log_support = np.log(len(spVerbDict[DictPattern]))
        
        for gen_phrase in spVerbDict[DictPattern]:
            if gen_phrase in gen_sen:
                print gen_phrase
                present =  1
                score_verb = coverage_verb*log_support
                max_score_verb = max(score_verb, max_score_verb) 
        
    max_score_genus = 0
    max_len = 0.0
    for DictPattern in spGenusDict:
        present =  0
        
        coverage_genus = float(len(DictPattern.strip().split()))/float(len(tokens_gen))
        log_support = np.log(len(spGenusDict[DictPattern]))
        
        for gen_phrase in spGenusDict[DictPattern]:
            if gen_phrase in gen_sen:
                print gen_phrase
                present =  1
                score_genus = coverage_genus*log_support
                max_score_genus = max(score_genus, max_score_genus) 
        
    score = max_score_target/coverage_target+ max_score_verb/coverage_verb + max_score_genus/coverage_genus -(1.0-(coverage_target+coverage_verb+coverage_genus))
    return score   


        
      


def check_definition_wcl3(tagged_sentence, wcl3TargetDict, wcl3VerbDict, wcl3GenusDict):
    #Intializing
    isDefinition_target = False
    isDefinition_verb = False
    isDefinition_genus = False
    
    #Getting phrases
    target_phrase = TaggedSentToTargetPhrase(tagged_sentence)
    verb_phrase = TaggedSentToVerbPhrase(tagged_sentence)
    genus_phrase = TaggedSentToGenusPhrase(tagged_sentence)


    #Target Phrase
    #Getting star pattern from Target phrase
    gen_sen = generalise_phrase(target_phrase)
    tokens_gen = gen_sen.strip().split()
    StarPattern = star_pattern(gen_sen)
    key = '<s>'
    idx = 0
    #Following graph of StarPattern
    if StarPattern in wcl3TargetDict:
        graph = wcl3TargetDict[StarPattern]
        while (key != None) and (key != '</s>') and (idx < len(tokens_gen)):
            
            if idx != (len(tokens_gen)-1):
                key  = find_key_continuity(graph, tokens_gen[idx], tokens_gen[idx+1], key)
            elif idx == (len(tokens_gen)-1): 
                key = find_key(graph[key], tokens_gen[idx])
            
            idx += 1
            if key == None:
                break
        
        if key !=  None: 
            
            if '</s>' in graph[key]:
                isDefinition_target = True  
    #If phrase doesn't exist in given sentence, Ignoring comparison
    elif StarPattern == '':
        isDefinition_target = True             
    #Verb Phrase        
    #Getting star pattern from Verb phrase
    gen_sen = generalise_phrase(verb_phrase)
    tokens_gen = gen_sen.strip().split()
    StarPattern = star_pattern(gen_sen)
    key = '<s>'
    idx = 0
    #Following graph of StarPattern
    if StarPattern in wcl3VerbDict:
        graph = wcl3VerbDict[StarPattern]
        while (key != None) and (key != '</s>') and (idx < len(tokens_gen)):
            
            if idx != (len(tokens_gen)-1):
                key  = find_key_continuity(graph, tokens_gen[idx], tokens_gen[idx+1], key)
            elif idx == (len(tokens_gen)-1): 
                key = find_key(graph[key], tokens_gen[idx])
            idx += 1
            if key == None:
                break
        
        if key !=  None: 
               
            if '</s>' in graph[key]:
                isDefinition_verb = True  
    #If phrase doesn't exist in given sentence, Ignoring comparison
    elif StarPattern == '':
        isDefinition_verb = True             

    #Genus phrase
    #Getting star pattern of genus phrase    
    gen_sen = generalise_phrase(genus_phrase)
    tokens_gen = gen_sen.strip().split()
    StarPattern = star_pattern(gen_sen)
    key = '<s>'
    idx = 0
    #Following graph of star pattern
    if StarPattern in wcl3GenusDict:
        graph = wcl3GenusDict[StarPattern]
        while (key != None) and (key != '</s>') and (idx < len(tokens_gen)):
            
            if idx != (len(tokens_gen)-1):
                key  = find_key_continuity(graph, tokens_gen[idx], tokens_gen[idx+1], key)
            elif idx == (len(tokens_gen)-1): 
                key = find_key(graph[key], tokens_gen[idx])
            idx += 1
            if key == None:
                break
        
        if key !=  None: 
            if '</s>' in graph[key]:
                isDefinition_genus = True 
    #If phrase doesn't exist in given sentence, Ignoring comparison            
    elif StarPattern == '':
        isDefinition_genus = True             

    isDefinition = isDefinition_target and isDefinition_verb and isDefinition_genus 
    if isDefinition == True:
        print 'The Sentence is classified as a True Definition according to WCL-3'
        try:
            print 'Target Word:', TaggedSentenceToTargetWord(tagged_sentence)
            print 'Hypernym:' , TaggedSentToHypernym(tagged_sentence) 
        except IndexError:
            pass                 
    return isDefinition
            
def check_definition_wcl3_gen(tagged_sentence, wcl3TargetDict, wcl3VerbDict, wcl3GenusDict):
    #Intializing
    isDefinition_target = False
    isDefinition_verb = False
    isDefinition_genus = False
    
    #Getting phrases
    # target_phrase = TaggedSentToTargetPhrase(tagged_sentence)
    # verb_phrase = TaggedSentToVerbPhrase(tagged_sentence)
    # genus_phrase = TaggedSentToGenusPhrase(tagged_sentence)
    sent_strip = strip_phrase_tags(tagged_sentence)
    sent_list = sent_strip.split()
    vp_index=-1
    for i in range(len(sent_list)):
        try:
            if phrase(sent_list[i])=='VP':
                vp_index=i
                target_phrase = ' '.join(sent_list[1:vp_index])
                verb_phrase = sent_list[vp_index]
                genus_phrase = ' '.join(sent_list[vp_index+1:])
                # f.write(target_phrase+' X '+verb_phrase+' X '+genus_phrase+'\n')
                #Target Phrase
                #Getting star pattern from Target phrase
                gen_sen = generalise_phrase(target_phrase)
                tokens_gen = gen_sen.strip().split()
                StarPattern = star_pattern(gen_sen)
                key = '<s>'
                idx = 0
                #Following graph of StarPattern
                if StarPattern in wcl3TargetDict:
                    graph = wcl3TargetDict[StarPattern]
                    while (key != None) and (key != '</s>') and (idx < len(tokens_gen)):
                        
                        if idx != (len(tokens_gen)-1):
                            key  = find_key_continuity(graph, tokens_gen[idx], tokens_gen[idx+1], key)
                        elif idx == (len(tokens_gen)-1): 
                            key = find_key(graph[key], tokens_gen[idx])
                        
                        idx += 1
                        if key == None:
                            break
                    
                    if key !=  None: 
                        
                        if '</s>' in graph[key]:
                            isDefinition_target = True  
                #If phrase doesn't exist in given sentence, Ignoring comparison
                elif StarPattern == '':
                    isDefinition_target = True             
                #Verb Phrase        
                #Getting star pattern from Verb phrase
                gen_sen = generalise_phrase(verb_phrase)
                tokens_gen = gen_sen.strip().split()
                StarPattern = star_pattern(gen_sen)
                key = '<s>'
                idx = 0
                #Following graph of StarPattern
                if StarPattern in wcl3VerbDict:
                    graph = wcl3VerbDict[StarPattern]
                    while (key != None) and (key != '</s>') and (idx < len(tokens_gen)):
                        
                        if idx != (len(tokens_gen)-1):
                            key  = find_key_continuity(graph, tokens_gen[idx], tokens_gen[idx+1], key)
                        elif idx == (len(tokens_gen)-1): 
                            key = find_key(graph[key], tokens_gen[idx])
                        idx += 1
                        if key == None:
                            break
                    
                    if key !=  None: 
                           
                        if '</s>' in graph[key]:
                            isDefinition_verb = True  
                #If phrase doesn't exist in given sentence, Ignoring comparison
                elif StarPattern == '':
                    isDefinition_verb = True             

                #Genus phrase
                #Getting star pattern of genus phrase    
                gen_sen = generalise_phrase(genus_phrase)
                tokens_gen = gen_sen.strip().split()
                StarPattern = star_pattern(gen_sen)
                key = '<s>'
                idx = 0
                #Following graph of star pattern
                if StarPattern in wcl3GenusDict:
                    graph = wcl3GenusDict[StarPattern]
                    while (key != None) and (key != '</s>') and (idx < len(tokens_gen)):
                        
                        if idx != (len(tokens_gen)-1):
                            key  = find_key_continuity(graph, tokens_gen[idx], tokens_gen[idx+1], key)
                        elif idx == (len(tokens_gen)-1): 
                            key = find_key(graph[key], tokens_gen[idx])
                        idx += 1
                        if key == None:
                            break
                    
                    if key !=  None: 
                        if '</s>' in graph[key]:
                            isDefinition_genus = True 
                #If phrase doesn't exist in given sentence, Ignoring comparison            
                elif StarPattern == '':
                    isDefinition_genus = True             

                isDefinition = isDefinition_target and isDefinition_verb and isDefinition_genus 
                if isDefinition == True:
                    print 'The Sentence is classified as a True Definition according to WCL-3'
                    return True
                    try:
                        print 'Target Word:', TaggedSentenceToTargetWord(tagged_sentence)
                        print 'Hypernym:' , TaggedSentToHypernym(tagged_sentence) 
                    except IndexError:
                        pass                 
        except AttributeError:
                    pass
    
    return False
# 

#############################################################################
# Functions for proposal implementations
def augment(gen_sent):
    # new_data = list(data)
    new_sent=[]
    new_sent.append(gen_sent.replace('NN NN','NN'))
    new_sent.append(gen_sent.replace('JJ NN','NN'))
    new_sent.append(gen_sent.replace('a NN','NN'))
    new_sent.append(gen_sent.replace('the NN','NN'))
    new_sent.append(gen_sent.replace('the TARGET','TARGET'))
    new_sent.append(gen_sent.replace('a TARGET','TARGET'))
    new_sent.append(gen_sent.replace('an NN','NN'))
    new_sent.append(gen_sent.replace('an TARGET','TARGET'))
    new_sent.append(gen_sent.replace('NN','NN NN'))
    new_sent.append(gen_sent.replace('NN','JJ NN'))
    new_sent.append(gen_sent.replace('NN','a NN'))
    new_sent.append(gen_sent.replace('NN','the NN'))
    new_sent.append(gen_sent.replace('TARGET','the TARGET'))
    new_sent.append(gen_sent.replace('TARGET','a TARGET'))
    new_sent.append(gen_sent.replace('NN','an NN'))
    new_sent.append(gen_sent.replace('TARGET','an TARGET'))
    new_sent.append(gen_sent.replace('JJ JJ','JJ'))
    new_sent.append(gen_sent.replace('JJ','JJ JJ'))
    new_sent.append(gen_sent.replace('NNS','NN'))
    new_sent.append(gen_sent.replace('NN','NNS'))
    new_sent.append(gen_sent.replace('the','a'))
    new_sent.append(gen_sent.replace(' a ',' the '))
    new_sent.append(gen_sent.replace('NN NN NN','NN'))
    new_sent.append(gen_sent.replace('NN','NN NN NN'))
    new_sent.append(gen_sent.replace('NN','NP'))
    new_sent.append(gen_sent.replace('NP','NN'))
    new_sent.append(gen_sent.replace('NP NP','NP'))
    new_sent.append(gen_sent.replace('NP','NP NP'))
    new_sent.append(gen_sent.replace('RB VV','VV'))
    return new_sent     

# ###########################################################################3

# Graph creation 
# Shuffle split performs well for WCL1               
random.seed(42)
random.shuffle(tagged_sentences)
# random.seed(42)
# random.shuffle(sentences)
total_correct_sentences = len(tagged_sentences)
# Splitting test sentences and training sentences
tagged_sentences_test_pos = tagged_sentences[int(total_correct_sentences*0.5):] # Test set

# Training set
tagged_sentences_train = tagged_sentences[:int(total_correct_sentences*0.5)]
#stop_words=frequent_set(sentences[:int(len(sentences)*0.5)],'default')
# Create WCL

wcl1_graph, star_dict =  wcl1(tagged_sentences_train)
# # print check_definition_wcl1(tagged_sentences[11], wcl1_graph)
# print 'WCL1 creation completed'




# # Evaluation Part
# for i in range(len(tagged_sentences_train)):
#     f.write(tagged_sentences_train[i]+"\n")
#     f.write(strip_phrase_tags(tagged_sentences_train[i])+"\n")
#     f.write(generalise_phrase(strip_phrase_tags(tagged_sentences_train[i]))+"\n\n")

# Use wiki_bad
with open(str(sys.argv[2]),'r') as fp:
    lines=fp.readlines()
    lines = [line.replace('\n','') for line in lines]
    # print('\n'.join(lines))
    
    for index,line in enumerate(lines):
        if index%2==0:
            sentences_bad.append(line[2:])
        else:
            parts=line.split(':')
            targets_bad.append(parts[0])
            tagged_sentences_bad.append(line[len(parts[0])+1:])
total_wrong_sentences = len(tagged_sentences_bad)



min_s = np.inf
max_s = 0

# for i in range(len(tagged_sentences_test_pos)):
#     print i, len(tagged_sentences)
#     s =  check_wcl1_coverage(tagged_sentences_test_pos[i], star_dict)
#     max_s = max(s,max_s)
#     min_s = min(s,min_s)
#     print max_s, min_s
#     f.write("%d %d\r\n" % (i,s))

# tagged_sentences_bad=tagged_sentences_bad[:len(tagged_sentences_test_pos)]
# Finding TP and FN

# Test WCL1

tp=0
fn=0
tn=0
fp=0
threshold = 0.35

# True positives
for i in range(len(tagged_sentences_test_pos)):
    print i, len(tagged_sentences_test_pos)
    if check_wcl1_coverage(tagged_sentences_test_pos[i], star_dict) > threshold:
        tp+=1
    else:
        fn+=1
        # f.write(tagged_sentences_test_pos[i]+"\n")
        # f.write(strip_phrase_tags(tagged_sentences_test_pos[i])+"\n")
        f.write(generalise_phrase(strip_phrase_tags(tagged_sentences_test_pos[i]))+"\n\n")


# True Negatives
for i in range(len(tagged_sentences_bad)):
    print i, len(tagged_sentences_bad)
    if check_wcl1_coverage(tagged_sentences_bad[i], star_dict) > threshold:
        fp+=1
        # f.write(tagged_sentences_bad[i]+"\n")
    else:
        tn+=1

print tp,fn,tn,fp
recall = float(tp)/(tp+fn)
precision = float(tp)/(tp+fp)

f_score = 2.0*recall*precision/(recall+precision)
# print len(tagged_sentences_test_pos)
print 'WCL-1\nPrecision: {0}\nRecall: {1}\nF Score: {2}\n'.format(precision,recall,f_score)
# # print len(stop_words)
# print stop_words







# ##############################################################
# WCL 3
min_s = np.inf
max_s = 0
_,_,_,sp_graph_target,sp_graph_verb,sp_graph_genus =  wcl3(tagged_sentences_train)
# for i in range(len(tagged_sentences_test_pos)):
#     print i, len(tagged_sentences)
#     s =  check_definition_wcl3_new(tagged_sentences_test_pos[i], sp_graph_target,sp_graph_verb,sp_graph_genus)
#     max_s = max(s,max_s)
#     min_s = min(s,min_s)
#     print max_s, min_s
#     f.write("%d %d\r\n" % (i,s))


# Test WCL3

# tp=0
# fn=0
# tn=0
# fp=0
# threshold=2.0
# # while threshold<=5:
#     # threshold = 3.0
# tp=0
# fn=0
# tn=0
# fp=0
# # True positives
# for i in range(len(tagged_sentences_test_pos)):
#     print i, len(tagged_sentences_test_pos)
#     if check_definition_wcl3_coverage(tagged_sentences_test_pos[i], sp_graph_target,sp_graph_verb,sp_graph_genus) > threshold:
#         tp+=1
#     else:
#         fn+=1
#         # f.write(tagged_sentences_test_pos[i]+"\n")
#         # f.write(strip_phrase_tags(tagged_sentences_test_pos[i])+"\n")


# # True Negatives
# for i in range(len(tagged_sentences_bad)):
#     print i, len(tagged_sentences_bad)
#     if check_definition_wcl3_coverage(tagged_sentences_bad[i], sp_graph_target,sp_graph_verb,sp_graph_genus) > threshold:
#         fp+=1
#         f.write(generalise_phrase(strip_phrase_tags(tagged_sentences_bad[i]))+"\n\n")

#         # f.write(tagged_sentences_bad[i]+"\n")
#     else:
#         tn+=1
# # with open('resultsWCL3','a') as g:
# print tp,fn,tn,fp
# # g.write('{0}\n{1} {2} {3} {4}\n'.format(threshold,tp,fn,tn,fp))
# recall = float(tp)/(tp+fn)
# precision = float(tp)/(tp+fp)

# f_score = 2.0*recall*precision/(recall+precision)
# # print len(tagged_sentences_test_pos)
# print 'WCL-3\nPrecision: {0}\nRecall: {1}\nF Score: {2}\n'.format(precision,recall,f_score)
    # g.write('WCL-3\nPrecision: {0}\nRecall: {1}\nF Score: {2}\n\n'.format(precision,recall,f_score))
    # threshold+=0.25
    # # print len(stop_words)
    # print stop_words




# for i in range(1872):
#     print i, len(tagged_sentences)
#     if check_definition_wcl1(tagged_sentences[i], wcl1_graph) == False:
#         f.write("%d\r\n" % (i))

