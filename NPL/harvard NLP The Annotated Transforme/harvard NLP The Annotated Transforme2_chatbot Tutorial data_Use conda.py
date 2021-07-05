# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 16:22:11 2020

@author: Qiandehou
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")
#matplotlib inline



from torch.jit import script,trace

from torch import optim

import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools



#Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

USE_CUDA=torch.cuda.is_available()
print(USE_CUDA)
device=torch.device("cuda" if USE_CUDA else "cpu")
print(device)
#Load&Preprocess Data

# The next step is to reformat our data file and load the data into structures that we can work with
# The cornel Movie_Dialogs Corpus is a rich dataset of movie character dialog
# 1. 220579 conversational exchanges between 10292 pairs of movie characters
# 2. 9035 characters from 617 movies
# 3. 304713 total utterances

#This dataset is large and diverse , and there is a great variation of language formality, time periods,sentiment etc.
#Our hope is that this diversity makes our model robust to many forms of inputs and quiries

#First, we'll take a look at some lines of our datafile to see the original format

corpus_name="cornell movie-dialogs corpus"
corpus=os.path.join("data\cornell_movie_dialogs_corpus",corpus_name)

def printLines(file,n=10):
    with open(file,'rb')as datafile:# 读取二进制文件,例如图片,视频等,使用'rb'模式打开文件
        lines=datafile.readlines()
    for line in lines[:n]:
        print(line)

print(corpus)
printLines(os.path.join(corpus,"movie_lines.txt"))

#Create formatted data file
#For convenicence, we'll create a nicely formatted data file in which each line contains a tab-seperated
#query sentence and a response sentence pair

# The following functions facilitate the parsing of the raw movie_lines.txt data file.
# 1. loadLines splits each line of the file into a dictionary of fields(lineID,characterID,movieID,character,text)
# 2. loadConversations groups fields of lines from loadLines into conversations based on movie_conversations.txt
# 3. extractSentencePairs extracts pairs of sentences from conversations

# Splits each line of the file into a dictionary of fields
def loadLines(fileName, fields):
    lines={}
    with open(fileName,'r',encoding='iso-8859-1')as f:
        for line in f:
            values=line.split(" +++$+++ ")
            # Extract fields
            lineObj={}
            for i,field in enumerate(fields):
            #MOVIE_LINES_FIELDS=["lineID","characterID","movieID","character","text"]
#                print("line:{}".format(i))
#                print("fiedld:{}".format(field))
                lineObj[field]=values[i]
            lines[lineObj['lineID']]=lineObj
    return lines

 #Groups fields of lines from 'loadLines' into conversations based on *movie_conversations.txt*
def loadConversations(fileName,lines,fields):
    conversations=[]
    #count=1
    with open(fileName,'r',encoding='iso-8859-1')as f:
        for line in f:
            #count=count+1
            values=line.split(" +++$+++ ")
             #Extract fields
            convObj={}
            for i,field in enumerate(fields):
            #MOVIE_CONVERSATIONS_FIELDS=["character1ID","character2ID","movieID","utteranceIDs"]
                convObj[field]=values[i]
                 #Convert string list (convObj["utteranceIDs"]=="['L598485','L598486',]")
            
            #debug: pay attention to the for circle layout
            utterance_id_pattern=re.compile('L[0-9]+') 
            #编译正则表达式模式，返回一个对象。可以把常用的正则表达式编译成正则表达式对象，方便后续调用及提高效率。
            #print(utterance_id_pattern)
            
            lineIds=utterance_id_pattern.findall(convObj["utteranceIDs"])
            #utterance: 用言语的表达; 说话; 话语; 言论
                 #Reassemble lines
            convObj["lines"]=[]
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    #print(count)
    return conversations



#Extracts pairs of sentences from conversations
def extractSentencePairs(conversations):
    qa_pairs=[]
    for conversation in conversations:
        #Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"])-1): #We ignore the last line(no answer for it)
            #print("len(inputLine):{}".format(len(conversation["lines"][i]["text"])))
            inputLine=conversation["lines"][i]["text"].strip()
            #print("len(inputLine):{}".format(len(inputLine)))
#            print("inputLine:{}".format(inputLine))
            targetLine=conversation["lines"][i+1]["text"].strip()
#--------------------------------------------------------------------------
#            str = "00000003210Runoob01230000000"; 
#            print str.strip( '0' );  # 去除首尾字符 0
# 
# 
#            str2 = "   Runoob      ";   # 去除首尾空格
#            print str2.strip();
#---------------------------------------------------------------------------------
            #print("targetLine:{}".format(targetLine))
            #fileter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine,targetLine])
    return qa_pairs

# Now we'll call these functions and create the file. We'll call it formatted_movie_lines.txt

#Define path to new file
datafile=os.path.join(corpus,"formatted_movie_lines.txt")

delimiter='\t' #delimiter: 定界符，分隔符;

# Unescape the delimiter
delimiter=str(codecs.decode(delimiter,"unicode_escape"))
print("delimiter:{}".format(delimiter))
# Initialize lines dict, conversations list, and field ids
lines={}
conversations=[]
MOVIE_LINES_FIELDS=["lineID","characterID","movieID","character","text"]
MOVIE_CONVERSATIONS_FIELDS=["character1ID","character2ID","movieID","utteranceIDs"]

#Load lines and process conversations
print("\nProcessing corpus...")
lines=loadLines(os.path.join(corpus,"movie_lines.txt"),MOVIE_LINES_FIELDS)
#Load lines and process conversations
print("\nProcessing conversations...")
conversations=loadConversations(os.path.join(corpus,"movie_conversations.txt"),
                lines,MOVIE_CONVERSATIONS_FIELDS)

#Write new csv file
print("\nWriting newly formatted file...")
with open(datafile,'w',encoding='utf-8')as outputfile:
    writer=csv.writer(outputfile,delimiter=delimiter,lineterminator='\n')
    for pair in extractSentencePairs(conversations):
        writer.writerow(pair)
#don't find anything according to the writer 2020-1-4
        
#Print a sample of lines
print("\nSample lines from file:")
printLines(datafile)

# Load and trim data

# Our next order of business is to create a vocabulary and load query/response sentence
# pairs into memory.

# Note that we are dealing with sequence of words, which do not have an implicit
# mapping to a discrete numerical space.Thus, we must create one by mapping each
# unique word that we encounter in our dataset to an index value.

# For this we define a Voc class, which keeps a mapping from words to indexes. a
# reverse mapping of indexes to words, a count of each word and a total word count.
# The class provides methods for adding a word to the vocabulary(addWord), adding
# all words in a sentence(addSentence) and trimming infrequently Seen words(trim).
# More on trimming later.

# Default word tokens
PAD_token=0 #Used for padding short sentences
SOS_token=1 #Start-of-sentence token
EOS_token=2 #End-of-sentence token

class Voc:
    def __init__(self,name):
        self.name=name
        self.trimmed=False
        self.word2index={}
        self.word2count={}
        self.index2word={PAD_token:"PAD",SOS_token:"SOS",EOS_token:"EOS"}
        self.num_words=3 #Count SOS,EOS,PAD
        
    def addSentence(self,sentence):
        for word in sentence.split(' '):
            self.addWord(word)
            
    def addWord(self,word):
        #print("word:{}".format(word))
        #print("word2index:{}".format(self.word2index))
        #print("word2count:{}".format(self.word2count))
        if word not in self.word2index:
            self.word2index[word]=self.num_words
            self.word2count[word]=1;
            self.index2word[self.num_words]=word
            self.num_words+=1
        else:
            self.word2count[word]+=1
    
    #Remove words below a certain count threshold
    def trim(self,min_count):
        if self.trimmed:
            return 
        self.trimmed=True
        
        keep_words=[]
        
        for k,v in self.word2count.items():
            #print("word2count.items():{}".format(self.word2count.items()))
            if v>=min_count:
                keep_words.append(k)
                
        print('keep_words {}/{}={:.4f}'.format(
                len(keep_words),len(self.word2index),len(keep_words)
                /len(self.word2index)))
        
        #Reinitialize dictionaries
        self.word2index={}
        self.word2count={}
        self.index2word={PAD_token:"PAD",SOS_token:"SOS",EOS_token:"EOS"}
        self.num_words=3 #Count default tokens
        
        for word in keep_words:
            self.addWord(word)
            
#Now we can assemble our vocabulary and query/response sentence Pairs.Before we are ready
#to use this data,we must perform some processing.
            
#First, we must convert the Unicode strings to ASCII using unicodeToAscii. Next,we should convert all letters
#to lowercase and trim all non-letter characters execpt for basic punctuation(normalizeString).Finally , to 
# aid in training convergence, we will filter out sentences with length greater than the MAX_LENGTH threshold(filterPairs)
    
MAX_LENGTH=30 # Maximum Sentence length to consider

# Turn a Unicode string to plain ACSII, thanks to consider
# https:\\stackoverflow.com/a/518232/2009427

# def unicodeToAscii(s):
#     return ' '.join(
#             c for c in unicodedata.normalize('NFD',s)
#             if unicodedata.category(c)!='Mn'
#             )

#delete the whitespace
def unicodeToAscii(s):
    return ''.join(
             c for c in unicodedata.normalize('NFD',s)
             if unicodedata.category(c)!='Mn'
             )
  
#Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s=unicodeToAscii(s.lower().strip())
#--------------------------------------------------------------------------------
#    re.sub(pattern, repl, string, count=0, flags=0)

#       pattern：表示正则表达式中的模式字符串；

#       repl：被替换的字符串（既可以是字符串，也可以是函数）；

#       string：要被处理的，要被替换的字符串；

#       count：匹配的次数, 默认是全部替换

#       flags：具体用处不详
#----------------------------------------------------------------------------------
    s=re.sub(r"([.!?])", r" \1",s)
    s=re.sub(r"[^a-zA-Z.!?]+",r" ",s)
    s=re.sub(r"\s+",r" ",s).strip()
    return s

#Read query/response pairs and return a voc object
def readVocs(datafile,corpus_name):
    print("Reading lines...")
    #Read the file and split into lines
    lines=open(datafile,encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs=[[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc=Voc(corpus_name)
    return voc,pairs

#Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    #Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' '))<MAX_LENGTH and len(p[1].split(' '))<MAX_LENGTH

#Filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

#Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus,corpus_name,datafile,save_dir):
    print("Start preparing training data...")
    voc,pairs=readVocs(datafile,corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs=filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:",voc.num_words)
    print("Counted pairs:",len(pairs))
    return voc,pairs

#Load/Assemble voc and pairs
save_dir=os.path.join("data","save")
voc,pairs=loadPrepareData(corpus,corpus_name,datafile,save_dir)
#Print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)
    
#Another tactic that is beneifcial to achieving faster convergence during training is trimming 
#rarely used words out of our vocabulary.Decreasing the feature space will also soften the 
#diffculty of the function that the model must learn to approximate.
#We will do this as a two step process:
    
# 1.Trim words used under MIN_COUNT threshold using the voc.trim function
# 2.Filter out pairs with trimmed words

MIN_COUNT=3 #Minimum word count threshold for trimming

def trimRareWords(voc,pairs,MIN_COUNT):
    #Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    #Filter out pairs with trimmed words
    keep_pairs=[]
    for pair in pairs:
        input_sentence=pair[0]
        output_sentence=pair[1]
        keep_input=True
        keep_output=True
        #Check input sentence
        for word in input_sentence.split(' '):
            #print("for word in input_sentence.split(' '):",word)
            if word not in voc.word2index:
                keep_input=False
                break
        
        #Check Output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output=False
                break
            
        #Only keep pairs that do not contain trimmed word(s) in their input or output
        #sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)
        
    print("Trimmed from {} pairs to {},{:.4f} of total".format(len(pairs),len(keep_pairs),
          len(keep_pairs)/len(pairs)))
    return keep_pairs

#Trim voc and pairs
pairs=trimRareWords(voc,pairs,MIN_COUNT)

#Preoare Data for Models\
#Although we have put a great deal of effort into preparing and massaging our data
#into a nice vocabulary object and list of sentence pairs, our models will ultimately
#expect numerical torch tensor as inputs.One way to prepare the processed data for
#the methods can be found in the seq2seq translation tutorial.In that tutorial ,we 
#use a batch size 1,meaning that all we have to do is convert the words in our sentence
# pairs to their corresponding indexex from the vocabulary and feed this to the models
#
#However, if you're interested in speeding up training and/or would like to leverage
#GPU parallelization , you will need to train with mini-batches.
#Using mini-batches also means that we must be mindful of the variation of sentence length
#in our batches. To accomodate sentence of different sizes in the same batch, we will
#make our batched input tensor of shape(max_length,batch_size),where sentences shorter
#than the max_length are zeros padded after an EOS_token.

#If we simply convert to our English sentence to tensors by converting words
#to their indexes(indexesFromSentence) and zero-pad, our tensor would have shape
#(batch_size,max_length) and indexing the first dimension would return a full sequence
#across all time-steps. However, we need to be able to index our batch along time,
#and across all sequences in the batch.Therefore, we transpose our input batch shape
#to (max_length,batch_size),so that indexing across the first dimension returns a time
#step across all sentences in the batch.we handle this transpose implicity in the
#zeropadding function

#The inputVar function handles the process of converting sentences to tensor,ultimately \
#creating a correctly shaped zerp-padded tensor.It also returns a tensor of lengths
#for each of the sequences in the batch which will be passed to our decoder later

#The outputVar function performs a similar to inputVar,but instead of returning a lengths
#tensor, it returns a binary mask tensor and a maximum target sentence length.The binary
#mask tensor has the same shape as the output target tensor, but every element that is a 
#PAD_token is 0 and all others are 1

#Batch2TrainData simply takes a bunch of pairs and returns the input and target tensors
#using the aforementioned functions
def indexesFromSentence(voc,sentence):
    return [voc.word2index[word] for word in sentence.split(' ')]+[EOS_token]

def zeroPadding(l,fillvalue=PAD_token):
    return list(itertools.zip_longest(*l,fillvalue=fillvalue))

def binaryMatrix(l,value=PAD_token):
    m=[]
    for i,seq in enumerate(l):
        m.append([])
        for token in seq:
            if token==PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

#Returns padded input sequence tensor and lengths
def inputVar(l,voc):
    indexes_batch=[indexesFromSentence(voc,sentence) for sentence in l]
    lengths=torch.tensor([len(indexes)for indexes in indexes_batch])
#    print("indexes_batch",indexes_batch)
    padList=zeroPadding(indexes_batch)
#    print("padList",padList)
    padVar=torch.LongTensor(padList)
    return padVar,lengths

def outputVar(l,voc):
    indexes_batch=[indexesFromSentence(voc,sentence) for sentence in l]
    max_target_len=max([len(indexes) for indexes in indexes_batch])
    padList=zeroPadding(indexes_batch)
    mask=binaryMatrix(padList)
    #print("mask:",mask)
    mask=torch.BoolTensor(mask)
    padVar=torch.LongTensor(padList)
    return padVar,mask,max_target_len

#Return all items for a  given batch of pairs
def batch2TrainData(voc,pair_batch):
    pair_batch.sort(key=lambda x:len(x[0].split(" ")),reverse=True)
    input_batch,output_batch=[],[]
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp,lengths=inputVar(input_batch,voc)
    output,mask,max_target_len=outputVar(output_batch,voc)
    
    #----------------------------------------------------------------------------------------------
    
    inp=torch.cat((inp,torch.zeros(MAX_LENGTH-inp.size(0),inp.size(1),dtype=torch.long)),0)
    output=torch.cat((output,torch.zeros(MAX_LENGTH-output.size(0),output.size(1),dtype=torch.long)),0)
    return inp.to(device),lengths.to(device),output.to(device)

 #Example for validation
#-------------------------------------------------------------------------------


#model part
#-------------------------------------------------------------------------------

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
     #nn.Sequential
     #A sequential container.
     #Modules will be added to it in the order they are passed in the constructor. 
     #Alternatively, an ordered dict of modules can also be passed in.
     
     
     # model = EncoderDecoder(
     #    Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
     #    Decoder(DecoderLayer(d_model, c(attn), c(attn), 
     #                         c(ff), dropout), N),
     #    nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
     #    nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
     #    Generator(d_model, tgt_vocab))
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
    
    
# nn.moduleList nn.Sequential    
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

#Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)    

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    
    
    
# attn = MultiHeadedAttention(h, d_model)
# ff = PositionwiseFeedForward(d_model, d_ff, dropout) 

# Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)   
# EncoderLayer(d_model, c(attn), c(ff), dropout) 
  
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # sublayer is residual connection
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        #SublayerConnection
        #forward:
        #  def forward(self, x, sublayer):
        #    "Apply residual connection to any sublayer with the same size."
        #  return x + self.dropout(sublayer(self.norm(x)))
        
        #sublayer= self.feed_forward = c(ff)
        # x=src     mask=src_mask
        
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
    
    
#self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
#Decoder(DecoderLayer(d_model, c(attn), c(attn), 
#                               c(ff), dropout), N),    
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
#x= tgt_embed(tgt)
#memory= self.encode(src, src_mask) 
#src_mask = src_mask
#tgt_mask = tgt_mask
       
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
    
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0



# plt.figure(figsize=(5,5))
# plt.imshow(subsequent_mask(20)[0])
# None    


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # print('scores.size():',scores.size())
    # print('mask.size():',mask.size())
    # print('-'*30)
    
    #\（处于行尾位置）	续行符
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
      
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# attn = MultiHeadedAttention(h, d_model)
# h=8
# d_model=512
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h# /整数除法    //浮点数除法
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

# encoderlayer.feed_forward():
# x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))

# x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        #zip(self.linears, (query, key, value))是把(self.linears[0],self.linears[1],
        #self.linears[2])和(query, key, value)放到一起然后遍历
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
    
    
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
    
class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
    
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
    
    
# model = make_model(V, V, N=2)   
def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(src_vocab,d_model), c(position)),
        nn.Sequential(Embeddings(src_vocab,d_model), c(position)),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model




class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            # self.trg = trg[:, :-1]
            # self.trg_y = trg[:, 1:]
            self.trg=trg
            self.trg_y=trg
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
    
        # run_epoch(data,targets, model, 
        #       SimpleLossCompute(model.generator, criterion, model_opt))
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask    
        
def run_epoch_train(training_batches, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    
    for iteration in range(start_iteration,n_iteration+1):
        training_batch=training_batches[iteration-1]
        input_variable,lengths,target_variable=training_batch
        #data, targets = get_batch(train_data, i)
        data=input_variable
        targets=target_variable

        #处理成长度相同 MAX_LENGTH 
        #yy=torch.cat((targets,torch.zeros(10-targets.size(0),20)),0)
        #torch.zeros()生成的默认张量数据类型，该处为float32,
        #但embedding需要的是long型，因此会报错
        #通过 AGU： dtype=torch.long进行设置
        
        #此处使用以下代码，在CUDA中运行会报错
        #推测是因为torch.zeros()生成的数据在cpu中，因此需要在前面的batch2Traindata中就处理完成并to(device)
        
         # data=torch.cat((data,torch.zeros(MAX_LENGTH-data.size(0),batch_size
         #                                  ,dtype=torch.long)),0)
         # targets=torch.cat((targets,torch.zeros(MAX_LENGTH-targets.size(0),batch_size
         #                                        ,dtype=torch.long)),0)   
        
        
        batch=Batch(data,targets,0)
        
    # for i, batch in enumerate(data_iter):
    #     # out = model.forward(batch.src, batch.trg, 
    #     #                     batch.src_mask, batch.trg_mask)
        # print('batch.src_mask.size():',batch.src_mask.size())
        # print('batch.trg_mask.size():',batch.trg_mask.size())
        # print('*'*40)
        out = model.forward(data.long(), targets.long(), 
                            batch.src_mask, batch.trg_mask)
        
        #src_mask = model.generate_square_subsequent_mask(data.size(0))
        
        
        
        
        
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        
        
        
        total_loss += loss
        total_tokens += batch.ntokens
        
        tokens += batch.ntokens
        
        if iteration % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f Total Secs: %f" %
                    (iteration, loss / batch.ntokens, tokens / elapsed,elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens



def run_epoch_evaluate(training_batches, model, loss_compute):
    print('^'*60)
    print('start_evaluation...')
    print('^'*60)
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    
    for iteration in range(start_iteration,n_iteration_evaluate):
        input_variable,lengths,target_variable=training_batches[n_iteration+iteration]
        #data, targets = get_batch(train_data, i)
        data=input_variable
        targets=target_variable
        batch=Batch(data,targets,0)
        out = model.forward(data.long(), targets.long(), 
                            batch.src_mask, batch.trg_mask)
    
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        
        
        
        total_loss += loss
        total_tokens += batch.ntokens
        
        tokens += batch.ntokens
    
    elapsed = time.time() - start
    print("evaluation Loss: %f Tokens per Sec: %f Total Secs: %f" %
                    ( loss / batch.ntokens, tokens / elapsed,elapsed))
    
    return total_loss / total_tokens
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)



class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))




class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
    
    
    



# def data_gen(V, batch, nbatches):
#     "Generate random data for a src-tgt copy task."
#     for i in range(nbatches):
#         data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
#         data[:, 0] = 1
#         src = Variable(data, requires_grad=False)
#         tgt = Variable(data, requires_grad=False)
#         yield Batch(src, tgt, 0)
        
#SimpleLossCompute(model.generator, criterion, model_opt)       
class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm
    



    
# Train the simple copy task.
V =voc.num_words
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, 2)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=5, betas=(0.9, 0.98), eps=1e-9))

n_iteration=1000
n_iteration_evaluate=100
batch_size=20

model.cuda()
criterion.cuda()
#model_opt.conda()

for epoch in range(10):
    model.train()
   
    #for _ in range(n_iteration)]
        
        
    print("Initializing...")
    print('-'*50)
    start_iteration=1
    total_loss = 0.
    start_time = time.time()
        
    training_batches=[batch2TrainData(voc,[random.choice(pairs) for _ in range(batch_size)])
                              for _ in range(n_iteration+n_iteration_evaluate)]
        #src_mask = model.generate_square_subsequent_mask(bptt).to(device)
        
        # for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        
            
        
        # for iteration in range(start_iteration,n_iteration+1):
        #     training_batch=training_batches[iteration-1]
            
          
    run_epoch_train(training_batches, model, 
                  SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
   
    run_epoch_evaluate(training_batches, model, 
                      SimpleLossCompute(model.generator, criterion, None))
    
    
    
    
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

model.eval()
src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) )
src_mask = Variable(torch.ones(1, 1, 10) )
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))



#!pip install torchtext spacy
#!python -m spacy download en
#!python -m spacy download de



# For data loading.
from torchtext import data, datasets

if True:
    import spacy
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, 
                     eos_token = EOS_WORD, pad_token=BLANK_WORD)

    MAX_LEN = 100
    train, val, test = datasets.IWSLT.splits(
        exts=('.de', '.en'), fields=(SRC, TGT), 
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
            len(vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
    
    
    
class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)






