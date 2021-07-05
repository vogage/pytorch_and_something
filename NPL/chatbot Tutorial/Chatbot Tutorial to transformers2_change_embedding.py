# -*- coding: utf-8 -*-
#"""
#Created on Mon Sep  9 09:14:16 2019

#@author: Qiandehou

#https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
#"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script,trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math

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
    return inp,lengths,output,mask,max_target_len

 #Example for validation
small_batch_size=64
batches=batch2TrainData(voc,[random.choice(pairs) for _ in range(small_batch_size)])
input_variable,lengths,target_variable,mask,max_target_len=batches


# print("batches:",batches)
# print("input_variable:",input_variable)
# print("lengths:",lengths)
# print("target_variable:",target_variable)
# print("mask:",mask)
# print("max_target_len:",max_target_len)


#Define Models
#seq2seq Model
#The brains of our chatbot is a sequence-to-sequence(seq2seq)model. The goal of a seq2seq
#model is to take a variable-length sequence as an inout,and return a variable-length sequence
#as an output using a fixed-sized model.

#Sutskever et al.discover that by using two separate recurrent neural nets together,
#we can accomplish this task.One RNN acts as an encoder,which encoders a variable length
#inout sequence to a fixed-length context vector.In theory, this context vector(the
#final hidden layer of the RNN)will contain semantic information about the query sentence
#that is input to the bot. The second RNN is a decoder, which takes an input word and 
# the context vector,and returns a guess for the next word in the sequence and a hidden
#state to use in the next iteration

#Image source:https://jeddy92.github.io/JEddy92.github.io/ts_seq2seq_intro/

#Encoder

#The encoder RNN iterates through the input sentence one token(e.g. word) at a time,at each
#time step outputing an "output" vector and a "hidden state" vector.The hidden state vector
#is then passed to the next time step, while the output vector is recorder.The encoder transforms
#the context it saw at each point in the sequence into a set of points in a high-dimensional
#space, which the decoder will use to generate a meaningful output for the given task

#At the heart of our encoder is a muti-layered Gated Recurrent Unit,invented by Cho et al.
# in 2014.We will use a bidirectional vaiant of the GRU, meaning that there are essentially
# two independent RNNs:one that is fed the input sequence in normal sequential order, and 
# one that is fed the input sequence in reverse order.The outputs of each network are summed
# at each time step.Using a bidirectional GRU will give us the advantage of encoding both
# past and future context

# #Image source:https://colah.github.io/posts/2015-09-NN-Types-FP/
#
# Note that an embedding layer is used to encoder our word indices in an arbitrarily sized feature space.
# For our methods , this layer will map each word to a feature space of size hidden_size.When trained,these
# values should encode semantic similarity between similar meaning words

# Finally,if passing a padded batch of sequences to an RNN module, we must pack and 
# unpack around the RNN pass using nn.utils.rnn.pack_padded_sequence and nn.utils.rnn.pad_packed_sequence
# respectively

#Computation Graph:
# 1. Convert word indexes to embeddings
# 2. Pack padded batch of sequences for RNN module
# 3. Forward pass through GRU
# 4. Unpack padding
# 5. Sum bidirectional GRU outputs
# 6. Return output and final hidden state

# Inputs:
# 1. input_seq: batch of input sentences;shape=(max_length,batch_size)
# 2. input_lengths:list of sentence lengths corresponding to each sentence
# in the batch; shape=(batch_size)
# 3. hidden: hidden state; shape=(n_layers x num_directions,batch_size,hidden_size)

# Outputs:
# 1. outputs: output features from the last hidden layer of the GRU(sum of bidirectional 
# outputs);shape=(max_length,batch_size,hidden_size)
# 2. hidden:updated hidden state from GPU; shape=(n_layers x num_directions,batch_size,hidden_size)

class TransformerModel(nn.Module):
# emsize = 200 # embedding dimension
# nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
# nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
# nhead = 2 # the number of heads in the multiheadattention models
# dropout = 0.2 # the dropout value
# model = TransformerModel(ninp, emsize, nhead, nhid, nlayers, dropout).to(device)
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ntoken,ninp,dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.softmax_layer=nn.Softmax(dim=2)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
# src: (S, N, E)
# src_mask: (S, S)
# src_key_padding_mask: (N, S)
# where S is the sequence length, N the batch size and 
# E the embedding dimension (number of features).
    def forward(self, src, src_mask):
        
       
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        output=self.softmax_layer(output)
        return output


class PositionalEncoding(nn.Module):
# self.pos_encoder = PositionalEncoding(ntoken,ninp,dropout)
    def __init__(self,ntoken, d_model, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(ntoken, d_model)
        position = torch.arange(0, ntoken, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
    


def batchify(data, bsz):
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data

#batch_size = 20
batch_size=small_batch_size
eval_batch_size = 10
#train_data = batchify(input_variable, batch_size)


n_iteration=1000
# training_batches=[batch2TrainData(voc,[random.choice(pairs) for _ in range(batch_size)])
# for _ in range(n_iteration)]                  
                  
                  
                  
# val_data = batchify(val_data, eval_batch_size)
# test_data = batchify(test_data, eval_batch_size)

bptt=35

def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

    
ntokens = voc.num_words # the size of vocabulary
#ntokens= MAX_LENGTH
emsize = 500 # embedding dimension
nhid = 500 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 4 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout)
model.to(device)
#voc.to(device)
#pairs.to(device)

criterion = nn.CrossEntropyLoss()
#lr = 5.0 # learning rate
lr=0.5#I think is sparse
#梯度爆炸，loss 越来越多
#最后分析原因是没有进行归一化处理
#使用softmax处理之后就收敛了
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)




import time
def train():
    model.train() # Turn on the train mode
    
    
    print("Initializing...")
    start_iteration=1
    total_loss = 0.
    start_time = time.time()
    
    
    training_batches=[batch2TrainData(voc,[random.choice(pairs) for _ in range(batch_size)])
    for _ in range(n_iteration)]
    
    
    
    #src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    
    # for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
    for iteration in range(start_iteration,n_iteration+1):
        training_batch=training_batches[iteration-1]
        #Extract fields from batch
        
        
        
        input_variable,lengths,target_variable,mask,max_target_len=training_batch
        #data, targets = get_batch(train_data, i)
        data=input_variable
        targets=target_variable

        #处理成长度相同 MAX_LENGTH 
        #yy=torch.cat((targets,torch.zeros(10-targets.size(0),20)),0)
        #torch.zeros()生成的默认张量数据类型，该处为float32,
        #但embedding需要的是long型，因此会报错
        #通过 AGU： dtype=torch.long进行设置
        data=torch.cat((data,torch.zeros(MAX_LENGTH-data.size(0),batch_size
                                         ,dtype=torch.long)),0)
        targets=torch.cat((targets,torch.zeros(MAX_LENGTH-targets.size(0),batch_size
                                               ,dtype=torch.long)),0)
        # print(data.dtype)
        # print(targets.dtype)
                
  # for input, target in dataset:
  #   def closure():
  #       optimizer.zero_grad()
  #       output = model(input)
  #       loss = loss_fn(output, target)
  #       loss.backward()
  #       return loss
  #   optimizer.step(closure)
        #data.to(device)
        #targets.to(device)
        
        optimizer.zero_grad()

        src_mask = model.generate_square_subsequent_mask(data.size(0))
        # data=data.long()
        # src_mask.to(device)
        # data.to(device)
        # targets.to(device)
        # print(data.device)
        # print(targets.device)
        # print(model.cuda())
        output = model(data, src_mask)
        #output.to(device)

        loss = criterion(output.reshape(-1,ntokens), targets.reshape(1,-1).squeeze(0))
        #loss.to(device)
        #print(loss)
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        

        total_loss += loss.item()
        log_interval = 200
        if iteration % log_interval == 0 and iteration > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d} percent_batches | '
                  'lr {:02.6f} | ms {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, iteration*100/n_iteration, scheduler.get_lr()[0],
                    elapsed * 1000 ,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


# def evaluate2(encoder,decoder,searcher,voc,sentence,max_length=MAX_LENGTH):
#     #Formate input sentence as a batch
#     #words->indexes
#     indexes_batch=[indexesFromSentence(voc,sentence)]
#     #Create lengths tensor
#     lengths=torch.tensor([len(indexes)for indexes in indexes_batch])
#     #Transpose dimension of batch to match model's exceptations
#     input_batch=torch.LongTensor(indexes_batch).transpose(0,1)
#     #Use appropriate device
#     input_batch=input_batch.to(device)
#     lengths=lengths.to(device)
#     #Decode sentence with searcher
#     tokens,scores=searcher(input_batch,lengths,max_length)
#     #indexes->words
#     decoded_words=[voc.index2word[token.item()]for token in tokens]
#     return decoded_words

# def evaluateInput(encoder,decoder,searcher,voc):
#     input_sentence=''
#     while(1):
#         try:
#             #Get input sentence
#             input_sentence=input('>')
#             #Check if it is quit case
#             if input_sentence=='q'or input_sentence=='quit':break
#             #Normalize sentence
#             input_sentence=normalizeString(input_sentence)
#             #Evaluate sentence
#             output_words=evaluate(encoder,decoder,searcher,voc,input_sentence)
#             #Format and print response sentence
#             output_words[:]=[x for x in output_words if not (x=='EOS' or x=='PAD')]
#             print('Bot:',' '.join(output_words))
            
#         except KeyError:
#             print("Error:Encouted unknown word.")




def evaluate1(eval_model,input_variable,target_variable,mask):
    eval_model.eval() # Turn on the evaluation mode
    # model.eval() is a kind of switch for some specific 
    # layers/parts of the model that behave differently 
    # during training and inference (evaluating) time. 
    # For example, Dropouts Layers, BatchNorm Layers etc. 
    # You need to turn off them during model evaluation, 
    # and .eval() will do it for you. In addition, 
    # the common practice for evaluating/validation 
    # is using torch.no_grad() in pair with model.eval()
    # to turn off gradients computation:
    
    total_loss = 0.
    #src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    # Disabling gradient calculation is useful for inference, 
    # when you are sure that you will not call Tensor.backward(). 
    # It will reduce memory consumption for computations that would 
    # otherwise have requires_grad=True.
    
    with torch.no_grad():#与eval_model.eval()配合使用
    #torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度。
    
        for i in range(0, input_variable.size(0) - 1, batch_size):
            #data, targets = get_batch(data_source, i)
            # if data.size(0) != bptt:
            #     src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
            
            
            
            data=input_variable
            targets=target_variable
             #处理成长度相同 MAX_LENGTH 
             #yy=torch.cat((targets,torch.zeros(10-targets.size(0),20)),0)
             #torch.zeros()生成的默认张量数据类型，该处为float32,
             #但embedding需要的是long型，因此会报错
             #通过 AGU： dtype=torch.long进行设置
            data=torch.cat((data,torch.zeros(MAX_LENGTH-data.size(0),batch_size
                                             ,dtype=torch.long)),0)
            targets=torch.cat((targets,torch.zeros(MAX_LENGTH-targets.size(0),batch_size
                                                   ,dtype=torch.long)),0)
            # print(data.dtype)
            # print(targets.dtype)
            #optimizer.zero_grad()
    
            src_mask = eval_model.generate_square_subsequent_mask(data.size(0))
            
            
            
            
            #data=input_variable
            output = eval_model(data.long(), src_mask)
            #output_flat = output.view(-1, ntokens)
            total_loss = criterion(output.reshape(-1,ntokens), targets.reshape(1,-1).squeeze(0))
    return total_loss 


best_val_loss = float("inf")
evaluation_iteration=100
epochs = 10 # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    
 
    val_loss=0
    evaluate_batches=[batch2TrainData(voc,[random.choice(pairs) for _ in range(batch_size)])
    for _ in range(evaluation_iteration)]
    for i in range(evaluation_iteration):
        evaluate_batch=evaluate_batches[i-1]

        
        
        
        input_variable,lengths,target_variable,mask,max_target_len=evaluate_batch
        val_loss += evaluate1(model,input_variable,target_variable,mask)
    val_loss=val_loss/evaluation_iteration
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()











































