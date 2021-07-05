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
    
MAX_LENGTH=10 # Maximum Sentence length to consider

# Turn a Unicode string to plain ACSII, thanks to consider
# https:\\stackoverflow.com/a/518232/2009427

def unicodeToAscii(s):
    return ' '.join(
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
# print("\npairs:")
# for pair in pairs[:10]:
#     print(pair)
    
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
    mask=torch.ByteTensor(mask)
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
small_batch_size=5
batches=batch2TrainData(voc,[random.choice(pairs) for _ in range(small_batch_size)])
input_variable,lengths,target_variable,mask,max_target_len=batches


print("batches:",batches)
print("input_variable:",input_variable)
print("lengths:",lengths)
print("target_variable:",target_variable)
print("mask:",mask)
print("max_target_len:",max_target_len)


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

class EncoderRNN(nn.Module):
    def __init__(self,hidden_size,embedding,n_layers=1,dropout=0):
        super(EncoderRNN,self).__init__()
        self.n_layers=n_layers
        self.hidden_size=hidden_size
        self.embedding=embedding
        
        #Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        # because our input size is a word embedding with number of features == hidden_size
        self.gru=nn.GRU(hidden_size,hidden_size,n_layers,
                        dropout=(0 if n_layers == 1 else dropout),bidirectional=True)
        
    def forward(self,input_seq,input_lengths,hidden=None):
        #Convert word indexes to embeddings
        embedded=self.embedding(input_seq)
        #Pack padded batch of sequence for RNN module
        # print("\nembedded.size()")
        # print(embedded.size())
        packed=nn.utils.rnn.pack_padded_sequence(embedded,input_lengths)
        #Forward pass through GRU
        outputs,hidden=self.gru(packed,hidden)
        #hidden_layer=2 and bidirectional=True so the hidden_layer.size() is 4
        #Unpack padding
        outputs,_=nn.utils.rnn.pad_packed_sequence(outputs)
        #Sum bidirectional GRU outputs
        outputs=outputs[:,:,:self.hidden_size]+outputs[:,:,self.hidden_size:]
        #Return output and final hidden state
        return outputs,hidden
    
#Decoder
        
    
# The decoder RNN generates the response sentence in a token-ty-token fashion.It use the encoder's
# context vectors, and internal hidden states to generate the next world in the sentence. It
# continues generating words until it outputs an EOS_token, representing the end of the sentence. A
# common problem with a vanilla seq2seq decoder is that if we rely soley on the context vector
# to encode the entire input sentence's meaning, it is likely that we will have information loss.This
# is especially the case when dealing with long input sequences,rather than using the entire fixed context
# context at every step.

#At a high level,attention is calculated using the decoder's current hidden state and the encoder's 
#outputs. The output attention wights have the same shape as the input sequence, allowing us to 
#multipy them by the encoder outputs, giving us a weighted sum which indicates the parts of encoder 
#output to pay attention to. Sean Robertson's figure describes this very well

#Luong attention layer
class Attn(nn.Module):
    def __init__(self,method,hidden_size):
        super(Attn,self).__init__()
        self.method=method
        if self.method not in ['dot','general','concat']:
            raise ValueError(self.method,"is not an appropriate attention method.")
        self.hidden_size=hidden_size
        if self.method=='general':
            self.attn=nn.Linear(self.hidden_size,hidden_size)
        elif self.method=='concat':
            self.attn=nn.Linear(self.hidden_size*2,hidden_size)
            self.v=nn.Parameter(torch.FloatTensor(hidden_size))
    def dot_score(self,hidden,encoder_output):
        return torch.sum(hidden*encoder_output,dim=2)
    
    def general_score(self,hidden,encoder_output):
        energy=self.attn(encoder_output)
        return torch.sum(hidden*energy,dim=2)
    
    def concat_score(self,hidden,encoder_output):
        energy=self.attn(torch.cat((hidden.expand(encoder_output.size(0),-1,-1),
                                    encoder_output),2)).tanh()
        return torch.sum(self.v*energy,dim=2)
        
    def forward(self,hidden,encoder_outputs):
        #Calculate the attention weights(energies) based on the given method
        
        
        
        if self.method=='general':
            attn_energies=self.general_score(hidden,encoder_outputs)
        elif self.method=='concat':
            attn_energies=self.concat_score(hidden,encoder_outputs)
        elif self.method=='dot':
            attn_energies=self.dot_score(hidden,encoder_outputs)
            
        #Transpose mix_length and batch_size dimensions
        attn_energies=attn_energies.t()
        
        #Return the softmax normalized probability scores(with added dimension)
        return F.softmax(attn_energies,dim=1).unsqueeze(1)
            
#Outputs: output , hidden
        
#decoder=LuongAttnDecoderRNN(attn_model,embedding,hidden_size,voc.num_words,decoder_n_layers,dropout)    
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self,attn_model,embedding,hidden_size,output_size,n_layers=1,dropout=0.1):
        super(LuongAttnDecoderRNN,self).__init__()

        #Keep for reference
        self.attn_model=attn_model
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.n_layers=n_layers
        self.dropout=dropout
        
        #Define layers
        self.embedding=embedding
        self.embedding_droput=nn.Dropout(dropout)
        self.gru=nn.GRU(hidden_size,hidden_size,n_layers,dropout=(0 if n_layers==1 else dropout))
        self.concat=nn.Linear(hidden_size*2,hidden_size)
        self.out=nn.Linear(hidden_size,output_size)
        self.attn=Attn(attn_model,hidden_size)
    def forward(self,input_step,last_hidden,encoder_outputs):
        #Note: we run this one step (word) at a time
        #Get embedding of current input word
        embedded=self.embedding(input_step)
        embedded=self.embedding_droput(embedded)
        #Forward through unidirectional GRU
        rnn_output,hidden=self.gru(embedded,last_hidden)
        #Calculate attention weights from the current GRU output
        attn_weights=self.attn(rnn_output,encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weight sum" context vector
        context=attn_weights.bmm(encoder_outputs.transpose(0,1))
#        print("/ncontext.size()")
        # print(context.size())
        #Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output=rnn_output.squeeze(0)
        context=context.squeeze(1)
        concat_input=torch.cat((rnn_output,context),1)
#        print("/nconcat_input.size()")
        # print(concat_input.size())
        concat_output=torch.tanh(self.concat(concat_input))
        #Predict next word using Luong eq. 6
        output=self.out(concat_output)
        output=F.softmax(output,dim=1)
        #Return output and final hidden state
        return output,hidden
    
# mask_loss,nTotal=maskNLLLoss(decoder_output,target_variable[t],mask[t])
def maskNLLLoss(inp,target,mask):#inp: input
    nTotal=mask.sum()#全部求和
    crossEntropy=-torch.log(torch.gather(inp,1,target.view(-1,1)).squeeze(1))
    loss=crossEntropy.masked_select(mask).mean()
    loss=loss.to(device)
    return loss,nTotal.item()             
        
def train(input_variable,lengths,target_variable,mask,max_target_len,encoder,decoder,embedding,
          encoder_optimizer,decoder_optimizer,batch_size,clip,max_length=MAX_LENGTH):
    #Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    #Set device options
    input_variable=input_variable.to(device)
    lengths=lengths.to(device)
    target_variable=target_variable.to(device)
    mask=mask.to(device)
    
    #Initialize variable
    loss=0
    print_losses=[]
    n_totals=0
    
    #Forward pass through encoder
    encoder_outputs,encoder_hidden=encoder(input_variable,lengths)
    
    #Create initial decoder input(start with SOS tokens for each sentence)
    decoder_input=torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input=decoder_input.to(device)
    
    #Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden=encoder_hidden[:decoder.n_layers]
    
    #Determine if we are using teacher forcing this iteration
    use_teacher_forcing=True if random.random()<teacher_forcing_ratio else False
    
    #Forward batch of sequence through decoder one time step at a time
    if use_teacher_forcing:#有监督学习
        for t in range(max_target_len):
            decoder_output,decoder_hidden=decoder(
                    decoder_input,decoder_hidden,encoder_outputs)
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output,decoder_hidden=decoder(
                    decoder_input,decoder_hidden,encoder_outputs)
            #No teacher forcing :next input is decoder's own current output
            _,topi=decoder_output.topk(1)
            decoder_input=torch.LongTensor([[topi[i][0]for i in range(batch_size)]])
            decoder_input=decoder_input.to(device)
            #Calculate and accumulate loss
            mask_loss,nTotal=maskNLLLoss(decoder_output,target_variable[t],mask[t])
            loss+=mask_loss
            print_losses.append(mask_loss.item()*nTotal)
            n_totals+=nTotal
            
    #Perform backpropatation
    loss.backward()
    
    #Clip gradients:gradients are modified in place
    #梯度裁剪，用来防止梯度消失与梯度爆炸
    _=nn.utils.clip_grad_norm_(encoder.parameters(),clip)
    _=nn.utils.clip_grad_norm_(decoder.parameters(),clip)
    
    #Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return sum(print_losses)/n_totals
    
#Training iterations
#trainIters(model_name,voc,pairs,encoder,decoder,encoder_optimizer,decoder_optimizer,
#           embedding,encoder_n_layers,decoder_n_layers,save_dir,n_iteration,batch_size,
#           print_every,save_every,clip,corpus_name,loadFilename)        
def trainIters(model_name,voc,pairs,encoder,decoder,encoder_optimizer,decoder_optimizer,
           embedding, encoder_n_layers,decoder_n_layers,save_dir,n_iteration,batch_size,print_every,
           save_every,clip,corpus_name,loadFilename):

    #Load batchers for each iteration
    training_batches=[batch2TrainData(voc,[random.choice(pairs) for _ in range(batch_size)])
    for _ in range(n_iteration)]

    #Initialization
    print("Initializing...")
    start_iteration=1
    print_loss=0
    if loadFilename:
        start_iteration=checkpoint['iteration']+1
    
    #Training loop
    print("Training...")
    for iteration in range(start_iteration,n_iteration+1):
        training_batch=training_batches[iteration-1]
        #Extract fields from batch
        input_variable,lengths,target_variable,mask,max_target_len=training_batch
    
        #Run a training iteration with batch
        loss=train(input_variable,lengths,target_variable,mask,max_target_len,encoder,
                   decoder,embedding,encoder_optimizer,decoder_optimizer,batch_size,clip)
        print_loss+=loss
    
        #Print progress
        if iteration%print_every==0:
            print_loss_avg=print_loss/print_every
            print("Iteration:{};Percent complete:{:.1f}%;Average loss:{:.4f}".format(iteration,
                  iteration/n_iteration*100,print_loss_avg))
            print_loss=0
        
        #Save checkpoint
        if(iteration%save_every==0):
            directory=os.path.join(save_dir,model_name,corpus_name,'{}-{}_{}'.format(encoder_n_layers,
                               decoder_n_layers,hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

#Define Evaluation 
class GreedySearchDecoder(nn.Module):#searcher function
    def __init__(self,encoder,decoder):
        super(GreedySearchDecoder,self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        
    def forward(self,input_seq,input_length,max_length):
        #Forward input through encoder model
        encoder_outputs,encoder_hidden=self.encoder(input_seq,input_length)
        #Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden=encoder_hidden[:decoder.n_layers]
        #Initialize decoder input with SOS_token
        decoder_input=torch.ones(1,1,device=device,dtype=torch.long)*SOS_token
        #Initialize tensors to append decoded words to
        all_tokens=torch.zeros([0],device,dtype=torch.long)
        all_scores=torch.zeros([0],device=device)
        #Iteratively decode one word token at a time
        for _ in range(max_length):
            #Forward pass through decoder
            decoder_output,decoder_hidden=self.decoder(decoder_input,decoder_hidden,
                                                       encoder_outputs)
            #Obtain most likely word token and its softmax score
            decoder_scores,decoder_input=torch.max(decoder_output,dim=1)
            #Record token and score
            all_tokens=torch.cat((all_tokens,decoder_input),dim=0)
            all_scores=torch.cat((all_scores,decoder_scores),dim=0)
            #Prepare current token to be next decoder input (add a dimension)
            decoder_input=torch.unsqueeze(decoder_input,0)
        #Return collections of word tokens and scores
        return all_tokens,all_scores

#Evaluate my text
def evaluate(encoder,decoder,searcher,voc,sentence,max_length=MAX_LENGTH):
    #Formate input sentence as a batch
    #words->indexes
    indexes_batch=[indexesFromSentence(voc,sentence)]
    #Create lengths tensor
    lengths=torch.tensor([len(indexes)for indexes in indexes_batch])
    #Transpose dimension of batch to match model's exceptations
    input_batch=torch.LongTensor(indexes_batch).transpose(0,1)
    #Use appropriate device
    input_batch=input_batch.to(device)
    lengths=lengths.to(device)
    #Decode sentence with searcher
    tokens,scores=searcher(input_batch,lengths,max_length)
    #indexes->words
    decoded_words=[voc.index2word[token.item()]for token in tokens]
    return decoded_words

def evaluateInput(encoder,decoder,searcher,voc):
    input_sentence=''
    while(1):
        try:
            #Get input sentence
            input_sentence=input('>')
            #Check if it is quit case
            if input_sentence=='q'or input_sentence=='quit':break
            #Normalize sentence
            input_sentence=normalizeString(input_sentence)
            #Evaluate sentence
            output_words=evaluate(encoder,decoder,searcher,voc,input_sentence)
            #Format and print response sentence
            output_words[:]=[x for x in output_words if not (x=='EOS' or x=='PAD')]
            print('Bot:',' '.join(output_words))
            
        except KeyError:
            print("Error:Encouted unknown word.")


#Run model

#Configure models
model_name='cb_model'
attn_model='dot'
#attn_model='general'
#attn_model='concat'
hidden_size=500
encoder_n_layers=2
decoder_n_layers=2
dropout=0.1
batch_size=64
#Set checkpoint to load from; set to None if starting from scratch
loadFilename=None
checkpoint_iter=4000
#loadFirename=os.path.join(save_dir,model_name,corpus,name,
#                           '{}-{}_{}'.format(encoder_n_layers,decoder_n_layers,hidden_size),
#                           '{}_checkpoint.tar'.format(checkpoint_iter))

#Load model if a loadFilename is provided
if loadFilename:
    #If loading on same machine the model was trained on
    checkpoint=torch.load(loadFilename)
    #If loading a model trained on GPU to   CPU
    # checkpoint=torch.load(loadFilename,map_location=torch.device('cpu'))
    encoder_sd=checkpoint['en']
    decoder_sd=checkpoint['de']
    encoder_optimizer_sd=checkpoint['en_opt']
    decoder_optimizer_sd=checkpoint['de_opt']
    embedding_sd=checkpoint['embedding']
    voc.__dict__=checkpoint['voc_dict']
    
print('Building encoder and decoder...')
#   Initialize word embeddings
embedding=nn.Embedding(voc.num_words,hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
#Initialize encoder & decoder models
encoder=EncoderRNN(hidden_size,embedding,encoder_n_layers,dropout)
decoder=LuongAttnDecoderRNN(attn_model,embedding,hidden_size,voc.num_words,decoder_n_layers,dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
#Use appropriate device
encoder=encoder.to(device)
decoder=decoder.to(device)
print('Models built and ready to go!')
    
#Run Training

#Configure training/optimization
clip=50.0
teacher_forcing_ratio=1.0
learning_rate=0.0001
decoder_learning_ratio=5.0
n_iteration=4000
print_every=1
save_every=500

#Ensure dropout layers are in train mode
encoder.train()
decoder.train()

#Initialize optimizers
print('Building optimizers...')
encoder_optimizer=optim.Adam(encoder.parameters(),lr=learning_rate)
decoder_optimizer=optim.Adam(decoder.parameters(),lr=learning_rate*decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

#If you have cuda, configure cuda to call
for state in encoder_optimizer.state.values():
    for k,v in state.items():
        if isinstance(v,torch.Tensor):
            state[k]=v.cuda()

# Run training iterations
print("Starting Training!")
trainIters(model_name,voc,pairs,encoder,decoder,encoder_optimizer,decoder_optimizer,
           embedding,encoder_n_layers,decoder_n_layers,save_dir,n_iteration,batch_size,
           print_every,save_every,clip,corpus_name,loadFilename)


#Run evaluation
#Set dropout layers to eval mode
encoder.eval()
decoder.eval()

#Initialize search module
searcher=GreedySearchDecoder(encoder,decoder)

#Begin chatting(uncomment and run the following line to begin)
evaluateInput(encoer,decoder,searcher,voc)























