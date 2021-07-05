# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 23:11:56 2019

@author: HP
"""

# This tutorial with through the process of transitioning a sequence-to-sequence model
# to TorchScript using the TorchScript API.The model that we will convert is the chatbot
#model from the Chatbot tutorial.You can either treat this tutorial as a “part2" to
# the Chatbot tutorial and deploy your own pretrained model,or you can start with this document
# and use a pretrained model that we host.In the latter case, you can reference the original Chatbot
# tutorial for details regarding data preprocessing, model theory and definition, and model
# training

#What is TorchScript

# During the research and development phase of a deep learning-based project, it is advantageous
# to interact with an eager, imperative interface like pyTorch's. This gives users the ability to write
# familiar,idiomatic Python ,allowing for the use of python data structures, control
# flow operations, print statements, and debugging utilities. Although the eager interface is a
# benefical tool for research and experimentation applications, when it comes time to deploy the model
# in a production environment, having a graph-based model representation is very beneficial.A deferred graph
# representation allows for optimizatios such as out-of-order execution, and the ability to target
# highly optimized for incrementally converting eager-mode code into TorchScript, a statically analyzable
# and optimizable subset of python that Torch uses to represent deep learning programs independently from the 
# Python runtime.

# The API for converting eager-mode PyTorch programs into TorchScripts is found in the toch.jit module.
# This Module has two core modalities for converting an eager-model to a TorchScript graph representation:
# tracing and scripting. The torch.jit.trace function takes a module or function and a set of example inputs.
# It then runs the example input through the function or module while tracing the computational steps
# that are encountered, and outputs a graph-based function that performs the traced operations. Tracing is 
# great for straightforward modules and functions that do not involve data-dependent control flow, such as 
# standard convolutional neural networks.However, if a function with data-dependent if statements and loops
# is traced,only the operations called along the execution route taken by the example input will be recorded
# In other words, the control flow itself is not captured. To convert modules and functions containing data-
# dependent control flow, a scripting mechanism is provided. The torch.jit.script function/decorator takes a
# module or function and does not requires example inputs. Scripting then explicitly converts the module or
# function code to TorchScript, including all control flows. One caveat with using scripting is that it
# only supports a subset of python, so you might need to rewrite the code to make it compatible with TorchScript
# syntax

# For all details relating to the suppotted features, see the TorchScript language reference. To provide the 
# maximum flexibility, you can also mix tracing and scripting modes together to represent your whole
# program, and these techniques can be applied incrementally.

# Acknowledgements
# This tutorial was inspired by the following sources
# 1. Yuan-Kuei Wu's pytorch-chatbot implementation:
# 2. Sean Robertson's practical-pytorch seq2seq-translation example:
# 3. FloyHub's Cornel Movie Corpus preprocessing code:

# Prepare Environment

# First, we will import the required modules and set some constants. If you are planning on using your
# own model. be sure that the MAX_LENTH constant is set correctly. As a reminder, this constant defines 
# the maximum allowed sentence length during training and the maximum length output that the model
# is capable of producing

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re
import unicodedata
import numpy as np

# device=torch.device("cpu")
USE_CUDA=torch.cuda.is_available()
device=torch.device("cuda" if USE_CUDA else "cpu")
MAX_LENGTH=10#Maximum sentence length

#Default word tokens
PAD_token=0 #Used for padding short sentence
SOS_token=1 #Start-of-sentence token
EOS_token=2 # End-of-sentence token


#Model Overview

# As mentioned, the model we are using is a sequence-to-sequence(seq2seq) model .This type
# of model is used when our input is a variable-length sequence, and our output is also a
# variable length sequence that is not necessarily a one-to-one mapping of the input.
# A seq2seq models is comprised of two recent neural networks(RNNs) that work cooperatively:
# an encoder and a decoder

# Encoder 
# The encoder RNN generates the response sentence in a token-by-token fashion.It uses the 
# encoder's context vectors, and internal hidden states to generate the next word in the 
# sequence.It continues generating words until it outputs an EOS_token, representing the end
# of the sentence. We use an attention mechanism in out decoder to help it to "pay attention"
# to certain part of the input when generating the output. For out model, we implement
# Luong et al's "Global attention" module, and use it as a submodule in our decode model.

# Data Handling
# Although our models conceptually deal with sequences of tokens, in reality, they deal with
# numbers like all machine learning models do. In this case, every word in the model's vocabulary,
# which was established before training, is mapped to an integer index.We use a Voc object to 
# contain the mapings from word to index, as well as the total number of words in the vocabulary.
# We will load the object later before we run the model.

# Also, in order for us to be able to run evaluations, we mush provide a tool for processing our
# string inputs. The normalizeString function converts all characters in a string to lower case
# and removes all non-letter characters.The indexesFromSentence function takes a sentence of words
# and returns the corresponding sequence of word indexes

class Voc:
    def __init__(self,name):
        self.name=name;
        self.trimmed=False
        self.word2index={}
        self.word2count={}
        self.index2word={PAD_token:"PAD",SOS_token:"SOS",EOS_token:"EOS"}
        self.num_words=3 #count SOS,EOS,PAD
        
    def addSentence(self,sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self,word):
        if word not in self.word2index:
            self.word2index[word]=self.num_words
            self.word2count[word]=1
            self.index2word[self.num_words]=word
            self.num_words+=1
        else:
            self.word2count[word]+=1;

#Remove words below a certain count threshold
    def trim(self,min_count):
        if self.trimmed:
            return 
        self.trimmed=True
        keep_words=[]
        for k,v in self.word2count.items():
            if v>=min_count:
                keep_words.append(k)
        
        print('keep_words{}/{}={:.4f}'.format(
                len(keep_words),len(self.word2index),len(keep_words)/len(self.word2index)
                ))
        #Reinitialize dictionaries
        self.word2index={}
        self.word2count={}
        self.index2word={PAD_token:"PAD",SOS_token:"SOS",EOS_token:"EOS"}
        self.num_words=3 #Count defaulf tokens
        for word in keep_words:
            self.addWord(word)
            
#Lowercase and remove non-letter characters
def normalizeString(s):
    s=s.lower()
    s=re.sub(r"([.!?])",r" \1",s)
    s=re.sub(r"[^a-zA-Z.!?]+",r" ",s)
    return s
    
#Takes string sentence, returns sentence of word index
def indexesFromSentence(voc,sentence):
    return[voc.word2index[word] for word in sentence.split(' ')]+[EOS_token]
    
        
#Define Encoder

#We implement our encoder's RNN with the torch.nn.GRU module which we feed a batch of sentences
#(vectors of word embeddings) and it internally iterates through the sentences one token at a
#time calculateing the hidden states.We initialize this modeule to bidirectional, meaning that
#iterates in model's forward function expects a padded input batch.To batch variable-length sentences,
#we allow a maximum of MAX_LENGTH tokens in tokens.To use padded batches with a PyTorch RNN module,
#we must wrap the forward pass call with torch.nn.utils.rnn.pack_padded_sequence and 
#torch.nn.utils.rnn.pad_packed_sequence data transformations. Note that the forward function also
#takes an input_length list,which contains the length of each sentence in the batch.This input is 
#used by the torch.nn.utils.rnn.pack_padded_sequence function when padding.
        
#TorchScript Notes:

#since the encoder's forward function does not contain any data-dependent control flow, we will use tracing
#to convert it to script mode.When tracing a module,we can leave the module definition as-is.We will
#initialize all models towards the end of this document before we run evaluations

class EncoderRNN(nn.Module):
    def __init__(self,hidden_size,embedding,n_layers=1,dropout=0):
        super(EncoderRNN,self).__init__()
        self.n_layers=n_layers
        self.hidden_size=hidden_size
        self.embedding=embedding
        
        #Initialize GRU; the input_size and hidden_size params are bothy set to 'hidden_size'
        # because our input size is a word embedding with number of features==hidden_size
        self.gru=nn.GRU(hidden_size,hidden_size,n_layers,
                        dropout=(0 if n_layers==1 else dropout),bidirectional=True)
        
    def forward(self,input_seq,input_lengths,hidden=None):
        #type:(Tensor,Tensor,Optional[Tensor])->Tuple[Tensor,Tensor]
        #Convert word indexes to embeddings
        embedded=self.embedding(input_seq)
        #Pack padded batch of sequences for RNN module
        packed=torch.nn.utils.rnn.pack_padded_sequence(embedded,input_lengths)
        #Forward pass through GPU
        outputs,hidden=self.gru(packed,hidden)
        #Unpack padding
        outputs,_=torch.nn.utils.pad_packed_sequence(outputs)
        #Sum bidirectional GRU outputs
        outputs=outputs[:,:,:self.hidden_size]+outputs[:,:,self.hidden_size:]
        #Return output and final hidden state
        return outputs,hidden
    
#Define Decoder's Attention Module
    
#Next, we'll define our attention module(Attn).Note that this module will be used as a 
#submodule in our decoder model. Luong et al. consider various"score functions", which take
#take the current decoder RNN output and the entire encoder outpur. and return attention
#"energies" .This attention energies tensor is the same size as the encoder output, and the 
#two are ultimately multiplied, resulting in weighted tensor whose largest values represent 
#the most important parts of the query sentence at a particular time-step of decoding 
    
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
        #Calculate the attention weights (energies) based on the given method
        if self.method=='general':
            attn_energies=self.general_score(hidden,encoder_outputs)
        
        elif self.method=='concat':
            attn_energies=self.concat_score(hidden,encoder_outputs)
        elif self.method=='dot':
            atten_energies=self.dot_score(hidden,encoder_outputs)
            
        #Transpose max_length and batch_size dimensions
        attn_energies=attn_energies.t()
        
        #Return the softmax normalized probability scores(with added dimension)
        return F.softmax(attn_energies,dim=1).unsqueeze(1)
    
    
    
# Define Decoder
# Similarly to the EncoderRNN, we use the torch.nn.GRU module for our decoder's RNN.
# This time , however, we use a unidirecional GRU.It is important to note that unlike
# the encoder, we will feed the decoder RNN one word at a time. We start by getting the 
# embedding of the current word and applying a dropout. Next, we forward the embedding and 
# the last hidden state to the GRU and obtain a current GRU output and hidden state.
# We then use our Attn module as a layer to obtain the attention weights, which we multiply
# by the encoder's output to obtain our attended encoder output. We use this attended encoder
# output to pay attention to. From bere, we use a linear layer and softmax normalization to 
# select the next word in the output sequence
        
# TorchScript Notes:
# ~~~~~~~~~~~~~~~~~~~~~
#
# Similarly to the ''EncoderRNN'',this module does not contain any
# data-dependent control flow. Therefore, we can once again use
# **tracing** to convert this model to TorchScript after it 
# is initialized and its parameters are loaded.

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self,attn_model,embedding,hidden_size,output_size,n_layers=1,dropout=0.1):
        super(LuongSttnDecoderRNN,self).__init__()
        
        #Keep for reference
        self.attn_model=attn_model
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.n_layers=n_layers
        self.dropout=dropout
        
        #Define layers
        self.embedding=embedding
        self.embedding_dropout=nn.Dropout(dropout)
        self.gru=nn.GRU(hidden_size,hidden_size,n_layers,dropout=(0 if n_layers==1 else dropout))
        self.concat=nn.Linear(hidden_size*2,hidden_size)
        self.out=nn.Linear(hidden_size,output_size)
        
        self.attn=Attn(attn_model,hidden_size)
        
    def forward(self,input_step,last_hidden,encoder_outputs):
        #Note: we run this one step (word) at a time
        #Get embedding of current input word
        embedded=self.embedding(input_step)
        embedded=self.embedding_dropout(embedded)
        #Forward through unidirectional GRU
        run_output,hidden=self.attn(run_output,encoder_outputs)
        #Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context=attn_weights.bmm(encoder_outputs.transpose(0,1))
        #Concatenate weighted context vector and GRU output using Luong eq. 5
        run_output=run_output.squeeze(0)
        context=context.squeeze(1)
        concat_input=torch.cat((rnn_output,context),1)
        concat_output=torch.tanh(self.concat(concat_input))
        #Predict next word using Luong eq 6
        output=self.out(concat_output)
        output=F.softmax(output,dim=1)
        #Return output and final hidden state
        return output, hidden
    


#Define Evaluation

#Greedy Search Decoder
        
# As in the chatbot, we use a GreedySearchDecoder module to facilitate the actual decoding
# process. This module has the trained encoder and decoder models as attributes, and drives
# the process of encoding an input sentence(a vector of word indexex), and iteratively
# decoding an output response sequence one word (word index) at a time

# Encoding the input sequence is straightforward: simply forward the entire sequence tensor and
# its corresponding lengths vector to the encoder. It is important to more that this module only
# deals with one input sequence at a time, NOT batches of sequences. Therefore. when run forward passes
# through our decoder model, which outputs softmax scores corresponding to the probability of each
# word being the correct next word in the decoded sequence. We initialize the decoder_input to a 
# tensor containing an SOS_token. After each pass through the decoder, we greedily append the word with 
# the heighest softmax probability to the decoded_words list. We also use this word as the decoder_input
# for the next iteration. The decoding process terminates either if the decoded_words list has reached
# a length of MAX_LENGTH of if the predicted word is the EOS_token

# TorchScript Notes:
# The forward method of this modeule involves iterating over the range of [0,max_length] when decoding
# an output sequence one word at a time. Because of this ,we should use scriping to convert this module
# to TorchScript. Unlike with our encoder and decoder models, which we can trace, we must make some
# necessary changes to the GreedySearchDecoder module in order to initialize an object without error.\
# in other words, we must ensure that out module adheres to the rules of the TorchScript mechanism, and
# does not utilize any language features outside of the subset of Python that TorchScript includes.

# To get an idea of some manipulations that may be required, we will go over the diffs between the 
# GreedySearchDecoder implementation from the chatbot turtorial and the implementation that we use
# in the cell below.Note that the line highlighted in red are lines removed from the original implementation
# and the lines hightlighted in green are new.
        
# Changes:
# 1. Added decoder_n_layers to the constructor arguments
#       This change stems from the fact that the encoder and decoder models that we pass to this module will be
#       child of TraceModule(not Module). Therefor, we cannot access the decoder's number of layers with
#       decoder.n_layers. Instead, we plan for this and pass this value in during module construction.
# 2. Store away new attributes as constants
#       In the original implementation,we were free to use variables from the surrounding(global)
#       scope in our GreedySearchDecoder's forward method.However, now that we are using scripting, we
#       we do not have this freedom, as the assumption with srcipting is that we cannot necessarily
#       hold on to python objects, especially when exporting. An easy solution to this is store these
#       values from the global scope as attributes to the module in the constructor,and add them to a
#       special list called __constants__ so that they be used as literal values when constructing the
#       graph in the forward method. An example of this usage is on NEW line 19, where instead of using
#       the device and SOS_token global values, we use our constant attributes self._device and 
#       self._SOS_token
#3. Enforce types of forward method arguments
#       By default, all parameters to a TorchScript function are assumed to be Tensor.If we need to pass
#       an angument of a different type, we can use function type annotations as introduced in PEP 3107.
#       In addition, it is possible to declare arguments of different types using MyPy-style type annotations
#       (see doc).
#4. Change initialization of decoder_input
#       In the original implementation, we initialized outr decoder_input tensor with torch.LongTensor(
#       [SOS_token]). When scrpting, we are not allowed to initialize tensors in a literal fashion like this
#       Instead, we can initialize our tensor with an explicit torch function such as torch.ones. In this 
#       case, we can easily replicate the scalar decoder_input tensor by multiplying 1 by our SOS_token vlaue
#       stored in the constant self._SOS_token.
        
class GreedySearchDecoder(nn.Module):
    def __init__(self,encoder,decoder,decoder_n_layers):
        super(GreedySearchDecoder,self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self._device=device
        self._SOS_token=SOS_token
        self._decoder_n_layers=decoder_n_layers
        
    __constants__=['_device','_SOS_token','_decoder_n_layers']
    
    def forward(self,input_seq:torch.Tensor,input_length:torch.Tensor,max_length:int):
        #Forward input through encoder model
        encoder_outputs,encoder_hidden=self.encoder(input_seq,input_length)
        #Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden=encoder_hidden[:self._decoder_n_layers]
        #Initialize tensors to append decoded words to 
        all_tokens=torch.zeros([0],device=self._device,dtype=torch.long)
        all_scores=torch.zeros([0],device=self._device)
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
        #Prepare collections of word tokens and scores
        return all_tokens,all_scores 









































































