import tensorflow as tf
from tensorflow.keras import Input, backend as K
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import InceptionV3,VGG16
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Bidirectional, Dot, Concatenate, Lambda, Attention, Conv2D, Embedding, BatchNormalization, MaxPool2D, GlobalMaxPool2D, Dropout, TimeDistributed, Dense, LSTM, GRU, Flatten, RepeatVector
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from IPython.display import Video, HTML
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import deque
import copy
from PIL import Image
from scipy import spatial
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import deque
import copy
import pickle
import unicodedata
import re
import numpy as np
import os
import io
import time
import pandas as pd
import cv2
from tqdm import tqdm,trange
import shutil
import csv
from nltk.translate.bleu_score import corpus_bleu
import json
import math
import random
tf.keras.backend.set_floatx('float64')


class M_Encoder(tf.keras.Model):
    def __init__(self,option):
        name = '_'.join(['encoder',option.encoder_type,str(option.encoder_units)])
        super(M_Encoder, self).__init__(name=name)
        self.option=option
        assert option.encoder_units>0 and type(option.encoder_units)==int,"Encoder type must be positive integer"
        
        self.timeDistDense = TimeDistributed(Dense(option.encoder_units, activation = 'relu',name='dense'), name='time_distributed_1')
        
        self.encoder = []
        for i in range(option.layer_count):
            if option.encoder_type=='GRU':
                self.encoder.append(Bidirectional(GRU(option.encoder_units, return_state=False, return_sequences = True, dropout = .5, name='en_gru'+str(i).zfill(2)),
                                        name='encoder'+str(i).zfill(2)))
            elif option.encoder_type=='LSTM':
                self.encoder.append(Bidirectional(LSTM(option.encoder_units, return_state=False, return_sequences = True, dropout = .5, name='en_lstm'+str(i).zfill(2)),
                                         name='encoder'+str(i).zfill(2)))
            else:
                assert False,"invalid encoder type"
            
        if option.seq_join_out>0:
            self.timeDistDense2 = TimeDistributed(Dense(option.seq_join_out, activation = 'relu',name='dense'), name='time_distributed_2')            
            
        self.build((None, option.max_len_target, 4096))
    
    def join_seq(self, x):
        x = tf.reshape(x, (-1,self.option.seq_join_out*self.option.max_len_target))
        return x
    
    def call(self,inputs):
        x_time = self.timeDistDense(inputs)
        x_en = x_time
        x_en_list = []
        for i in range(self.option.layer_count):
            x_en = self.encoder[i](x_en)
            x_en_list.append(x_en)
        if self.option.seq_join_out>0:
            x_time = self.timeDistDense2(x_time)
            x_time = self.join_seq(x_time)
            x_time = tf.expand_dims(x_time,axis=-2)
        return x_en, x_time, x_en_list

class M_JoinSeq(tf.keras.Model):
    def __init__(self,option):
        name = '_'.join(['joinseq',str(option.decoder_units)])
        super(M_JoinSeq, self).__init__(name=name)
        self.join_concat = Concatenate(axis=2, name='join_concat')
        self.join_dense = Dense(2*option.encoder_units+option.embed_out, activation='relu', name='join_dense')

    def call(self,inputs):
        x = self.join_concat(inputs)
        x = self.join_dense(x)
        return x
    
class M_Embedding(tf.keras.Model):
    def __init__(self,option):
        name = '_'.join(['embedding',str(option.num_words),str(option.embed_out)])
        super(M_Embedding, self).__init__(name=name)
        self.word2idx = option.get_tokenizer().word_index
        embeddings_index = self.embeddings_index_creator(option.embed_path)
        self.embed_in = len(self.word2idx) + 1
        self.embedding_matrix = self.embedding_matrix_creator(embeddings_index, word_index=self.word2idx,embedding_out=option.embed_out)
        self.embedding = Embedding(self.embed_in, option.embed_out, name='embedding', trainable = False)
        self.build((None,))
        self.set_weights([self.embedding_matrix])
        print('Embedding Layer Created')
        
    def embeddings_index_creator(self, embed_path):
        embeddings_index = {}
        with open(embed_path, encoding='utf-8') as f:
            for line in tqdm(f,file=None):
                values = line.split()
                word = values[0]
                try:
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
                except:
                    pass
            f.close()
        return embeddings_index
    
    def embedding_matrix_creator(self, embeddings_index, word_index, embedding_out):
        embedding_matrix = np.zeros((len(word_index) + 1, embedding_out))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix
    
    def call(self,inputs):
        x = self.embedding(inputs)
        return x

class M_Decoder(tf.keras.Model):
    def __init__(self,option):
        name = '_'.join(['decoder',option.decoder_type,str(option.decoder_units)])
        super(M_Decoder, self).__init__(name=name)
        
        assert option.decoder_units>0 and type(option.decoder_units)==int,"Encoder type must be positive integer"
        
        if option.decoder_type=='GRU':
            self.decoder = GRU(option.decoder_units, return_state = True, return_sequences = True, dropout = 0.5, name='decoder')
        elif option.decoder_type=='LSTM':
            self.decoder = LSTM(option.decoder_units, return_state = True, return_sequences = True, dropout = 0.5, name='decoder')
        else:
            assert False,"invalid encoder type"
        
        self.decoder_dense = Dense(option.num_words, activation = 'softmax', name='decoder_dense')
        
        self.build((None, 1, 2*option.encoder_units+option.embed_out))
    
    def call(self,inputs,initial_state=None):
        x = self.decoder(inputs,initial_state=initial_state)
        if type(x)==list:
            out = self.decoder_dense(x[0])
            return out, x[1:]
        else:
            out = self.decoder_dense(x)
            return x
    
class M_Attention(tf.keras.Model):
    def __init__(self,option):
        name = '_'.join(['attention'])
        super(M_Attention,self).__init__(name=name)
        self.attn_dense0 = Dense(2*option.encoder_units,activation='tanh',name='attn_dense0')
        self.attn_repeat_layer = RepeatVector(option.max_len_target,name='repeat_vector')
        self.attn_concat_layer = Concatenate(axis=-1,name='attn_concat')
        self.attn_dense1 = Dense(10, activation='tanh', name='attn_dense1')
        self.attn_dense2 = Dense(1, activation=self.softmax_over_time, name='attn_dense2')
        self.attn_dot = Dot(axes=1,name='attn_dot')
    
        self.build([(None,option.max_len_target,2*option.encoder_units),(1,option.decoder_units,)])

    def softmax_over_time(self,x):
        assert (K.ndim(x)>2),"x dims too small"
        e = K.exp(x - K.max(x,axis = 1, keepdims = True))
        s = K.sum(e, axis = 1, keepdims = True)
        return e/s

    def one_step_attention(self, h,st_1):
        st_1 = self.attn_repeat_layer(st_1)
        x = self.attn_concat_layer([h,st_1])
        x = self.attn_dense1(x)
        x = self.attn_dense2(x)
        context = self.attn_dot([x,h])
        return context
    
    def call(self, inputs):
        en_output, s = inputs
        s = self.attn_dense0(s)
        context = self.one_step_attention(en_output,s)
        return context
    
class M_Model(tf.keras.Model):
    def __init__(self,option):
        name = self.get_model_name(option)
        super(M_Model, self).__init__(name=name)
        assert option.layer_count > 0 and type(option.layer_count)==int,'Layer_count must be positive'
        assert option.attention==True,'Code for no attention is not yet available'

        self.option = option
    
        self.word2idx = option.get_tokenizer().word_index
        self.idx2word = { v: k for k, v in self.word2idx.items()}

        self.eos = self.word2idx['eeee']

        self.encoder = M_Encoder(option)

        self.embedding = option.get_embedding()

        self.decoder = M_Decoder(option)

        self.attention = [M_Attention(option)] if option.attention else None
        
        self.join_seq = M_JoinSeq(option) if option.seq_join_out else None 

        self.stacker = Lambda(self.stack_and_transpose,name='stacker')
        
        self.context_last_word_concat_layer = Concatenate(axis=2,name='concat_last_word')

        self.argmax = Lambda(self.arg_max_func, name='argmax')
        self.flatten = Flatten(name='flatten')
        
        self.hist = None
        
    def build_model(self):
        build_input_shape = []
        build_input_shape.append((None, self.option.max_len_target,4096))
        build_input_shape.append((None, 1))
        if self.attention:
            build_input_shape.append((None, self.option.decoder_units))
            build_input_shape.append((None, self.option.decoder_units))
        self.build(input_shape=build_input_shape)
    
    def arg_max_func(self, x):
        x = tf.math.argmax(x,axis=-1)
        x = tf.expand_dims(x,-1)
        # x = tf.expand_dims(x,-1)
        return x

    def stack_and_transpose(self,x):
        x = K.stack(x)
        x = K.permute_dimensions(x, pattern=(1,0,3,2))
        x = tf.squeeze(x,axis=-1)
        return x

    def get_model_name(self,option):
        model_name='_'.join(['en',option.encoder_type,'de',option.decoder_type,'layers',str(option.layer_count)])
        if option.attention:
            model_name=model_name+'_withAttention'
        return model_name
    
    def decode_sequence(self, dataset, start=0, length=1, log=False, return_parents=False, save=True):
        out_paragraph = []
        BLEU_1 = 0
        BLEU_2 = 0
        BLEU_3 = 0
        BLEU_4 = 0
        original = []
        parents = []
        for i in range(0, length):
            input_seq,y_in,y_out,parent = dataset[start+i]
            parent = parent.decode()
            parents.append(parent)
            input_data = []
            input_data.append(tf.expand_dims(input_seq,axis = 0))
            target_seq = np.zeros((1,1))
            target_seq[0,0] = self.word2idx['ssss']
            input_data.append(target_seq)
            if self.attention:
                s = np.zeros((1,self.option.decoder_units))
                input_data.append(s)
                c = np.zeros((1,self.option.decoder_units))
                input_data.append(c)

            outputs = self.predict(input_data)
            output_seq = []
            for out in outputs[0]:
                idx = self.argmax(out).numpy()[0]
                if self.eos == idx or self.word2idx['pppp']==idx:
                    break
                word = ' '
                if idx>0:
                    word = self.idx2word[idx]
                    output_seq.append(word)
            sentence = ' '.join(output_seq)
            out_paragraph.append(sentence)
            references = self.option.get_tokenizer().clean_cap(self.option.get_tokenizer().caption_dictionary[parent])
            BLEU_1+=corpus_bleu([references], [output_seq], weights=(1  ,  0,  0,  0))*100.00
            BLEU_2+=corpus_bleu([references], [output_seq], weights=(1/2,1/2,  0,  0))*100.00
            BLEU_3+=corpus_bleu([references], [output_seq], weights=(1/3,1/3,1/3,  0))*100.00
            BLEU_4+=corpus_bleu([references], [output_seq], weights=(1/4,1/4,1/4,1/4))*100.00
            if log:
                print([references], [output_seq])
            sentence = ' '.join(references[0])
            original.append(sentence)
        scores = {'BLEU_1':BLEU_1/max(length,1),'BLEU_2':BLEU_2/max(length,1),'BLEU_3':BLEU_3/max(length,1),'BLEU_4':BLEU_4/max(length,1)}
        if save:
            decoded_dict = {}
            for i in range(len(out_paragraph)):
                decoded_dict.update({i:{'pred':out_paragraph[i],'real':original[i],'parent':parents[i]}})
            try:
                json.dump(decoded_dict,open(os.path.join(self.option.model_path, str(self.hist.epoch[-1]+1).zfill(2)+'_sample.json'),'w'))
            except:
                json.dump(decoded_dict,open(os.path.join(self.option.model_path, '00_sample.json'),'w'))
        if return_parents:
            return out_paragraph, scores, original, parents
        return out_paragraph, scores, original
    
    def custom_fit(self, dataset, val_data=None, epochs=1, reset=False):
        save_path=self.option.checkpoints_path
        if epochs:
            total_batches = sum(1 for _ in dataset.padded_batch(self.option.batch_size).as_numpy_iterator())
            print('Total Batches:',total_batches)
        if os.path.exists(self.option.history_path) and not reset:
            self.hist = tf.keras.callbacks.History()
            self.load_weights(save_path)
            self.hist.set_model(self)
            self.hist.on_train_begin()
            self.hist.history = json.load(open(self.option.history_path, 'r'))
            self.hist.epoch = self.hist.history['epoch']
            curr_epoch = self.hist.epoch[-1]+1
            print("Starting training from ",curr_epoch," epochs")
        else:
            self.hist = tf.keras.callbacks.History()
            self.hist.set_model(self)
            self.hist.on_train_begin()
            curr_epoch = 0
            self.save_weights(save_path)
            print("Checkpoint Initialized")
        for epoch in range(curr_epoch,epochs+curr_epoch):
            score=0
            loss = 0
            accuracy = 0
            batch_count = 0
            data = dataset.shuffle(900, reshuffle_each_iteration=True)
            data = data.padded_batch(self.option.batch_size).as_numpy_iterator()
            for i in trange(total_batches):
                element = next(data)
                X, y_in, y_out,parent = element
                BATCH_SIZE = X.shape[0]
                input_list = []
                input_list.append(X)
                input_list.append(y_in)
                if self.option.attention:
                    z1 = np.zeros((BATCH_SIZE,self.option.decoder_units))
                    input_list.append(z1)
                    z2 = np.zeros((BATCH_SIZE,self.option.decoder_units))
                    input_list.append(z2)
                loss_t, accuracy_t = self.train_on_batch(
                    input_list, 
                    tf.keras.utils.to_categorical(y_out, num_classes = self.option.num_words)
                )
                if not np.any(np.isnan(loss_t)): 
                    loss+=loss_t
                    accuracy+=accuracy_t
                    batch_count+=1
            if not val_data==None:
                _,score,_ = self.decode_sequence(val_data,start=0,length=5+epoch*10)
            if batch_count:
                loss=loss/batch_count
                accuracy=accuracy/batch_count
                print("Batches in epoch ", batch_count)
                self.hist.on_epoch_end(epoch=epoch,logs={'loss':loss,'accuracy':accuracy,'bleu':score,'epoch':epoch})
                print('Epoch:',epoch,' loss:',loss,' acc:',accuracy,' bleu:',score)
                self.save_weights(save_path)
                json.dump(self.hist.history, open(self.option.history_path, 'w'))
        return self.hist
        
    def call(self, inputs, training=False):
        encoder_inputs, decoder_inputs, init_s, init_c = inputs
        state = [init_s]
        if self.option.decoder_type=='LSTM': state.append(init_c)

        en_output, en_time, en_output_list = self.encoder(encoder_inputs)
        outputs = []
        
        if not training: target_seq = decoder_inputs

        for t in range(self.option.max_len_target):
            if training:
                selector = Lambda(lambda x,t: x[:, t:t+1],arguments={'t':t},name='lambda'+str(t))
                target_seq = selector(decoder_inputs,t)   

            xt = self.embedding(target_seq)
            decoder_input = self.attention[0]([en_output,state[0]])
            decoder_input = self.context_last_word_concat_layer([decoder_input,xt])
            if self.join_seq: decoder_input = self.join_seq([decoder_input,en_time])
            out, state = self.decoder(decoder_input, initial_state=state)
            outputs.append(out)

            if not training:
                flat = self.flatten(out)
                idx = self.argmax(flat)
                target_seq = idx
        outputs = self.stacker(outputs)
        return outputs
    
class M_Novel_Model(M_Model):
    def __init__(self,option):
        super(M_Novel_Model, self).__init__(option=option)
        self.attention = []
        for layer in range(option.layer_count):
            self.attention.append(M_Attention(option))
        if option.layer_count>1: 
            self.enc_stack = Concatenate(-1,name='enc_stacker')
            self.stacked_dense = Dense(2*self.option.encoder_units, activation='relu', name='stacked_dense')

    def call(self, inputs, training=False):
        encoder_inputs, decoder_inputs, init_s, init_c = inputs
        state = [init_s]
        if self.option.decoder_type=='LSTM': state.append(init_c)

        en_output, en_time, en_output_list = self.encoder(encoder_inputs)
        outputs = []
        
        if not training: target_seq = decoder_inputs

        for t in range(self.option.max_len_target):
            if training:
                selector = Lambda(lambda x,t: x[:, t:t+1],arguments={'t':t},name='lambda'+str(t))
                target_seq = selector(decoder_inputs,t)   

            xt = self.embedding(target_seq)
            encoder_out = []
            for layer in range(len(en_output_list)):
                encoder_out.append(self.attention[layer]([en_output_list[layer],state[0]]))
            if len(encoder_out)>1: 
                decoder_input = self.enc_stack(encoder_out)
                decoder_input = self.stacked_dense(decoder_input)
                
            else: decoder_input = encoder_out[0]
                
            decoder_input = self.context_last_word_concat_layer([decoder_input,xt])
            
            if self.join_seq: decoder_input = self.join_seq([decoder_input,en_time])
            out, state = self.decoder(decoder_input, initial_state=state)
            outputs.append(out)

            if not training:
                flat = self.flatten(out)
                idx = self.argmax(flat)
                target_seq = idx
        outputs = self.stacker(outputs)
        return outputs