import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import unicodedata
import re
import os
import random
from models_and_utils.models import M_Embedding

class TokenizerWrap(Tokenizer):
    def __init__(self, options):
        Tokenizer.__init__(self, num_words=options.num_words)
        self.mark_start = 'ssss '
        self.mark_end = ' eeee'
        self.pad = ' pppp'
        self.temporal_length = options.temporal_length
        self.mode_dict = {0:'validation',1:'test',2:'train'}
        
#         self.caption_dictionary = self.get_caption_dict(options.caption_path)
        self.caption_dictionary = self.get_full_caption_dict(options.caption_path)
        self.texts = self.create_tokenizer(self.caption_dictionary)
        self.fit_on_texts(self.texts)
        
        self.index_to_word = dict(zip(self.word_index.values(), self.word_index.keys()))
        self.word_to_index = dict(zip(self.word_index.keys(), self.word_index.values()))
    
    def word_to_token(self, token):
        token = 0 if word not in word_to_index else self.word_to_index[word]
        return token
    
    def token_to_word(self, token):
        word = " " if token == 0 else self.index_to_word[token]
        return word

    def tokens_to_string(self, tokens):
        words = [self.index_to_word[token]
                 for token in tokens
                 if token != 0]
        text = " ".join(words)
        return text
    
    def captions_to_tokens(self, captions_list):
        tokens = self.texts_to_sequences(captions_list)
        tokens = pad_sequences(tokens, maxlen=self.temporal_length, padding='post', truncating='post')
        y_in = tokens[:, 0:-1]
        y_out = tokens[:, 1:]
        return y_in, y_out
        
    def unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')

    def preprocess_sentence(self, w, start_end = True):
        w = self.unicode_to_ascii(w.lower().strip())
        w = re.sub(r"([?.!,Ã‚Â¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        w = re.sub(r"[^a-zA-Z?.!,Ã‚Â¿]+", " ", w)
        w = w.strip()
        if start_end:
            w = self.mark_start + w + self.mark_end
            for i in range(12):
                w = w+self.pad
        return w

    def mark_captions(self, captions_list):
        captions_marked = [self.preprocess_sentence(caption)
                            for caption in captions_list]
        return captions_marked
    
    def get_full_caption_dict(self, path):
        df = pd.read_csv(path,encoding='utf-8')
        df = df.iloc[:,1::]
        df = df.values.tolist()
        caption_dictionary = {}
        for data in df:
            count = 0
            for caption in data[1:]:
                if str(caption)=='nan':
                    break
                else:
                    count+=1
            caption_dictionary.update({data[0]:self.mark_captions(data[1:count])})
        return caption_dictionary
    
    def get_caption_dict(self,path):
        captions = pd.read_csv(path)
        parents = captions['FileName']
        caption_list = captions[['0','1','2','3','4']].values
        caption_dict = dict()
        for i in range(len(parents)):
            caption_dict.update({parents[i]:self.mark_captions(caption_list[i])})
        return caption_dict
    
    def create_tokenizer(self,caption_dictionary):
        cap_list = []
        for parent,rows in caption_dictionary.items():
            for row in rows:
                cap_list.append(row)
        return cap_list
    def clean_cap(self, captions):
        clean_captions = []
        for caption in captions:
            caption = caption.split(' ')
            clean_caption = []
            for word in caption:
                if word not in ['eeee','pppp','ssss','.']:
                    clean_caption.append(word)
            clean_captions.append(clean_caption)
        return clean_captions

    def get_data_list(self, path):
        return list(pd.read_csv(path)['0'])

    def get_parent(self, path):
        return '_'.join(path.split('_')[1:4])

    def data_generator(self, mode=0, data_size=1000000):
        assert mode in [0,1,2],"Invalid mode"
        mode = self.mode_dict[mode]
        data_dir = 'data_pickle'

        data_path = mode+'.csv'
        data_list = self.get_data_list(os.path.join(data_dir,data_path))
        data_len = min(data_size,len(data_list))
    #     print("Working with ",data_len," data items out of ",len(data_list))
        caption_dict = self.caption_dictionary

        for data in data_list[:data_len]:
            parent = self.get_parent(data)
            y = random.choice(caption_dict[parent])
            y_in, y_out = self.captions_to_tokens([y])
            with open(os.path.join(data_dir,mode,data),'rb') as f:
                X = pickle.load(f)
            yield tf.convert_to_tensor(X,dtype=tf.float64), tf.convert_to_tensor(y_in[0],dtype=tf.int64), tf.convert_to_tensor(y_out[0],dtype=tf.int64), tf.convert_to_tensor(parent,dtype=tf.string)
