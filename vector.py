from sub_dataset import *
import torch
import torch.nn.functional as F
import numpy as np
import math
import random
import re

def char_mask(line,max_len):
    #line = ''.join(['' if ch == '_' else ch for ch in line])
    rand = [random.randint(0,len(line)-1) for i in range(0,int(math.ceil(len(line)/10)))] #Only 10% is masked
    new_line = ['M' if i in rand else ch for i,ch in enumerate(line.split(' '))]
    new_line = ['S'] + new_line + ['E']
    new_line = ['U' if char=='' else char for char in new_line]
    len_old_line = len(new_line)
    if len(line)<max_len:
        sub_add = ['U']*(max_len-len(line))
        new_line = new_line + sub_add
    elif len(line)>max_len:
        new_line = new_line[:max_len] 
    return new_line, len_old_line


def create_line(line):
    max_len = 100
    masked_line, line_len = char_mask(line,max_len)
    return masked_line, line_len


def context_target(vector,window):
    rand = window*2
    d_true, d_false, data, line_data = [],[],[],[]
    for i in range(window,len(vector)-window):
        # True targets
        current = vector[i]
        start = vector[i-window:i]
        end = vector[i+window-1:i+window+1]
        context = start+end
        true_ = [([current],[val],[1]) for val in context]
        
        # False targets
        false_ = [([current],[vector[random.randint(i+window+1,len(vector)-window)]],[0]) for i in range(rand)]        
       
        for val in true_:
            d_true.append(val)
        for val in false_:
            d_false.append(val)

    #print(len(d_true), len(d_false))
    line_data = d_true + d_false 
    """
        data = false_ + true_
        for val in data:
            data.append(val)
    line_data = data
    """
    return line_data


def convert_to_vec(lines, char_dict):
    window = 2 # Change if needed
    dict_ = {'S':2,'E':3,'M':4,'U':5} # start, end, mask and unknown values
    for idx, val in enumerate(char_dict.keys()):
        dict_[val]=idx+4
    dict_['_'] = len(dict_)

    sum_ = 0
    data = []
    for line in lines: #For testing pass fewer lines 
        new_line, line_len = create_line(line) # With start, end and masked words
        new_line = [dict_[val] if val in dict_.keys() else dict_['U'] for val in new_line]
        line_data = context_target(new_line, window) # Building false negatives where anything beyond window has target of 0 else 1
        for val in line_data:
            data.append(val)
    return data, dict_, window


def build_char_data(lines,char_dict,test=False):
    data, char_dict, window = convert_to_vec(lines, char_dict)
    if(test):
        data = data[:50]
    return data, char_dict, window
