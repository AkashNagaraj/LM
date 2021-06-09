## File that chooses a small subdataset of 10000 lines based on frequency ##

def preprocess(lines):
    new_lines = []
    for line in lines:
        if line.rstrip():
            line = line.lower()
            new_lines.append(line)
    return new_lines


def get_char(char_dict,line):
    if line:
        for ch in line[-3].split(' '):
            if ch in char_dict:
                char_dict[ch]+=1 
            else:
                char_dict[ch] = 0
        return char_dict 
      

def check_frequency(x):
    char = {}
    for line in x:
        char = get_char(char, line)
    freq = [char[ch] for ch in char.keys() if ch.isalpha() and char[ch]>0] #Identify sub group with most occurence
    avg_freq = sum([int(float(val)) for val in char.values()])/len(freq)
    return freq, avg_freq, char


def choose_set(lines,split):
    max_, avg_ = 0,0
    dict_ = {}
    for i in range(1,len(split)):
        current_lines = lines[split[i-1]:split[i]]
        freq, sum_, char_dict = check_frequency(current_lines)
        if(len(freq)>max_):
            index = [split[i-1],split[i]]
            max_ = len(freq)
            avg = sum_
            dict_ = char_dict
        elif(len(freq)==max_):
            if sum_>avg:
                index = [split[i-1],split[i]]
                max_ = len(freq)
                avg = sum_
                dict_ = char_dict
    return(index, avg, dict_) # Return the index of lines that have most alphabet occurences with the overall avg


def choose_sentences():
    data_dir = "data/ptbdataset/ptb.char.train.txt" # File is only 1.2mb
    lines = open(data_dir,'r').readlines()
    lines = preprocess(lines)
    open(data_dir,'r').close()
    split = [i for i in range(0,len(lines),10000)]
    split = split + [len(lines)] # Last lines that are less than 10000
    index, avg, dict_ = choose_set(lines,split) # Choose the set of 10000 lines to be used
    return(lines[index[0]:index[1]], avg, dict_)


