import os
import random
import numpy as np
import types
from io import FileIO
import collections as cl

import pickle

from base64 import encodebytes as bs64enc
from base64 import decodebytes as bs64dec
##from dataclasses import dataclass, field

import ospck


def load(path, default=None, makedirs=True):
    if makedirs: ospck.makedirs(path)
    if os.path.isfile(path):
        with open(path, "rb") as f:
            try: obj = pickle.load(f)
            except TypeError: obj = default
            except pickle.UnpicklingError: obj = default
        return obj
    return default
def save(path, obj):
    ospck.makedirs(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

    
def filecopy_gen(read, write, buffer=4096, fraction=True, report_buffer=None):
    size = os.path.getsize(read)
    size_running = 0
    ospck.makedirs(write)
    with open(write, "wb") as wf:
        with open(read, "rb") as rf:
            if report_buffer is None: report_buffer = buffer
            while len(b:=rf.read(buffer)):
                wf.write(b)
                size_running += len(b)
                if size_running>=report_buffer:
                    if fraction: yield size_running/size
                    else:
                        yield size_running
                        size_running = 0
            if not fraction and size_running:
                yield size_running
                size_running = 0
def filecopy(*args, **kwargs):
    for x in filecopy_gen(*args, **kwargs): pass

def foldercopy_gen(read, write, buffer=4096, fraction=True, report_buffer=None):
    read = ospck.get_folder(read)
    write = ospck.get_folder(write)
    ospck.makedirs(write)
    size = ospck.getsize_folder(read)
    size_running = 0
    if report_buffer is None: report_buffer = buffer
    #folders = 
    for f in ospck.list_folders(read):
        dirs, name, ext = ospck.explode(f)
        ff = ospck.implode([write], name)
        if not os.path.exists(ff): os.mkdir(ff)
        for i in foldercopy_gen(f, ff, buffer=buffer, fraction=False, report_buffer=report_buffer*len(folders)):
            size_running += i
            if size_running>=report_buffer:
                if fraction: yield size_running/size
                else:
                    yield size_running
                    size_running = 0
        else:
            if not fraction and size_running:
                yield size_running
                size_running = 0
    for f in ospck.list_files(read):
        dirs, name, ext = ospck.explode(f)
        ff = ospck.implode([write], name, ext)
        for i in filecopy_gen(f, ff, buffer=buffer, fraction=False, report_buffer=report_buffer*len(items)):
            size_running += i
            if size_running//report_buffer:
                if fraction: yield size_running/size
                else:
                    yield size_running
                    size_running = 0
        else:
            if not fraction and size_running:
                yield size_running
                size_running = 0
def foldercopy(*args, **kwargs):
    for x in foldercopy_gen(*args, **kwargs): pass



def copy_gen(read, *args, **kwargs):
    if os.path.isfile(read):
        for x in filecopy_gen(read, *args, **kwargs): yield x
    else:
        for x in foldercopy_gen(read, *args, **kwargs): yield x
def copy(*args, **kwargs):
    for x in copy_gen(*args, **kwargs): pass



def is_valid_txt(path): return path.endswith(".txt") and os.path.isfile(path)


def read_txt_lines(path, encoding="utf8", **kwargs):
    with open(path, "r", encoding=encoding, **kwargs) as f:
        while 1:
            line = f.readline()
            if not line: break # end of file
            yield line
def read_txt(path, buffer=None, encoding="utf8", **kwargs):
    with open(path, "r", encoding=encoding, **kwargs) as f:
        while 1:
            chars = f.read(buffer)
            if not chars: break # end of file
            yield chars






def get_attr_as_list(x, *args): return [getattr(x, str(k), None) for k in args]
def get_attr_as_dict(x, **kwargs): return {k:getattr(x, str(v), None) for k,v in kwargs.items()}

def reverse_dict(d): return {v:k for k,v in d.items()}

def is_dict(x): return type(x)==dict
def is_type(x): return type(x)==type
def is_tuple(x): return type(x)==tuple
def is_list(x): return type(x)==list
def is_set(x): return type(x)==set
def is_array(x): return type(x)==np.ndarray
def is_str(x, array=True): return type(x) in [str,np.str_] or (array and is_array(x) and str(x.dtype)[:2]=="<U")
def is_float(x, array=True): return type(x)==float or isinstance(x, np.floating) or (array and is_array(x) and is_float(getattr(np, str(x.dtype))(1)))
def is_int(x, array=True): return type(x)==int or isinstance(x, np.integer) or (array and is_array(x) and is_integer(getattr(np, str(x.dtype))(1)))
def is_iter(x): return type(x) in [tuple,set,list,np.ndarray] or hasattr(x, "__iter__")
def is_iterable(x): return is_iter(x)
def is_mutable(x): return type(x) in [list,np.ndarray,dict]
def is_function(x): return isinstance(x, types.FunctionType)
def is_callable(x): return is_function(x) or hasattr(x, "__call__")
def is_hashable(x): return not is_mutable(x)

def is_string(x, *args, **kwargs): return is_str(x, *args, **kwargs)
def is_integer(x, *args, **kwargs): return is_int(x, *args, **kwargs)









def filesave(path, x): # save bytes as a file
    directory = ospck.implode(ospck.explode(path)[0])
    os.makedirs(directory, exist_ok=True)
    with FileIO(path, "w") as f:
        f.write(x)
def fileload(path): # load any file as a bytes
    with FileIO(path, "r") as f:
        return f.read()

def pcksave2dictsave(path): dictsave(path, pckload(path))
def dictsave2pcksave(path): pcksave(path, dictload(path))








def indexfeed(values): # values -> indexes sorted by values
    values = list(zip(range(len(values)), values))
    values.sort(reverse=True, key=lambda x:x[1])
    return list(zip(*values))[0]
def valuesplit(total, r): # total value broken into 2 values using a ratio
    if type(r) in [int,float]: r = r.as_integer_ratio()
    xx = total/(r[0]+r[1])
    return round(xx*r[0]), round(xx*r[1])

# list manipulation
def listsplit(l, r): # divide a list in two with a ratio
    i = round(len(l)*(r/(r+1)))
    return l[:i], l[i:]
def listpick(l, n=1): # return n random items from list, no repeats
    picks = []
    while len(l)>0 and len(picks)<n: picks.append(l.pop(random.randint(0,len(l)-1)))
    return picks# [ for _ in range(n) if len(l)>0 else pass]

def listswap(l, i, ii): # swap two items
    if type(l) in [list,str] and i!=ii:
        i = i%len(l)
        ii = ii%len(l)
        if i < 0: i = i+(len(l)-1)
        if ii < 0: ii = ii+(len(l)-1)
        if type(l) in [list]: l[i], l[ii] = l[ii], l[i]
        elif type(l) in [str]:
            ii, i = max(i, ii), min(i, ii)
            l = l[:i]+l[ii]+l[i+1:ii]+l[i]+l[ii+1:]
    return l
def listmove(l, i, ii): # move an item to another index
    if type(l) in [list,str]:
        i = i%len(l)
        ii = ii%len(l)
        if type(l) in [list]:
            ii, i = max(i, ii), min(i, ii)
            p = l[i]
            for j in range(i, ii): l[j] = l[j+1]
            l[ii] = p
        elif type(l) in [str]:
            p = l[i]
            l = l[:i]+l[i+1:]
            l = l[:ii]+p+l[ii:]
    return l
def listconveyor(l, i): # shift items with an offset
    if type(l) in [list,str]:
        ii = -i%len(l)
        l = l[ii:]+l[:ii]
    return l
def listyeet(l, i): # reshuffle an item back in
    if type(l) in [list,str]:
        i = i%len(l)
        ii = random.randint(0, len(l)-1)
        l = listmove(l, i, ii)
    return l









###########

# STRING
def str2bytes(s, enc="utf8"): return bytes(s, enc) #filesave(path+".str", bytes(s, enc))
def bytes2str(b, enc="utf8"): return str(b, enc)
def strsave(path, s, enc="utf8"): filesave(path+".str", bytes(s, enc))
def strload(path, enc="utf8"):
    if os.path.exists(path+".str"): return str(fileload(path+".str"), enc)
    
def floatsave(path, f): filesave(path+".float", bytes(str(f), "utf8"))
def floatload(path):
    if os.path.exists(path+".float"): return float(str(fileload(path+".float"), "utf8"))

def nonesave(path): filesave(path+".none", b"")
def noneload(path):
    if os.path.exists(path+".none"): return
    
def typesave(path, t): filesave(path+".type", bytes(str(t.__name__), "utf8"))
def typeload(path):
    if os.path.exists(path+".type"): return eval(str(fileload(path+".type"), "utf8"))

# SINGLE INTEGER
def intsave(path, i): filesave(path+".int", bytes(str(i), "utf8"))
def intload(path):
    if os.path.exists(path+".int"): return int(str(fileload(path+".int"), "utf8"))
    

# ANY SHAPE ARRAY
def array2bytes(a):
    pre = np.array(a.shape).astype(np.uint64).tobytes()+b"0"
    if a.dtype==np.uint8: t = 1
    elif a.dtype==np.uint16: t = 2
    elif a.dtype==np.uint32: t = 3
    elif a.dtype==np.uint64: t = 4
    elif a.dtype==np.int8: t = 5
    elif a.dtype==np.int16: t = 6
    elif a.dtype==np.int32: t = 7
    elif a.dtype==np.int64: t = 8
    elif a.dtype==np.float16: t = 9
    elif a.dtype==np.float32: t = 10
    elif a.dtype==np.float64: t = 11
    elif a.dtype==np.bool_: t = 12
    else: t = 12+int(a.itemsize/4) # str_
    pre = pre+bytes([t])+b"0"
    return pre+a.tobytes()
def bytes2array(b):
    shape, t, b = b.split(b"0", 2)
    shape = np.frombuffer(shape, dtype=np.uint64)
    t = t[0]
    if t==1: t = np.uint8
    elif t==2: t = np.uint16
    elif t==3: t = np.uint32
    elif t==4: t = np.uint64
    elif t==5: t = np.int8
    elif t==6: t = np.int16
    elif t==7: t = np.int32
    elif t==8: t = np.int64
    elif t==9: t = np.float16
    elif t==10: t = np.float32
    elif t==11: t = np.float64
    elif t==12: t = np.bool_
    else: t = f"U{t-12}"
    return np.array(np.frombuffer(b, dtype=t)).reshape(shape)

##def arraysave(path, a):
##    if a.size and not a.dtype in [np.float16,np.float32,np.float64] and 4>a.ndim>1 and a.min()>=0 and a.max()<256: # if image like
##        if a.ndim==3:
##            if a.shape[-1]==1: mode="L"
##            elif a.shape[-1]==2: mode="LA"
##            elif a.shape[-1]==3: mode="RGB"
##            elif a.shape[-1]==4: mode="RGBA"
##            else:
##                a = a[:,:,:4]
##                mode = "RGBA"
##        elif a.ndim==2: mode = "L"
##        directory = ospck.implode(ospck.explode(path)[0])
##        os.makedirs(directory, exist_ok=True)
##        PILimg.fromarray(a.astype(np.uint8), mode).save(path+".png", optimize=True)
##        if os.path.exists(path+".array"): os.remove(path+".array")
##        os.rename(path+".png", path+".array")
##    else: filesave(path+".array", array2bytes(a))
##def arrayload(path):
##    if os.path.exists(path+".array"):
##        f = FileIO(path+".array", "rb")
##        b = f.read(4)
##        f.close()
##        if b==b"\x89PNG": return np.asarray(PILimg.open(path+".array"))
##        return bytes2array(fileload(path+".array"))
def arraysave(path, a): filesave(path+".array", array2bytes(a))
def arrayload(path):
    if os.path.exists(path+".array"): return bytes2array(fileload(path+".array"))

# DICT
def dictsave(path, d, add=False): # folder
    if not add: ospck.delete_folder(path)
    os.makedirs(path, exist_ok=True)
    for k,v in d.items():
        t = type(v)
        p = path+"\\"+openers[type(k)]+str(k)
        if t==str: strsave(p, v)
        elif t==type: typesave(p, v)
        elif t==float: floatsave(p, v)
        elif t==int: intsave(p, v)
        elif t==bool: intsave(p, 1 if v else 0)
        elif t==dict: dictsave(p, v)
        elif t in [list,tuple]: iterablesave(p, v)
        elif t==np.ndarray: arraysave(p, v)
        elif v==None: nonesave(p)
        else:
            try: v._save(p)
            except: nonesave(p)
def dictload(path):
    d = {}
    for x in ospck.list_folders(path):
        dirs, name, ext = ospck.explode(x)
        k = reverse_openers[name[0]](name[1:])
        d[k] = dictload(x)
    for x in ospck.list_files(path):
        dirs, name, ext = ospck.explode(x)
        p = path+"\\"+name
        k = reverse_openers[name[0]](name[1:])
        if ext=="str": d[k] = strload(p)
        elif ext=="type": d[k] = typeload(p)
        elif ext=="float": d[k] = floatload(p)
        elif ext=="int": d[k] = intload(p)
        elif ext=="array": d[k] = arrayload(p)
        elif ext=="iterable": d[k] = iterableload(p)
        elif ext=="none": d[k] = None
    return d
##class dictsavenload(): # deprecated
##    def _save(self, path): dictsave(path, self.__dict__)
##    def _load(self, path): return self.load(dictload(path))
##    def load(self, d):
##        if d:
##            for k,v in d.items(): setattr(self, k, v)
##            return True


def configsave(path, d): # bools/ints/floats/strings -> readable format
    # [float..] key = value
    # [string.] key = value
    # [integer] key = value
    # [boolean] key = value
    f = FileIO(path+".config", "w")
    for k,v in d.items():
        kt = type(k)
        vt = type(v)
        if kt==str:
            if vt==int:
                f.write(bytes("[integer] "+k+" = ", "utf8"))
                f.write(bytes(str(v)+"\n", "utf8"))
            elif vt==float:
                f.write(bytes("[float..] "+k+" = ", "utf8"))
                f.write(bytes(str(v)+"\n", "utf8"))
            elif vt==str:
                f.write(bytes("[string.] "+k+" = ", "utf8"))
                f.write(bytes(v+"\n", "utf8"))
            elif vt==bool:
                f.write(bytes("[boolean] "+k+" = ", "utf8"))
                f.write(bytes(str(v)+"\n", "utf8"))
            elif vt==type:
                f.write(bytes("[type...] "+k+" = ", "utf8"))
                f.write(bytes(str(v.__name__)+"\n", "utf8"))
    f.close()
def configload(path):
    if not os.path.exists(path+".config"): return {}
    d = {}
    f = FileIO(path+".config", "r")
    text = str(f.read(), "utf8")
    f.close()
    for line in text.split("\n"):
        if line:
            try:
                t, k, eq, v = line.split(" ", 3)
                if "int" in t[1:-1]: d[k] = int(v)
                elif "float" in t[1:-1]: d[k] = float(v)
                elif "str" in t[1:-1]: d[k] = v
                elif "bool" in t[1:-1]: d[k] = True if eval(v) else False
                elif "type" in t[1:-1]: d[k] = eval(v)
            except: pass
    return d

## LIST
def iterablesave(path, l): # list/tuple
    content = b"l" if type(l)==list else b"t"
    for x in l:
        t = type(x)
        if t==str: a = b"s"+bytes(x, "utf8")
        elif t==int: a = b"i"+bytes(str(x), "utf8")
        elif t==float: a = b"f"+bytes(str(x), "utf8")
        elif t==bool: a = b"b"+bytes(str(1 if x else 0), "utf8")
        elif t in [list,tuple]: a = iterablesave(None, x)
        elif t==np.ndarray: a = b"a"+array2bytes(x)
        else: a = b"n"
        content += bytes(str(len(a)), "utf8")+b"_"+a
    if path: filesave(path+".iterable", content)
    else: return content
def iterableload(path, content=None):
    if path: content = fileload(path+".iterable")
    elif not content: return None
    it, content = content[0:1], content[1:]
    l = []
    while b"_" in content:
        la, content = content.split(b"_", 1)
        la = int(str(la, "utf8"))
        t = str(content[0:1], "utf8")
        if t=="s": l.append(str(content[1:la], "utf8"))
        elif t=="i": l.append(int(str(content[1:la], "utf8")))
        elif t=="f": l.append(float(str(content[1:la], "utf8")))
        elif t=="b": l.append(True if eval(str(content[1:la], "utf8")) else False)
        elif t in "tl": l.append(iterableload(None, content[0:la]))
        elif t=="a": l.append(bytes2array(content[1:la]))
        elif t=="n": l.append(None)
        content = content[la:]
    if it==b"t": return tuple(l)
    return l
###########


    

def listsearch(x, l, start=0, end=None): # return indexes of x in l # NOT BYTE FRIENDLY
    if end==None: end = len(l)
    return [i+start for i,xx in enumerate(l[start:end]) if x==xx]
def listsearch_b(x, l, start=0, end=None, wordlen=1): # return indexes of x in l # BYTE FRIENDLY
    if end==None: end = len(l)
    result = []
    i = start
    while l[i:min(end, i+wordlen)]:
        xx = l[i:i+wordlen]
        if x==xx: result.append(i)
        i += wordlen
    return result


    




def dict2xml(path, d, encoding="utf8"):
    f = open(path+".xml", "w", encoding=encoding)
    f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
    def recur(x, depth=0):
        f.write("\n")
        for k,v in x.items():
            tag = str(k)
            f.write("\t"*depth+"<"+tag+">")
            if type(v)==dict: recur(v, depth+1)
            else:
                f.write(str(v))
            f.write("</"+tag+">\n")
    recur(d)
    f.close()

def xml2dict(path, encoding="utf8"):
    f = open(path+".xml", "r", encoding=encoding)
    dpath = []
    d = {}
    text = f.read()
    startstreak = 0
    endstreak = 0
    while 1:
        try: i = text.index(">")
        except ValueError: break
        t = text[:i+1]
        text = text[i+1:]
        if t[1]=="?": continue
        elif "</" in t:
            endstreak += 1
            v = t.split("</", 1)[0]
            if endstreak==1:
                dd = d
                for xx in dpath[:-1]: dd = dd[xx]
                dd[dpath[-1]] = str(v)
            dpath.pop()
            
            startstreak = 0
        else:
            startstreak += 1
            
            k = t.split("<", 1)[1][:-1]
            dpath.append(k)
            
            dd = d
            for xx in dpath[:-1]: dd = dd[xx]
            dd[dpath[-1]] = {}
            
            endstreak = 0
    f.close()
    return d


def index_nearness(element, list_or_string, i=0): # how close is element to the correct index
    offset = len(list_or_string)
    i %= offset
    if list_or_string[i]==element: return 0
    if element in list_or_string[i+1:]: offset = min(offset, list_or_string[i+1:].index(element)+1)
    if element in list_or_string[:i]: offset = min(offset, list_or_string[:i][::-1].index(element)+1)
    return offset
    
def strloss(x, y):
    loss = 0
    # character count mismatch +count
    char = dict.fromkeys(list(x)+list(y))
    for xx in char.keys():
        y_n = y.count(xx)
        x_n = x.count(xx)
        loss += abs(y_n-x_n)/max(y_n, x_n)
    loss /= len(char)
    # misplaced characters +offset
    t = len(y)*len(x)
    for i,xx in enumerate(x): loss += index_nearness(xx, y, i)/t
    # add multiplied loss for len mismatch
    return loss/2+loss*(abs(len(y)-len(x))/len(y))

def strguess(x, options, maxloss=0.5):
    if type(x) in [list,tuple]:
        indexes = multi_strguess(x, options, maxloss)
        return options[indexes[1]] if indexes else None
    guess_values = [strloss(x, xx) for xx in options]
    index = np.argmin(guess_values)
    if guess_values[index]<maxloss: return options[index]

def multi_strguess(x, options, maxloss=0.5): # pick best match from x if x is an iterable
    guesses = []
    for i,xx in enumerate(x):
        values = [strloss(xx, xxx) for xxx in options]
        bestguess = np.argmin(values)
        guesses.append((i, bestguess, values[bestguess]))
    best = min(guesses, key=lambda x:x[2])
    if best[2]<maxloss:
        # input index, result index
        return best[0], best[1]


def strsearch_classic(string, options, amount=1):
    return [options[i] for i in np.argsort([strloss(string, x) for x in options], kind="stable")[:amount]]
def strsearch(string, options, amount=1): # longer string -> faster search
    str_l = len(string)
    options2 = [x for x in options if str_l<=len(x)]
    if len(options2)==len(options):
        options2 = [x for x in options if string in x]
        if len(options2)>amount: options = options2
    if len(options2)>amount: options = options2
    char_match = False
    for i in range(str_l):
        options2 = [x for x in options if i<len(x) and string[i]==x[i]] # min(i+earliest_start, len(x)-1)
        if len(options2)>amount: options = options2
        char_match = char_match or bool(options2)
    for j in range(str_l):
        options2 = [x for x in options if string[j] in x]
        if len(options2)>amount: options = options2
        char_match = char_match or bool(options2)
    if not char_match: return [] # no character matches -> give up
    elif len(options)>amount:
        options2 = [x for x in options if str_l==len(x)]
        if len(options2)>amount: options = options2
    return [options[i] for i in np.argsort([strloss(string, x) for x in options], kind="stable")[:amount]]

def strmatch(string, options): # find best exact match
    str_l = len(string)
    for i in range(1, str_l+1):
        xx = string[:i]
        options2 = []
        for x in options:
            if (ii:=x.find(xx))>=0: options2.append((ii, x))
        if options2:
            options2.sort(key=lambda x:x[1].replace(xx, "."*len(xx)).lower())#x[0] if x[0]>0 else str_l)
            options = [x[1] for x in options2]
    return options[0]

##def strsearch2(string, options, amount=1, turbo=True):
##    if turbo:
##        str_l = len(string)
##        str_start = 0
##        opt_start = None
##        max_option_len = 0
##        for i in range(str_l):
##            for x in options:
##                if i==0: max_option_len = max(max_option_len, len(x))
##                if string[i] in x:
##                    opt_start = x.index(string[i]) if opt_start is None else min(opt_start, x.index(string[i]))
##                    str_start = i
##            if opt_start!=str_l: break
##            else: pass
##        if opt_start is None: return []
##        threshold = 0
##        options2 = options
##        while threshold<max_option_len:
##            for i in range(str_start, str_l):
##                options2 = [x for x in options if index_nearness(string[i], x, i-str_start+opt_start)<=threshold]
##                if len(options2)>amount: options = options2
##            threshold += 1
##    return [options[i] for i in np.argsort([strloss(string, x) for x in options], kind="stable")[:amount]]


##def matchloss(i, u, tolerance=1, incr=10): # compare int,float,str,array,list and tuples for a matchloss float
##    if 0 > tolerance > incr**3: return 0 # cutoff point
##    nexttolerance = tolerance*incr
##    if type(u) in [int,float] and type(i) in [int,float]:
##        if i == u: matchvalue = 0
##        elif i*u<0: matchvalue = 1 # opposite values
##        else: matchvalue = abs(u-i)/max(abs(u), abs(i))
##    elif type(u) in [np.ndarray] and type(i) in [np.ndarray]:
##        und, ind = abs(max(u.ndim, i.ndim)), abs(min(u.ndim, i.ndim))
##        ush, ish = np.abs(np.maximum(u.shape[:ind], i.shape[:ind])), np.abs(np.minimum(u.shape[:ind], i.shape[:ind]))
##        u, i = np.abs(np.maximum(u[ish-1], i[ish-1])), np.abs(np.minimum(u[ish-1], i[ish-1]))
##        matchvalue = np.mean((und-ind)/und) # dimension mismatch
##        matchvalue += np.mean((ush-ish)/(ush)) # shape mismatch
##        matchvalue += np.mean((u-i)/(u)) # matchloss(u.tolist(), i.tolist(), tolerance)
##    elif type(u) in [tuple] and type(i) in [tuple]: # len, pos
##        if u == i: matchvalue = 0
##        elif len(u) > 0 or len(i) > 0:
##            if len(i) != len(u):
##                ul = min(u, i, key=lambda x:len(x))
##                il = max(u, i, key=lambda x:len(x))
##            else: ul, il = u, i
##            matchvalue = matchloss(len(ul), len(il), nexttolerance)*0.8
##            matchvalue += (sum([matchloss(ul[x], il[x], nexttolerance) for x in range(len(ul))])/len(ul))*0.2
##        else: matchvalue = 1
##        
##    elif type(u) in [list] and type(i) in [list]: # len, pos, amounts
##        if u == i: matchvalue = 0
##        elif len(u) > 0 or len(i) > 0:
##            ulist = list(dict.fromkeys(u))
##            ilist = list(dict.fromkeys(i))
##            if len(ulist) != len(ilist):
##                ul = min((ulist, u), (ilist, i), key=lambda x:len(x[0]))
##                il = max((ulist, u), (ilist, i), key=lambda x:len(x[0]))
##            else: ul, il = (ulist, u), (ilist, i)
##            matchvalue = matchloss(len(ul[0]), len(il[0]), nexttolerance)*0.25+matchloss(len(ul[1]), len(il[1]), nexttolerance)*0.25
##            matchvalue += (sum([matchloss(ul[0][x], il[0][x], nexttolerance) for x in range(len(ul[0]))])/len(ul[0]))*0.25 # positional comparing
##            # (uniqueitem, amount) comparing
##            aml = list(zip(il[0], [il[1].count(x) for x in il[0]]))
##            ams = list(zip(ul[0], [ul[1].count(x) for x in ul[0]]))
##            matchvalue += (sum([sum([matchloss(x, xx, nexttolerance) for xx in ams])/len(ams) for x in aml])/len(aml))*0.25
##        else: matchvalue = 1
##    elif type(u) in [str] and type(i) in [str]: matchvalue = strloss(u,i)
##    elif type(u) in [dict] and type(i) in [dict]: # len, key
##        if u == i: matchvalue = 0
##        elif len(u) > 0 or len(i) > 0:
##            if len(i) != len(u):
##                ul = min(u, i, key=lambda x:len(x))
##                il = max(u, i, key=lambda x:len(x))
##            else: ul, il = u, i
##            matchvalue = matchloss(len(ul), len(il), nexttolerance)*0.2
##            matchvalue += (sum([matchloss(v, il[k], nexttolerance) if k in il else 1/nexttolerance for k,v in ul.items()])/len(ul))*0.8
##        else: matchvalue = 1
##    else: matchvalue = 1
##    return min(abs(matchvalue/tolerance), 1)
##
##def matchguess(x, options, maxloss=0.5):
##    guess_values = [matchloss(x, xx) for xx in options]
##    bestguess = np.argmin(guess_values)
##    if guess_values[bestguess]<maxloss: return options[bestguess]



















class config():
    def __init__(self):
        self.defaults = {}
        self.limits = {}
        self.active = {}
        self.memory = {}
        self.path = None

    def checkmissingvalues(self):
        succ = 0
        for k,v in self.defaults.items():
            if not k in self.active:
                self.active[k] = v
                succ += 1
        if len(self.defaults) != len(self.active):
            succ = 0
            for k,v in self.active.copy().items():
                if not k in self.defaults:
                    del self.active[k]
                    succ += 1

    def load(self, path=None):
        succ = 0
        if not path:
            if self.path: path = self.path
            else: path = "config"
        loaded = configload(path)
        for k,v in loaded.items():
            if not k in self.limits:
                continue
            succ += self.set(k, v, self.defaults, self.limits[k], self.active)
        if len(self.defaults)!=len(self.active): self.checkmissingvalues()
        if succ > 0:
            self.path = path
            return True

    def save(self, path=None):
        restoreditems = self.reset_temp()
        if not path:
            if self.path: path = self.path
            else: path = "config"
        configsave(path, self.active)
        for i in restoreditems: self.set_temp(*i)
        return True

    def set(self, k, v, defaults=None, limits=None, d=None):
        if defaults==limits==d==None: defaults, limits, d = self.defaults, self.limits[k], self.active
        accept = False
        if type(v)==dict:
            succ = 0
            for kk,vv in list(v.items()):
                if self.set(kk, vv, defaults[k], limits, value): succ += 1
            d[k] = v
            return succ
        elif type(limits)==list and v in limits: accept = True
        elif type(limits)==tuple:
            if limits[0]<=v<=limits[1]: accept = True
        elif type(v)==limits: accept = True
        elif limits==None: accept = True
        if accept:
            d[k] = v
            return 1
        elif k in defaults: d[k] = defaults[k]
        return 0

    def get(self, key, datatype=None):
        v = self.active.get(key, self.defaults.get(key))
        if v!=None: return v
        if datatype!=None:
            if datatype == int: return 0
            elif datatype == float: return 0.
            elif datatype == bool: return False
            elif datatype == str: return ""
            elif datatype == list: return []
            elif datatype == tuple: return ()
        return datatype

    def get_max(self, key): # from limits gets maximum
        if key in self.limits:
            if type(self.limits[key]) == tuple: return max(self.limits[key])
        return False # no designated maximum value
    def get_min(self, key): # from limits gets minimum
        if key in self.limits:
            if type(self.limits[key]) == tuple: return min(self.limits[key])
        return False # no designated minimum value

    def set_temp(self, key, value):
        if key in self.memory: return False
        current = self.get(key)
        if current!=None:
            self.memory[key] = self.memory.get(key, [])+[current]
            self.set(key, value)
            return True
    def reset_temp(self, key=None):
        if key:
            if key in self.memory:
                self.set(key, self.memory[key].pop())
                if not self.memory[key]: del self.memory[key]
                return True
        else:
            items = []
            for k in list(self.memory.keys()):
                v = self.memory[k].pop(0)
                self.set(k, v)
                del self.memory[k]
                items.append((k,v))
            return items






class manager():
    class item(): pass
    def __init__(self):
        self.keys = np.zeros(0, dtype=np.str_)
        self.free = np.zeros(0)
    def __len__(self): return (self.free==0).size
    def index(self, name): return np.logical_and(np.equal(self.keys, name), self.free==0).argmax()
    def exists(self, name): return bool(name and np.equal(self.keys, name)[self.free==0].any())
    def new(self, name, **kwargs):
        if self.exists(name): i = self.index(name)
        elif self.free.any():
            i = self.free.argmax()
            self.free[i] = 0
            self.keys[i] = name
        else:
            i = len(self.keys)
            self.free = np.append(self.free, 0)
            self.keys = np.append(self.keys, name)
        x = self.item()
        x.manager = self
        setattr(self, f"item{i}", x)
        x.index = i
        x.name = name
        for k,v in kwargs.items(): setattr(x, k, v)
        return x
    def get(self, name, default=None):
        if self.exists(name): return getattr(self, f"item{self.index(name)}")
        return default
    def set(self, name, x):
        if self.exists(name) and type(x)==self.item: setattr(self, f"item{self.index(name)}", x)
    def clear(self):
        for i in np.arange(self.free.size)[self.free==0]:
            self.free[i] = 1
            delattr(self, f"item{i}")
        self.free = self.free[:0]
        self.keys = self.keys[:0]
    def delete(self, name): self.remove(name)
    def remove(self, name):
        if self.exists(name):
            i = self.index(name)
            self.free[i] = 1
            delattr(self, f"item{i}")
            if self.free[i:].all():
                self.free = self.free[:i]
                self.keys = self.keys[:i]
    def names(self): return self.keys[self.free==0].tolist()
    def iterate(self):
        for i in np.arange(self.free.size)[self.free==0]: yield getattr(self, f"item{i}")
    def enumerate(self):
        for i,x in enumerate(self.iterate()): yield i, x


class sequencer():
    def __init__(self, size=2): # -> list of integers to loop through
        self.i = 0
        if type(size) in [int,float]: self.size = list(range(int(size)))
        elif type(size) in [list,tuple]: self.size = list(size)
        elif type(size) in [np.ndarray]: self.size = size.reshape(-1).tolist()
    def __len__(self): return len(self.r)
    def __call__(self):
        self.i = (self.i+1)%len(self.size)
        return self.size[self.i]








class dictarray(): # combined dict & array
    def __init__(self, shape=0):
        self.k = []
        self.a = np.zeros(shape)
    def __len__(self): return len(self.k)
    def index(self, k): return self.k.index(k)
    def exists(self, k): return k in self.k
    def new(self, *args, **kwargs): return self.set(*args, **kwargs)
    def set(self, k, v):
        if self.exists(k): self.a[self.index(k)] = v
        else:
            self.a = np.append(self.a, v)
            self.k.append(k)
    def add(self, k, v): self.a[self.index(k)] += v
    def delete(self, k):
        i = self.index(k)
        self.k = self.k[:i]+self.k[i+1:]
        self.a = np.append(self.a[:i], self.a[i+1:], axis=0)
    def get(self, k, default=0):
        if k in self.k: return self.a[self.k.index(k)]
        return default

    def array(self): return self.a.copy()
    def keys(self): return np.array(self.k)
    def values(self): return self.a





class functimer(dictarray):
    def __init__(self):
        super().__init__()
        self.f = {}
    def set_func(self, k, f, *args, **kwargs):
        if k in self.k: self.f[k] = (f, args, kwargs)
    def __call__(self, ms):
        self.a -= ms
        expired = np.array(self.k)[self.a<=0].tolist()
        for k in expired:
            if k in self.f:
                f, args, kwargs = self.f[k]
                self.delete(k)
                f(*args, **kwargs)
                continue
            self.delete(k)
        return expired
    def delete(self, k):
        if self.exists(k):
            super().delete(k)
            if k in self.f: del self.f[k]
    def clear(self): self.__init__()

class yieldtimer(dictarray):
    def __init__(self):
        super().__init__()
        self.y = {}
    def set_yield(self, k, x):
        if k in self.k: self.y[k] = x
    def __call__(self, ms):
        self.a -= ms
        for k in np.array(self.k)[self.a<=0].tolist():
            if k in self.y: yield self.y[k]
            self.delete(k)
    def delete(self, k):
        if self.exists(k):
            super().delete(k)
            if k in self.y: del self.y[k]
    def clear(self): self.__init__()


class expirelists():
    def new(self, x):
        setattr(self, x, [])
        setattr(self, x+"_t", np.zeros(0))
    def add(self, x, v, t):
        l = getattr(self, x)
        l.append(v)
        l_t = getattr(self, x+"_t")
        setattr(self, x+"_t", np.append(l_t, [t]))
    def pop(self, x, i):
        l = getattr(self, x)
        l_t = getattr(self, x+"_t")
        setattr(self, x, l[:i]+l[i+1:])
        setattr(self, x+"_t", np.append(l_t[:i], l_t[i+1:]))
    def remove(self, x, v):
        l = getattr(self, x)
        if v in l: self.pop(x, l.index(v))
    def set(self, x, v, t):
        l = getattr(self, x)
        if v in l:
            i = l.index(v)
            l_t = getattr(self, x+"_t")
            l_t[i] = t
    def get(self, x, v):
        l = getattr(self, x)
        if v in l: return getattr(self, x+"_t")[l.index(v)]
    def check(self, x, t):
        l_t = getattr(self, x+"_t")
        l_t -= t
        clear = l_t<=0
        setattr(self, x+"_t", l_t[~clear])
        return np.flip(np.arange(l_t.shape[0])[clear])
    def iterate(self, x, t):
        for i in self.check(x, t):
            l = getattr(self, x)
            yield l.pop(i)
    def nullify(self, x):
        l_t = getattr(self, x+"_t")
        l_t[:] = -1












##
##
##
##class lazy_init_common(): # lazy initialize an attribute common to all instances
##    def __init__(self, init_func, *args, **kwargs):
##        self.init_func = init_func
##        self.args = args
##        self.kwargs = kwargs
##        self._initialized = None
##    def __get__(self, instance, owner):
##        if self._initialized is None: self._initialized = self.init_func(*self.args, **self.kwargs)
##        return self._initialized
##
##class lazy_init_attr(object): # wrapper to lazy initialize an attribute per instance
##    def __init__(self, function):
##        self.fget = function
##    def __get__(self, obj, cls):
##        value = self.fget(obj)
##        setattr(obj, self.fget.__name__, value)
##        return value
####class my_class():
####    x = lazy_init_common(sorted, np.random.rand(8))
####    def __init__(self): pass
####    @lazy_init_attr
####    def lazy_x(self): return sorted(np.random.rand(3))
##
####@dataclass() # -> creates __init__, __repr__, __eq__
####class dataclass:
####    x:int
####    y:int = field(default=10)
####    z:list = field(default_factory=list)
##    
####@dataclass(frozen=True)
####class frozendataclass:
####    x:int
####    y:int
##
##
##


class binarytree(): # keys strings only
    def __init__(self, key="", value=0): # set a root value to somewhere in the middle for speed
        self.parent = None
        self.key = key
        self.value = value
    def set(self, key, value):
        if key==self.key: return self.value
        else:
            if sorted([key, self.key])[0]==key:
                if hasattr(self, "less"): return self.less.set(key, value)
                self.less = binarytree(key, value)
                self.less.parent = self
                return self.less.value
            else:
                if hasattr(self, "more"): return self.more.set(key, value)
                self.more = binarytree(key, value)
                self.more.parent = self
                return self.more.value
    def get(self, key, default=None):
        if key==self.key: return self.value
        else:
            s = sorted([key, self.key])
            if s[0]==key and hasattr(self, "less"): return self.less.get(key, default)
            elif s[0]==self.key and hasattr(self, "more"): return self.more.get(key, default)
        return default

    def delete_check(self):
        if not self.value and not (hasattr(self, "less") or hasattr(self, "more")):
            if getattr(self.parent, "less", None) is self: del self.parent.less
            elif getattr(self.parent, "more", None) is self: del self.parent.more

    def pop(self, key):
        if key==self.key: value, self.value = self.value, None
        else:
            value = None
            s = sorted([key, self.key])
            if s[0]==key and hasattr(self, "less"): value = self.less.pop(key)
            elif s[0]==self.key and hasattr(self, "more"): value = self.more.pop(key)
        if value!=None: self.delete_check()
        return value








class database:
    def __init__(self):
        self.index = np.zeros(0,dtype=np.str_)
        self.columns = np.array(["index"])
        
    def __len__(self): return len(self.index)
    def exists(self, index): return (self.index==index).any()

    def _get_dtype(self, v):
        v_type = type(v)
        if v_type==int: v_dtype = np.int32
        elif v_type==float: v_dtype = np.float32
        else: v_dtype = np.dtype("U20")
        return v_dtype
    
    def _new_column(self, k, dtype):
        n = len(self.index)
        setattr(self, k, np.zeros(n, dtype=dtype))
        self.columns = np.append(self.columns, str(k))

    def new(self, index, **kwargs):
        if self.exists(index): self.remove(index)
        unaffected_cols = [k for k in self.columns if not k in kwargs.keys() and k!="index"]
        for k,v in kwargs.items():
            v_dtype = self._get_dtype(v)
            if not hasattr(self, k): self._new_column(k, v_dtype)
            col = getattr(self, k)
            setattr(self, k, np.append(col, [v], axis=0))
        for k in unaffected_cols:
            col = getattr(self, k)
            setattr(self, k, np.append(col, [0], axis=0))
        self.index = np.append(self.index, str(index))
        return True
    def set(self, index, **kwargs):
        if self.exists(index):
            i = np.argmax(self.index==index)
            for k,v in kwargs.items():
                v_dtype = self._get_dtype(v)
                if not hasattr(self, k): self._new_column(k, v_dtype)
                col = getattr(self, k)
                col[i] = v
            return True
        return False
    def add(self, index, **kwargs):
        if self.exists(index):
            i = np.argmax(self.index==index)
            for k,v in kwargs.items():
                v_dtype = self._get_dtype(v)
                if not hasattr(self, k): self._new_column(k, v_dtype)
                col = getattr(self, k)
                if hasattr(col[i], "__add__") and hasattr(v, "__add__"): col[i] += v
                else: col[i] = v
            return True
        return False
    def get(self, index, key=None):
        if self.exists(index):
            i = np.argmax(self.index==index)
            if key is None: return {k:getattr(self, k)[i] for k in self.columns}
            elif key in self.columns: return getattr(self, key)[i]
    def remove(self, index):
        def column_delete(k):
            delattr(self, k)
            self.columns = self.columns[self.columns!=k]
        if self.exists(index):
            valid = self.index!=index
            for k in self.columns.tolist():
                if k!="index":
                    arr = getattr(self, k)[valid]
                    setattr(self, k, arr)
                    if (arr=="").all() or (arr==0).all(): column_delete(k)
            return True
        return False
    
    def sort(self, k, reverse=False):
        if len(self) and hasattr(self, k):
            sort_col = getattr(self, k)
            sort_indexes = np.argsort(sort_col)
            if reverse: sort_indexes = np.flip(sort_indexes)
            for k in self.columns: setattr(self, k, getattr(self, k)[sort_indexes])
            return True
        return False

    def view(self, page, amount, exclude=()):
        if len(self)==0: return np.zeros((0,len(self.columns)+1), dtype=self._get_dtype(0))
        lastpage = int(len(self)/amount)
        i = min(page, lastpage)*amount
        s = slice(i,i+amount)
        return {k:getattr(self, k)[s] for k in self.columns if not k in exclude}

    
    def sample(self, n=10, **kwargs): return [self.get(index) for index in self.index_sample(n, **kwargs)]
    def index_sample(self, n=10, **kwargs):
        indexes = self.filter(**kwargs)
        return random.sample(indexes.tolist(), min(n, len(indexes)))
    
    def filter(self, **kwargs):
        valid = np.ones(self.index.size, dtype=np.bool_)
        for k,v in kwargs.items():
            if k in self.columns: valid = np.logical_and(valid, getattr(self, k)==v)
        return self.index[valid]





class database2:
    class instance_class():
        mask = None
        sort = None
    def __init__(self, size=1e4):
        self.instances = {}
        self.__size = int(size)
        self.index = np.zeros(self.__size, dtype=np.str_) # i -> name
        self.__columns = np.zeros(0, dtype=np.str_)
        
        self.__index_dict = {} # name -> i
        self.__index_bool = self.index!=""
        self.__freeside_start = 0
        self.__additional_free_space = []
        
    def __len__(self): return self.__freeside_start-len(self.__additional_free_space)
    def exists(self, index): return index in self.__index_dict

    def clear(self, size=None):
        self.instances.clear()
        if size is not None: self.__size = size
        self.index = np.zeros(self.__size, dtype=np.str_) # i -> name
        for k in self.__columns.tolist(): self.delete_column(k)
        self.__index_dict.clear()
        self.__index_bool = self.index!=""
        self.__freeside_start = 0
        self.__additional_free_space.clear()

    def clean(self): # columns
        indexes = np.arange(self.__size)[~self.__index_bool]
        if indexes.size:
            for k in self.__columns.tolist():
                col = getattr(self, k)
                col[indexes] = "" if is_string(col) else 0
                self._set_col_value(indexes[-1], k, 0)

    def _get_dtype(self, v, orig_dtype=None):
        if is_integer(v): v_dtype = np.dtype(np.int8 if -128<=v<128 else (np.int16 if -2**15<=v<2**15 else (np.int32 if -2**31<=v<2**31 else np.int64)))
        elif is_float(v): v_dtype = np.dtype(np.float16 if -2**15<=v<2**15 else (np.float32 if -2**31<=v<2**31 else np.float64))
        elif is_string(v): v_dtype = np.dtype(f"U{len(v)}")
        if orig_dtype is not None: # compare
            odt = str(orig_dtype)
            vdt = str(v_dtype)
            if odt[:2]!=vdt[:2]: return None # wrong type
            if vdt[:2]=="<U" and int(odt[2:])>=len(v): v_dtype = orig_dtype
            if vdt[:2]=="fl" and int(odt[5:])>=int(vdt[5:]): v_dtype = orig_dtype
            if vdt[:2]=="in" and int(odt[3:])>=int(vdt[3:]): v_dtype = orig_dtype
        return v_dtype

    
    def new_column(self, k, v):
        dtype = self._get_dtype(v)
        setattr(self, k, np.zeros(self.__size, dtype=dtype))
        self.__columns = np.append(self.__columns, str(k))
        if str(dtype)[:2]=="<U": setattr(self, f"_{k}_not_empty", np.zeros(self.__size, dtype=np.bool_))
    def delete_column(self, k):
        col = getattr(self, k)
        if str(col.dtype)[:2]=="<U": delattr(self, f"_{k}_not_empty")
        delattr(self, k)
        self.__columns = self.__columns[self.__columns!=k]
    def get_column(self, k, instance=None, apply_mask=True):
        inst = self.get_instance(instance)
        mask = inst.mask
        col = getattr(self, k)
        if inst.sort is not None: col = col[inst.sort]
        if apply_mask and mask is not None:
            if inst.sort is not None: mask = mask[inst.sort]
            col = col[mask]
        return col
    
    def _set_col_value(self, i, k, v, add=False):
        vb = bool(v)
        if vb:
            if not hasattr(self, k): self.new_column(k, v)
            col = getattr(self, k)
            if add:
                if hasattr(col[i], "__add__") and hasattr(v, "__add__"): v = col[i]+v
                else: return False
            new_dtype = self._get_dtype(v, col.dtype)
            if new_dtype and new_dtype!=col.dtype:
                col = col.astype(new_dtype)
                setattr(self, k, col)
            col[i] = v
        else:
            col = getattr(self, k)
            col[i] = np.zeros(1, dtype=col.dtype)[0]
        if str(col.dtype)[:2]=="<U":
            col = getattr(self, f"_{k}_not_empty")
            col[i] = vb
##        if not vb: # SO EXPENSIVE
##            s = col.sum()
##            if s==0 or (s>len(self.__index_dict) and (~self.__index_bool[col]).all()): self.delete_column(k)

    

    def reserve_space(self):
        if self.__additional_free_space: i = self.__additional_free_space.pop()
        else:
            if self.__freeside_start==self.__size: return None
            else:
                i = self.__freeside_start
                self.__freeside_start += 1
        return i
    def release_space(self, i):
        if i==self.__freeside_start-1:
            self.__freeside_start -= 1
            afs = sorted(self.__additional_free_space)
            while afs and afs[-1]==self.__freeside_start-1:
                self.__freeside_start -= 1
                afs.pop()
            self.__additional_free_space = afs
        else: self.__additional_free_space.append(i)
    def is_freespace(self, i): return i>=self.__freeside_start or i in self.__additional_free_space

    def new(self, index, **kwargs):
        index_str = str(index)
        if self.exists(index): i = self.__index_dict[index_str]
        else: i = self.reserve_space()
        if i is None: return False # full

        self.__index_dict[index_str] = i
        new_dtype = self._get_dtype(index_str, self.index.dtype)
        if new_dtype and new_dtype!=self.index.dtype: self.index = self.index.astype(new_dtype)
        self.index[i] = index_str
        self.__index_bool[i] = True
        
        for k,v in kwargs.items(): self._set_col_value(i, k, v)
        for k in self.__columns:
            if not k in kwargs: self._set_col_value(i, k, 0)
        for k,v in self.instances.items(): v.mask[i] = False
        return True
    def set(self, index, **kwargs):
        if self.exists(index):
            i = self.__index_dict[index]
            for k,v in kwargs.items(): self._set_col_value(i, k, v)
            return True
        return False
    def add(self, index, **kwargs):
        if self.exists(index):
            i = self.__index_dict[index]
            for k,v in kwargs.items(): self._set_col_value(i, k, v, add=True)
            return True
        return False
    def remove(self, index):
        if self.exists(index):
            i = self.__index_dict[index]
            del self.__index_dict[index]
            self.index[i] = ""
            self.__index_bool[i] = False
            for k,v in self.instances.items(): v.mask[i] = False
            self.release_space(i)
            return True
        return False
    def get(self, index, k=None, default=None):
        if self.exists(index):
            i = self.__index_dict[index]
            if k is None: return {"index":index}|{k:getattr(self, k)[i] for k in self.__columns}
            elif k in self.__columns: return getattr(self, k)[i]
        return default
    



    def new_instance(self, instance):
        self.instances[instance] = self.instance_class()
        self.instances[instance].mask = self.__index_bool.copy()
    def get_instance(self, instance):
        if instance in self.instances: return self.instances[instance]
        self.new_instance(instance)
        return self.instances[instance]
    def delete_instance(self, instance=None):
        if instance in self.instances: del self.instances[instance]
    
    
    def sort(self, *ks, reverse=False, instance=None):
        inst = self.get_instance(instance)
        if len(self):
            sort = inst.sort
            for k in ks:
                if hasattr(self, k):
                    col = getattr(self, k)
                    if sort is None: sort = np.argsort(col)
                    else: sort = sort[np.argsort(col[sort], kind="stable")] # quicksort fails here
                else: continue
            else:
                inst.sort = sort
                if reverse: inst.sort = np.flip(inst.sort)
                return True
        return False
    def reverse(self, instance=None):
        inst = self.get_instance(instance)
        if inst.sort is None: inst.sort = np.flip(np.arange(self.__size))
        else: inst.sort = np.flip(inst.sort)


    def view(self, page=0, amount=10, exclude=(), instance=None):
        if len(self)==0: return
        pages = len(self)//amount
        i = min(page, pages)*amount
        s = slice(i,i+amount)
        return {"index":self.get_column("index", instance)[s]}|{k:self.get_column(k, instance)[s] for k in self.__columns if not k in exclude}
##    def select(self, **kwargs):
##        selection = None
##        for k,v in kwargs.items():
##            col = getattr(self, k)
##            if selection is not None: selection = np.logical_and(selection, col==v)
##            else: selection = col==v
##            if not selection.any(): return {}
##        return {"index":getattr(self, "index")[selection]}|{k:getattr(self, k)[selection] for k in self.__columns}
    def search(self, index, n=10): return strsearch(index, self.index, n)

    def sample(self, n=10, instance=None): return [self.get(index) for index in self.index_sample(n, instance=instance)]
    def index_sample(self, n=10, instance=None):
        inst = self.get_instance(instance)
        valid = self.__index_bool[:self.__freeside_start]
        indexes = self.index[:self.__freeside_start][valid]
        if inst.mask is not None: indexes = indexes[inst.mask[:self.__freeside_start][valid]]
        return random.sample(indexes.tolist(), min(n, len(indexes)))

    def set_mask(self, mask=None, instance=None):
        inst = self.get_instance(instance)
        inst.mask = mask
    def add_mask(self, mask, instance=None, exclude=True):
        inst = self.get_instance(instance)
        if inst.mask is None: inst.mask = mask
        else: inst.mask = (np.logical_or if not exclude else np.logical_and)(mask, inst.mask)
        





class database2_manager:
    def __init__(self, disk_location):
        self.disk_location = disk_location
        self.folders = {}
        self.files = {}
    def __len__(self):
        l = 0
        for v in self.folders.values(): l += len(v)
        for v in self.files.values(): l += len(v)
        return l
    
    def navigate(self, *dirs, make_dirs=False):
        x = self
        for d in dirs:
            if d in x.folders: x = x.folders[d]
            elif make_dirs:
                x.folders[d] = database2_manager(os.path.join(self.disk_location, d))
                x = x.folders[d]
            else: return None
        return x

    def _unload(self, dbl, name):
        save(os.path.join(dbl.disk_location, name)+f".database", dbl.files[name])
        dbl.files[name] = None
    def _reload(self, dbl, name): dbl.files[name] = load(os.path.join(dbl.disk_location, name)+f".database")
    def toggle(self, path):
        dirs, name, ext = ospck.explode(path)
        dbl = self.navigate(*dirs)
        if dbl and name in dbl.files:
            if dbl.files[name] is None: self._reload(dbl, name)
            else: self._unload(dbl, name)
    def unload(self, path):
        dirs, name, ext = ospck.explode(path)
        dbl = self.navigate(*dirs)
        if dbl and name in dbl.files: self._unload(dbl, name)
    def reload(self, path):
        dirs, name, ext = ospck.explode(path)
        dbl = self.navigate(*dirs)
        if dbl and name in dbl.files: self._reload(dbl, name)
            
    
    def new(self, path, size=1e4):
        dirs, name, ext = ospck.explode(path)
        x = self.navigate(*dirs, make_dirs=True)
        x.files[name] = database2(size)
    def get(self, path, force=False):
        dirs, name, ext = ospck.explode(path)
        x = self.navigate(*dirs)
        if x and name in x.files:
            db = x.files[name]
            if db is None and force:
                x.reload(name)
                db = x.files[name]
                x.unload(name)
            return db

    def save(self, path=None):
        if path is None: path = self.disk_location
        for k,v in self.folders.items(): v.save(os.path.join(path, k))
        for k,v in self.files.items():
            k += f".database"
            if v is None: filecopy(os.path.join(self.disk_location, k), os.path.join(path, k))
            else: save(os.path.join(path, k), v)
    def load(self, path=None):
        if path is None: path = self.disk_location
        for f in ospck.list_folders(path):
            dirs, name, ext = ospck.explode(f)
            self.folders[name] = database2_manager(f)
            self.folders[name].load(f)
        for f in ospck.list_files(path):
            dirs, name, ext = ospck.explode(f)
            if ext=="database": self.files[name] = load(f)
        













def selection_sort(l, reverse=False):
    n = len(l)
    for i in range(n):
        min_i = i
        for j in range(i+1, n):
            if l[min_i]>l[j]: min_i = j
        if i!=min_i: l[i],l[min_i] = l[min_i],l[i]
def bubble_sort(l): # swap items next to each other until sorted
    n = len(l)
    for i in range(n):
        swapped = False
        for j in range(0, n-i-1):
            if l[j]>l[j+1]:
                l[j], l[j+1] = l[j+1], l[j]
                swapped = True
        if not swapped: break
def quick_sort(l, low=None, high=None):
    def partition(l, low, high):
        pivot = l[high]
        i = low-1
        for j in range(low, high):
            if l[j]<=pivot:
                i += 1
                l[i], l[j] = l[j], l[i]
        l[i+1], l[high] = l[high], l[i+1]
        return i+1
    if low==None: low = 0
    if high==None: high = len(l)-1
    if low<high:
        pi = partition(l, low, high)
        quick_sort(l, low, pi-1)
        quick_sort(l, pi+1, high)
def heap_sort(l):
    n = len(l)
    def heapify(n, i):
        largest = i # Initialize largest as root
        left = 2*i+1
        right = 2*i+2
        if left<n and l[largest]<l[left]: largest = left
        if right<n and l[largest]<l[right]: largest = right
        if largest != i:
            l[i], l[largest] = l[largest], l[i]
            heapify(n, largest)
    for i in range(n//2-1, -1, -1): heapify(n, i)
    for i in range(n-1, 0, -1):
        l[i], l[0] = l[0], l[i]
        heapify(i, 0)
def count_sort(integers):
    n = len(integers)
    M = max(integers)
    count_array = [0]*(M+1)
    for num in integers: count_array[num] += 1
    for i in range(1, M+1): count_array[i] += count_array[i-1]
    output_array = [0]*n
    for i in range(n-1, -1, -1):
        output_array[count_array[integers[i]]-1] = integers[i]
        count_array[integers[i]] -= 1
    integers[:] = output_array[:]









def insertion_sort(l, start=0, end=None, cmp=None): # move smaller items left VERY SLOW
    if cmp is None: cmp = lambda x,y: x<y
    if end is None: end = len(l)
    n = end-start
    for i in range(start+1, n):
        v = l[i]
        ii = i-1
        while ii>=0 and cmp(v, l[ii]):
            l[ii+1] = l[ii]
            ii -= 1
        l[ii+1] = v

def merge_sort(l, start=0, end=None, cmp=None): # USE FOR LARGE ARRAYS -> guaranteed O(n log(n))
    if cmp is None: cmp = lambda x,y: x<y
    if end is None: end = len(l)
    n = end-start
    if n>1:
        mid = n//2 # Finding the mid of the array
        L = l[start:start+mid]
        R = l[start+mid:end]
        merge_sort(L, cmp=cmp) # Sorting the first half
        merge_sort(R, cmp=cmp) # Sorting the second half
        i = j = k = 0
        while i<len(L) and j<len(R):
            if cmp(L[i], R[j]): # L[i]<=R[j]
                l[k] = L[i]
                i += 1
            else:
                l[k] = R[j]
                j += 1
            k += 1
        while i<len(L): # Checking if any element was left
            l[k] = L[i]
            i += 1
            k += 1
        while j<len(R):
            l[k] = R[j]
            j += 1
            k += 1

def reverse(l, start=0, end=None):
    if end is None: end = len(l)
    for i in range(start, end):
        ii = end-(i-start)-1
        if ii<=i: break
        l[i], l[ii] = l[ii], l[i]









class LinkedList():
    head = None
    tail = None
    length = 0
    reverse = False
    class Node():
        def __init__(self, data):
            self.data = data
            self.next = None
            self.prev = None

    def __len__(self): return self.length

    def __init__(self, data=None):
        if hasattr(data, "__iter__"):
            for d in data: self.append(d)
        elif data is not None: self.head = self.Node(data)

    def __insert_before(self, node, data):
        new_node = self.Node(data)
        new_node.next = node
        new_node.prev = node.prev
        if node.prev is not None: node.prev.next = new_node
        node.prev = new_node
        self.length += 1
        return new_node
        
    def __insert_after(self, node, data):
        new_node = self.Node(data)
        new_node.prev = node
        new_node.next = node.next
        if node.next is not None: node.next.prev = new_node
        node.next = new_node
        self.length += 1
        return new_node

    def print_node(self, node):
        print(id(node))
        print("\t", id(node.prev) if node.prev is not None else None)
        print("\t", id(node.next) if node.next is not None else None)

    def append(self, data):
        if self.length>0:
            if self.reverse: self.head = self.__insert_before(self.head, data)
            else: self.tail = self.__insert_after(self.tail, data)
        else:
            self.head = self.Node(data)
            self.tail = self.head
            self.length = 1

    def insert(self, index, data):
        if index<0: index += self.length+1
        if index==self.length: self.append(data)
        elif 0<=index<self.length:
            if self.reverse!=(index<(self.length>>1)):
                node = self.head
                for _ in range(index): node = node.next
                new_node = self.__insert_before(node, data)
                if node is self.head: self.head = new_node
            else:
                node = self.tail
                for _ in range(self.length-index-1): node = node.prev
                new_node = self.__insert_after(node, data)
                if node is self.tail: self.tail = new_node
        else:
            raise IndexError

    def pop(self, index=None):
        if self.length==0: return
        if index is None: index = self.length-1
        elif index<0: index += self.length+1
        if 0<=index<self.length:
            if index<(self.length>>1):
                node = self.head
                for _ in range(index):
                    node = node.next
            else:
                node = self.tail
                for _ in range(self.length-index-1):
                    node = node.prev
            if node.prev is not None: node.prev.next = node.next
            if node.next is not None: node.next.prev = node.prev
            if node is self.head: self.head = node.next
            if node is self.tail: self.tail = node.prev
            self.length -= 1
            return node.data
        else:
            raise IndexError

    def clear(self):
        self.length = 0
        if self.tail is not None:
            node = self.tail
            while node.prev is not None:
                node = node.prev
                node.next.prev = None
                node.next = None
            self.head = self.tail = None

    def __iter__(self):
        node = self.tail if self.reverse else self.head
        while node is not None:
            yield node.data
            node = node.prev if self.reverse else node.next


    
        
        
class IntTable():
    # memory = O(3n)
    # store values & argsort per column
    capacity = 8
    length = 0
    dtype = np.int64
    index_dtype = np.int64
    reversed = False
    sort_column = -1
    
    def __init__(self, capacity, width):
        self.capacity = int(capacity)
        self.current_sort_order = np.zeros(self.capacity, dtype=self.index_dtype)
        self.value_array = np.zeros((self.capacity,width), dtype=self.dtype)
        self.argsort_array = np.zeros((self.capacity,width), dtype=self.index_dtype)
        self.sorted_values = np.zeros((self.capacity,width), dtype=self.dtype)
        self.clear_sort()

    def find_new_index_recur(self, sorted_values, value, start, end):
        mid = (end-start)//2+start
        if sorted_values[mid]==value: return mid+1
        if sorted_values[mid]<value:
            if (mid+1)>=end: return end
            return self.find_new_index_recur(sorted_values, value, mid, end)
        if start==mid: return start
        return self.find_new_index_recur(sorted_values, value, start, mid)

    def is_full(self): return self.length==self.capacity
    def is_empty(self): return self.length==0
    
    def insert(self, *values):
        if len(values)!=self.value_array.shape[1] or self.is_full(): return False
        for i in range(self.value_array.shape[1]):
            value_array = self.value_array[:self.length+1,i]
            argsort_array = self.argsort_array[:self.length+1,i]
            sorted_values = self.sorted_values[:self.length+1,i]
            value_array[self.length] = values[i]
            if not self.is_empty():
                new_index = self.find_new_index_recur(sorted_values, values[i], 0, self.length)
                argsort_array[new_index+1:self.length+1] = argsort_array[new_index:self.length]
                sorted_values[new_index+1:self.length+1] = sorted_values[new_index:self.length]
                argsort_array[new_index] = self.length
                sorted_values[new_index] = values[i]
            else:
                argsort_array[0] = 0
                sorted_values[0] = values[i]
        self.current_sort_order[self.length] = self.length
        self.length += 1
        return True

    def replace(self, index, *values, sorted=True):
##        if self.is_empty(): return False
##        l = self.length
##        print("replace")
##        if sorted: index = self.current_sort_order[:l][index]
##        for i in range(self.value_array.shape[1]):
##            value_array = self.value_array[:l,i]
##            argsort_array = self.argsort_array[:l,i]
##            sorted_values = self.sorted_values[:l,i]
##
##            old_index = argsort_array[index]
##            if value_array[index]>values[i]:
##                new_index = self.find_new_index_recur(sorted_values[:old_index+1], values[i], 0, old_index+1)
####            elif self.length>old_index:
##            else:
##                new_index = old_index+self.find_new_index_recur(sorted_values[old_index:], values[i], 0, self.length-old_index)
####            else: new_index = old_index
##            print(old_index, new_index)
##            
##            value_array[index] = values[i]
##            argsort_array[index] = new_index
##            sorted_values[index] = values[i]
##
####            if i==self.sort_column:
####                self.current_sort_order[index] = new_index
        return True

    def clear_sort(self):
        self.current_sort_order = np.zeros(self.capacity, dtype=self.index_dtype)
        self.current_sort_order[:self.length] = np.arange(self.length, dtype=self.index_dtype)

    def calc_sort(self):
        for i in range(self.value_array.shape[1]):
            value_array = self.value_array[:self.length,i]
            argsort_array = np.argsort(value_array)
            self.argsort_array[:self.length,i] = argsort_array
            self.sorted_values[:self.length,i] = value_array[argsort_array]
        
    def sort(self, column=0):
        self.sort_column = column
        argsort_array = self.argsort_array[:self.length, column]
        self.current_sort_order[:self.length] = self.current_sort_order[:self.length][argsort_array]

    def show(self, index=0, amount=10):
        return self.value_array[:self.length][self.current_sort_order[:self.length][::((1-self.reversed)*2-1)]][index:(index+amount)]
    
    def get(self, index=None):
        if index is None: index = self.length-1
        elif self.length==0 or index<0 or index>=self.length: return
        return self.value_array[:self.length][index]

    def pop(self, index=None):
        if index is None: index = self.length-1
        if self.is_empty() or index<0 or index>=self.length: return

        popped = self.value_array[:self.length][index].copy()
        self.value_array[index:self.length] = self.value_array[index+1:self.length+1]
        self.argsort_array[index:self.length] = self.argsort_array[index+1:self.length+1]
        self.sorted_values[index:self.length] = self.sorted_values[index+1:self.length+1]

        self.current_sort_order[self.current_sort_order>=index] -= 1
        self.current_sort_order[index:self.length] = self.current_sort_order[index+1:self.length+1]
        self.length -= 1
        return popped


if __name__ == "__main__":
####    it = IntTable(1e6, 3)
######    def test_print():
######        print(it.a_values[it.a_argsort[:it.length]])
######        print(it.b_values[it.b_argsort[:it.length]])
######        print(it.c_values[it.c_argsort[:it.length]])
####    l = []
####    import random
####    import timepck
####    
####    n = 100000
####    t_start = timepck.nspec()
####    for i in range(n):
######        l.append((random.randint(0, 10000), random.randint(0, 10000), random.randint(0, 10000)))
####        it.insert(random.randint(0, 10000), random.randint(0, 10000), random.randint(0, 10000))
####    t_end = timepck.nspec()-t_start
####    print(t_end // int(1e6), t_end//n)
####
####    t_start = timepck.nspec()
####    it.sort(0)
######    ll = sorted(l, key=lambda x:x[0])
####    t_end = timepck.nspec()-t_start
####    print(t_end // int(1e6), t_end//n)
    
##    print(it.get(0))
##    it.replace(5, 10,10,10)
####    it.sort(0)
####    it.calc_sort()
##    print(it.show(0, 10))
####    it.clear_sort()
####    it.calc_sort()
####    it.sort(0)
####    print(it.show(0, 10))
    
##    t_start = timepck.nspec()
##    it.calc_sort()
##    it.sort(2)
##    t_end = timepck.nspec()-t_start
##    print(t_end // int(1e6), t_end//n)
    
##    t_start = timepck.nspec()
##    it.reversed = True
##    result = it.show(0, 20)
##    t_end = timepck.nspec()-t_start
##    print(t_end // int(1e6), t_end//n)
##    print(result)
    
##    print(it.get(10))
##    print(it.pop(10))
##    print(it.get())
##    print(it.pop())
##    print(it.get(0))
##    print(it.pop(0))
##    print(it.get(0))
##    print(it.pop(0))

    
##    import timepck
##
##    ll = LinkedList()
##    l = []
##
##    @timepck.func_timer_decor
##    def test(func, amount):
##        for i in range(int(amount)):
##            func(i)
##            
##    @timepck.func_timer_decor
##    def test_reverse(func, amount):
##        for i in range(int(amount)-1, -1, -1):
##            func(i)
##
##    n = 1e6
##    test(l.append, n)
##    test_reverse(l.pop, n)
##    
##    test(ll.append, n)
##    test_reverse(ll.pop, n)
    
    
######    
########    ll.append(1)
########    ll.append(2)
########    ll.append(3)
########    ll.append(4)
########    ll.append(5)
######    
######    ll.insert(-1, 2)
######    ll.insert(-1, 3)
######    ll.insert(-1, 4)
######    ll.insert(-1, 1)
######    
########    for i in range(5):
########        print(i)
########        ll.insert(0, i*10)
######    print("")
########    ll.reverse = False
######    for i,v in enumerate(ll):
######        print(i,v)
########    print("")
########    for i in [x for x in ll]:
########        print(ll.pop())

    
##    import json
##    from timepck import nspec
##    t_start = nspec()
##    with open("100000_coders.json", "r", encoding="utf8") as f: data = json.load(f)
##    print((nspec()-t_start)/1e6)
##
##    def compare(x, y):
##        if x["lastName"]<y["lastName"]: return True
##        elif x["lastName"]==y["lastName"]: return x["firstName"]<y["firstName"]
##        return False
##    
##    f_sort = merge_sort
####    f_sort = insertion_sort
##    
##    t_start = nspec()
##    f_sort(data, 0, cmp=compare)
##    print((nspec()-t_start)/1e6)
##
####    reverse(data)
##    for i in range(100):
##        x = data[i]
##        print(x["lastName"], x["firstName"])



##    print(strmatch("xdef", [ospck.randomstr(6) for i in range(10000)]))
    
##    if 1: # sorting test
##        import timepck
##        import random
##
##        n = int(1e5)
##        print("n", n)
##        ll = [str(random.randint(0,n)) for i in range(n)]
##
##        print("numpy sort")
##        t_start = timepck.nspec()
##        l = np.array(ll)
##        l.sort(kind="quicksort")
##        l = l.tolist()
##        print((timepck.nspec()-t_start)/1e6, "ms")
##        print(l[:20], end="\n\n")
##        
##        for x in ["heap_sort","merge_sort","quick_sort"]: # "bubble_sort", "insertion_sort","count_sort"
##            l = ll.copy()
##            print(x)
##            t_start = timepck.nspec()
##            globals()[x](l)
##            print((timepck.nspec()-t_start)/1e6, "ms")
##            print(l[:20], end="\n\n")
            
##    if 1: # strsearch_test
##        import timepck
##        import random
##        t_start = timepck.nspec()
##        options = [" "+ospck.randomstr(random.randint(8,15)) for i in range(int(1e6))] #.replace("z", "a")
##        print((timepck.nspec()-t_start)/1e6, "ms")
##        w = "appleorange"
##        t_start = timepck.nspec()
##        results = strsearch(w, options, 10)
##        print((timepck.nspec()-t_start)/1e6, "ms")
##        print(results)
####        t_start = timepck.nspec()
####        results = strsearch2(w, options, 10)
####        print((timepck.nspec()-t_start)/1e6, "ms")
####        print(results)
    
    
##    for i,x in enumerate(copy_gen("dbl", "dbl2", fraction=True)): print(i, x)

##    dbl = database2_manager("dbl")
##    dbl.load()
##    db = dbl.get("db1")
##
##    dbl_a = dbl.navigate("a")
##    db = dbl_a.get("db1")
##    
##    dbl.new("db1", 1e5)
##    dbl.new("db2", 1e5)
##    dbl.new("a/db1", 1e5)
##    dbl.new("a/db2", 1e5)
##
##    dbl.unload("db1")
##    dbl.unload("a/db1")
##    dbl.reload("a/db1")
##    dbl.save("dbl3")

    
##    def database2_test():
##        import timepck
##        import random
##        
##        n = int(1e6)
##        db = database2(n)
##        db = load("datapck_database2_full.save")
####        db.remove("1")
##        print("n =", n, len(db))
##        
######        save("datapck_database2_empty.save", db)
########        n = n//10
######        t_start = timepck.nspec() # random.randint(1, 1e3)
######        for i in range(n): db.new(f"{i}", a=random.randint(1, 1e4), b=random.randint(1, 1e2))
######        print("\n", "new", (timepck.nspec()-t_start)/1e6, "ms", end="\n\n")
######        
######        save("datapck_database2_full.save", db)
##
####        t_start = timepck.nspec()
####        for i in range(n): db.set(f"{i}", a=random.randint(1, 1e3), b=random.randint(1, 1e3))
####        print("\n", "set", (timepck.nspec()-t_start)/1e6, "ms", end="\n\n")
####        t_start = timepck.nspec()
####        for i in range(n): db.add(f"{i}", a=random.randint(1, 1e3), b=random.randint(1, 1e3))
####        print("\nadd", (timepck.nspec()-t_start)/1e6, "ms")
####        save("datapck_database2_test.save2", db)
##
##        db.add_mask(db.a>1, instance="test1")
##        db.add_mask(db.b>5, instance="test1")
##        
##        t_start = timepck.nspec()
##        db.sort("b", "a", instance="test1")
##        db.sort("a", "b", instance="test2")
##        db.sort("index", "b", instance="test3")
##        print("\n", "sort x3", (timepck.nspec()-t_start)/1e6, "ms", end="\n\n")
##        
##        t_start = timepck.nspec()
##        view1 = db.view(instance="test1")
##        view2 = db.view(instance="test2")
##        view3 = db.view(instance="test3")
##        print("\n", "view x3", (timepck.nspec()-t_start)/1e6, "ms", end="\n\n")
##        print(view1)
##        print(view2)
##        print(view3)
##
####        t_start = timepck.nspec()
####        selected = db.select(a=100000)
####        print("\n", "select", (timepck.nspec()-t_start)/1e6, "ms", end="\n\n")
####        print(selected)
####
####        t_start = timepck.nspec()
####        out = db.search("51795000", 1)
####        print("\n", "search", (timepck.nspec()-t_start)/1e6, "ms", end="\n\n")
####        print(out)
##        
######        print("\n", "sample", db.sample(10, "test1"), end="\n\n")
##        
######        t_start = timepck.nspec()
######        for i in range(n): db.remove(f"{i}")
######        print("\n", "remove", (timepck.nspec()-t_start)/1e6, "ms", end="\n\n")
##        
####        save("datapck_database2_test.save4", db)
####        db.clean()
####        save("datapck_database2_test.save4_clean", db)
####        db.clear()
####        save("datapck_database2_test.save4_clear", db)
##        
######        t_start = timepck.nspec()
######        for i in range(n): db.new(f"{i}", a=random.randint(1, 1e3)) # , b=random.randint(1, 1e3), c=random.randint(1, 1e3)
######        print("\n", "new", (timepck.nspec()-t_start)/1e6, "ms", end="\n\n")
######        del db
##    database2_test()

    
    
##    db.new("item2", c=6)
##    db.delete_instance(instance="test")
##    db.sort("a", instance="test")
##    db.reverse("test")
##    db.add_mask(db.a>1, instance="test")
##    db.add_mask(db.a==0, instance="test", positive=True)
##    print(db.view(instance="test"))
    
##    a = 8
##    b = 2
##    print(bin(a|b))
##    print(bin(a&b))
##    print(bin(a%b^b))

##    a = np.zeros(1, dtype=np.unicode_)
##    print(a.dtype, isinstance(a[0], np.flexible), a)

##    a = [10]#"asd"
##    print(type(a)())

####    v = "0"
####    a = np.array(["asd","1"], dtype=np.str_)
####    a = a.astype(np.array([v]).dtype)
####    a[0] = v
####    print(a.itemsize)
    
##    print((a=="").all(), a.dtype)
    
##    print(strguess("abcd", [ospck.randomstr(4) for i in range(100)], 1))

    

##    import timepck
##    import random
##    
##    bt = binarytree()
##    numbers = list(range(10**6))
##    random.shuffle(numbers)
##    d = {}
##    for i,n in enumerate(numbers):
##        d[n] = i
##        bt.set(str(n), str(i))
##
##    numbers2 = numbers.copy()
##    random.shuffle(numbers2)
##    print("shuffled")
##    
##    starttime = timepck.nspec()
##    indexes = [numbers.index(n) for n in numbers2[:1000]]
##    endtime = timepck.nspec()
##    print("list index time", (endtime-starttime)/10**6, "ms") # ~2000 times slower
##    
##    starttime = timepck.nspec()
##    indexes = [d.get(n) for n in numbers2[:1000]]
##    endtime = timepck.nspec()
##    print("dict get time", (endtime-starttime)/10**6, "ms") # ~15 times faster
##    
##    starttime = timepck.nspec()
##    indexes2 = [int(bt.get(str(n))) for n in numbers2[:1000]]
##    endtime = timepck.nspec()
##    print("binarytree get time", (endtime-starttime)/10**6, "ms")
##    print(indexes==indexes2)
##    bt.save("btree_test")
    

    
##    m = my_class()
##    mm = my_class2()
##    print(mm.x)
##    print(m.x)
##    print(m.lazy_x)
##    print(m.lazy_x)
##    mm = my_class()
##    print(mm.x)
##    print(mm.x)
    
##    print(get_class("np.uint8"))
    
##    import random
##    l = list(range(10))
##    for f in [selection_sort, bubble_sort, insertion_sort, merge_sort, quick_sort, heap_sort, count_sort]:
##        random.shuffle(l)
##        print(f, l)
##        f(l)
##        print("->", l)
    
##    da = dictarray()
##    x = da.new("asd", 3)
##    da.save("test", ext="save")
##    da = dictarray()
##    da.load("test", ext="save")
##    print(da.get("asd"))


    
##    a = asd()
##    print(get_attr_as_list(a, "a", "__init__"))
##    print(get_attr_as_dict(a, label="a", asd="__init__"))
    
##    a = asd()
##    d = {
##        "INT":3,
##        "LIST":list(range(5)),
##        "SET":set(range(5)),
##        "emptylist": [],
##        "a:sd": np.random.rand(5),
##        }
####    d_ = packup(d)
####    print(d_)
####    print(unpack(d_))
##    
##    a = asd()
##    a.d = d
##    a.a = a
##    a.save("asd")
##    
##    a = asd()
##    a.load("asd")
##    print(a, a.__dict__)
    
##    a = asd()
##    a.b = asd()
##    a.b.a = a
##    a.save("test")
##    print(a)
##    print(a.b)
##    print(a.b.a)
##    print("")
    
##    a = asd()
##    a.load("test")
##    print(a)
##    print(a.b)
##    print(a.b.a)
##    print(a.b.__dict__)
##    print(a.b.a.__dict__)
    
##    e = expirelists()
##    e.new("asd")
##    e.add("asd", "a", 1)
##    e.set("asd", "a", 2)
##    print(e.asd_t)
    
##    print(isinstance(e.new, type(e.add)))
##    print(isfunction(e.add), type(e.add))
    pass
        
