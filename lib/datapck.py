import os
import random
import types
from io import FileIO

import numpy as np
import pickle

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
    for _ in filecopy_gen(*args, **kwargs): pass

def foldercopy_gen(read, write, buffer=4096, fraction=True, report_buffer=None):
    read = ospck.get_folder(read)
    write = ospck.get_folder(write)
    ospck.makedirs(write)
    size = ospck.getsize_folder(read)
    size_running = 0
    if report_buffer is None: report_buffer = buffer
    items = list(ospck.list_folders(read))
    for f in items:
        _, name, ext = ospck.explode(f)
        ff = ospck.implode([write], name)
        if not os.path.exists(ff): os.mkdir(ff)
        for i in foldercopy_gen(f, ff, buffer=buffer, fraction=False, report_buffer=report_buffer*len(items)):
            size_running += i
            if size_running>=report_buffer:
                if fraction: yield size_running/size
                else:
                    yield size_running
                    size_running = 0
        if not fraction and size_running:
            yield size_running
            size_running = 0

    items = list(ospck.list_files(read))
    for f in items:
        _, name, ext = ospck.explode(f)
        ff = ospck.implode([write], name, ext)
        for i in filecopy_gen(f, ff, buffer=buffer, fraction=False, report_buffer=report_buffer*len(items)):
            size_running += i
            if size_running//report_buffer:
                if fraction: yield size_running/size
                else:
                    yield size_running
                    size_running = 0
        if not fraction and size_running:
            yield size_running
            size_running = 0
def foldercopy(*args, **kwargs):
    for _ in foldercopy_gen(*args, **kwargs): pass



def copy_gen(read, *args, **kwargs):
    if os.path.isfile(read):
        for x in filecopy_gen(read, *args, **kwargs): yield x
    else:
        for x in foldercopy_gen(read, *args, **kwargs): yield x
def copy(*args, **kwargs):
    for _ in copy_gen(*args, **kwargs): pass



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
    else: t = 12+a.itemsize//4 # str_
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
##def dictsave(path, d, add=False): # folder
##    if not add: ospck.delete_folder(path)
##    os.makedirs(path, exist_ok=True)
##    for k,v in d.items():
##        t = type(v)
##        p = path+"\\"+openers[type(k)]+str(k)
##        if t==str: strsave(p, v)
##        elif t==type: typesave(p, v)
##        elif t==float: floatsave(p, v)
##        elif t==int: intsave(p, v)
##        elif t==bool: intsave(p, 1 if v else 0)
##        elif t==dict: dictsave(p, v)
##        elif t in [list,tuple]: iterablesave(p, v)
##        elif t==np.ndarray: arraysave(p, v)
##        elif v is None: nonesave(p)
##        else:
##            try: v._save(p)
##            except: nonesave(p)
##def dictload(path):
##    d = {}
##    for x in ospck.list_folders(path):
##        _, name, ext = ospck.explode(x)
##        k = reverse_openers[name[0]](name[1:])
##        d[k] = dictload(x)
##    for x in ospck.list_files(path):
##        _, name, ext = ospck.explode(x)
##        p = path+"\\"+name
##        k = reverse_openers[name[0]](name[1:])
##        if ext=="str": d[k] = strload(p)
##        elif ext=="type": d[k] = typeload(p)
##        elif ext=="float": d[k] = floatload(p)
##        elif ext=="int": d[k] = intload(p)
##        elif ext=="array": d[k] = arrayload(p)
##        elif ext=="iterable": d[k] = iterableload(p)
##        elif ext=="none": d[k] = None
##    return d

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
                t, k, _, v = line.split(" ", 3)
                if "int" in t[1:-1]: d[k] = int(v)
                elif "float" in t[1:-1]: d[k] = float(v)
                elif "str" in t[1:-1]: d[k] = v
                elif "bool" in t[1:-1]: d[k] = (v.lower()=="true" or (v.isnumeric() and bool(int(v))))
                elif "type" in t[1:-1]: d[k] = eval(v)
            except ValueError: pass
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
    if end is None: end = len(l)
    return [i+start for i,xx in enumerate(l[start:end]) if x==xx]
def listsearch_b(x, l, start=0, end=None, wordlen=1): # return indexes of x in l # BYTE FRIENDLY
    if end is None: end = len(l)
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
        if "</" in t:
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
    for xx in char:
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
    if len(options)>amount:
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
    if low is None: low = 0
    if high is None: high = len(l)-1
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





if __name__ == "__main__":
    pass
        
