import os
import sys

def randombin(l, blacklist=""):
    # 10000 calls -> 0.02690509999999996
    while 1:
        x = ''.join(format(x, '08b') for x in os.urandom(max(int(l/8), 1)))[:l+1]
        if not x in blacklist: return x
        
def randomint(l, blacklist=""): # random.randint(0,10**l) is faster
    # 10000 calls -> 0.06438519999999998
    while 1:
        x = str(bytes([48+i//27 for i in os.urandom(l)]), "ascii")
        if not x in blacklist: return x
        
def randomstr(l, blacklist=""): # a-z
    # 10000 calls -> 0.07719710000000002
    while 1:
        x = str(bytes([97+i//10 for i in os.urandom(l)]), "ascii")
        if not x in blacklist: return x
        
def randomhex(l, blacklist=""):
    # 10000 calls -> 0.09409780000000001
    while 1:
        x = ''.join(format(x, 'x') for x in os.urandom(max(int(l*0.6), 1)))[:l+1]
        if not x in blacklist: return x

def strhash(string):
    value = 0
    for i in range(len(string)):
        value += bytes(string[i], "utf8")[0]<<i
    return value


def explode(path):
    dirs = name = ext = None
    path = path.replace("/", os.path.sep).replace("\\", os.path.sep)
    drive, path = os.path.splitdrive(path) # if UNC
    if not drive: drive, path = os.path.splitdrive(path)
    dirs = []
    if path:
        name = os.path.basename(path)
        if "." in name: name, ext = name.rsplit(".", 1)
        dirpath, tail = os.path.split(os.path.dirname(path))
        while tail:
            dirs.append(tail)
            dirpath, tail = os.path.split(dirpath)
    if drive: dirs.append(drive)
    dirs = dirs[::-1]
    if "" in dirs[-1:]: dirs = dirs[:-1]
    return dirs, name, ext

def implode(dirs, name=None, ext=None): return (os.path.sep.join(dirs) if dirs else ".")+str(os.path.sep+name+str("."+ext if ext else "") if name else "")
    
def list_files(path):
    try:
        if os.path.isdir(path):
            for i in os.listdir(path):
                i = os.path.join(path, i)
                if os.path.isfile(i): yield i
    except: pass # denied
def list_folders(path):
    try:
        if os.path.isdir(path):
            for f in os.listdir(path):
                f = os.path.join(path, f)
                if os.path.isdir(f): yield f
    except: pass # denied

def list_files_recur(path):
    for f in list_folders(path):
        for i in list_files_recur(f): yield i
    for i in list_files(path): yield i

def delete_file(path):
    if os.path.isfile(path):
        os.remove(path)
        return True
    return False

def delete_folder(path):
    if os.path.isdir(path):
        for x in list_files(path): delete_file(x)
        for x in list_folders(path): delete_folder(x)
        os.rmdir(path)
        return True
    return False

def getsize(path): return os.path.getsize(path)
def getsize_folder(path):
    size = 0
    for i in list_files_recur(path): size += os.path.getsize(i)
    return size



def basename(path): return os.path.basename(path).rsplit(".",1)[0]



def print_up(n=1): sys.stdout.write("\033[F"*n)

def get_folder(path):
    if "." in os.path.basename(path): path = os.path.dirname(path)
    return path

def parent_folder(path, n=0): return os.path.abspath(path).rsplit(os.path.sep, n+1)[0]


def makedirs(path):
    path = get_folder(path)
    if path: os.makedirs(path, exist_ok=True)



def sort_by_path(l):
    l = [explode(x) for x in l]
    l.sort(key=lambda x:x[2].lower() if x[2] else "") # extension first
    l.sort(key=lambda x:x[1].lower()) # name
    l.sort(key=lambda x:os.path.sep.join(x[0]).lower()) # dirs
    return [implode(*x) for x in l]
def sort_by_size(l):
    l.sort(key=lambda x:getsize_folder(x))
    return l


def folder_dict(path, max_depth=-1, fullpaths=False):
    # key == path
    # value == dict or size
    d = {}
    if os.path.isdir(path):
        for i in list_files(path):
            dirs, name, ext = explode(i)
            d[i if fullpaths else (name if not ext else name+"."+ext)] = os.path.getsize(i)
        if max_depth!=0:
            for f in list_folders(path):
                dirs, name, ext = explode(f)
                d[f if fullpaths else (name if not ext else name+"."+ext)] = folder_dict(f, max_depth-1, fullpaths=fullpaths)
    else:
        dirs, name, ext = explode(path)
        d[path if fullpaths else (name if not ext else name+"."+ext)] = os.path.getsize(path)
    return d

def combine_folder_dicts(d0, d1):
    for k,v in d1.items():
        if k in d0 and type(v)==dict: v = combine_folder_dicts(d0[k], v)
        d0[k] = v
    return d0



if __name__ == "__main__":
    pass
    
