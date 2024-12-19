import math
import numpy as np


def is_a_valid_point(x):
    if hasattr(x, "__len__"):
        try:
            np.linalg.norm(x)
            return len(x)
        except ValueError: pass

def is_array(x): return type(x)==np.ndarray
def is_str(x, array=True): return type(x) in [str,np.str_] or (array and is_array(x) and str(x.dtype)[:2]=="<U")
def is_float(x, array=True): return type(x)==float or isinstance(x, np.floating) or (array and is_array(x) and is_float(getattr(np, str(x.dtype))(1)))
def is_integer(x, array=True): return type(x)==int or isinstance(x, np.integer) or (array and is_array(x) and is_integer(getattr(np, str(x.dtype))(1)))
def is_iterable(x): return type(x) in [tuple,set,list,np.ndarray]
def is_mutable(x): return type(x) in [list,np.ndarray,dict]

def acenter(shape): return np.int_(np.divide(np.subtract(shape, 1), 2))

def around(a):
    t = type(a)
    if is_iterable(a): a = np.array(a)
    if not is_array(a): return round(a)
    a = np.int_(a)+np.sign(a)*(np.mod(np.abs(a),1)>=0.5)
    a = a.astype(np.int_)
    if t!=np.ndarray: return t(a)
    return a

def make_even(a):
    t = type(a)
    if t!=np.ndarray: a = np.array(a)
    valid = a%2!=0
    a[valid] += np.sign(a[valid])
    if t!=np.ndarray: return t(a)
    return a


def amax(a, i=0, arg=False): # i-th max value/index
    a = np.array(a)
    if i<0: return amin(a, -i-1, arg=arg)
    if i>a.size//2: return amin(a, a.size-i-1, arg=arg)
    indexes = np.zeros(0, dtype=np.int64)
    values = np.zeros(0, dtype=a.dtype)
    for ii in range(i):
        ii = a.argmax()
        indexes = np.append(indexes, ii)
        values = np.append(values, a.reshape(-1)[ii])
        a.reshape(-1)[ii] = a.min()
    value = a.argmax() if arg else a.max()
    a.reshape(-1)[indexes] = values[:]
    return value
def amin(a, i=0, arg=False): # i-th min value/index
    a = np.array(a)
    if i<0: return amax(a, -i-1, arg=arg)
    if i>a.size//2: return amax(a, a.size-i-1, arg=arg)
    indexes = np.zeros(0, dtype=np.int64)
    values = np.zeros(0, dtype=a.dtype)
    for ii in range(i):
        ii = a.argmin()
        indexes = np.append(indexes, ii)
        values = np.append(values, a.reshape(-1)[ii])
        a.reshape(-1)[ii] = a.max()
    value = a.argmin() if arg else a.min()
    a.reshape(-1)[indexes] = values[:]
    return value



def intersects(start0,end0,start1,end1):
    def ccw(a,b,c): return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    return ccw(start0,start1,end1)!=ccw(end0,start1,end1) and ccw(start0,end0,start1)!=ccw(start0,end0,end1)
def intersects_bulk(line, *lines):
    def ccw(a,b,c): return (c[:,1]-a[:,1])*(b[:,0]-a[:,0]) > (b[:,1]-a[:,1])*(c[:,0]-a[:,0])
    lines = np.array(lines)
    line = np.expand_dims(line, axis=0)
    a = ccw(line[:,0],lines[:,0],lines[:,1])
    b = ccw(line[:,1],lines[:,0],lines[:,1])
    c = ccw(line[:,0],line[:,1],lines[:,0])
    d = ccw(line[:,0],line[:,1],lines[:,1])
    return np.logical_and(a!=b, c!=d)


def intersection(start0,end0,start1,end1):
    def det(a, b): return a[0]*b[1] - a[1] * b[0]
    xdiff = (start0[0]-end0[0], start1[0]-end1[0])
    ydiff = (start0[1]-end0[1], start1[1]-end1[1])
    if (div:=det(xdiff, ydiff))!=0:
        d = (det(start0, end0), det(start1,end1))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y
def intersection_bulk(line, *lines, return_valid=False):
    def det(a, b): return a[0]*b[1]-a[1]*b[0]
    lines = np.array(lines)
    xdiff0 = line[0][0]-line[1][0]
    xdiff1 = lines[:,0,0]-lines[:,1,0]
    ydiff0 = line[0][1]-line[1][1]
    ydiff1 = lines[:,0,1]-lines[:,1,1]
    div = xdiff0*ydiff1-xdiff1*ydiff0
    valid = div!=0
    d0 = det(*line)
    d1 = lines[valid][:,0,0]*lines[valid][:,1,1]-lines[valid][:,0,1]*lines[valid][:,1,0]
    x = d0*xdiff1-d1*xdiff0
    y = d0*ydiff1-d1*ydiff0
    x = x[valid]/div[valid]
    y = y[valid]/div[valid]
    result = np.concatenate([np.expand_dims(x, axis=1), np.expand_dims(y, axis=1)], axis=1)
    if return_valid: return result, valid
    return result



def distance(a, b): return np.linalg.norm(np.subtract(a, b), axis=-1) # n-dimensional point to point distance

def nearestvalue(array, value): return array[np.argmin(np.abs(np.asarray(array)-value))]
def nearestvalue_i(array, value): return int(np.argmin(np.abs(np.asarray(array)-value)))

def furthestpoint(a, *b): return b[np.argmax([distance(a, x) for x in b])]
def furthestpoint_i(a, *b): return np.argmax([distance(a, x) for x in b])

def nearestpoint(a, *b): return b[np.argmin([distance(a, x) for x in b])]
def nearestpoint_i(a, *b): return np.argmin([distance(a, x) for x in b])


def scale(a, return_range=False):
    if (max_val:=np.abs(a).max())>0: a = a/max_val
    else: a = np.zeros_like(a)
    if return_range: return a, (np.abs(a).min(), max_val)
    return a
def scale_bulk(*arrays, return_range=False):
    max_val = 0
    min_val = 0
    for a in arrays:
        max_val = max(np.abs(a).max(), max_val)
        min_val = min(np.abs(a).min(), min_val)
    arrays = list(map(lambda x: x/max_val if max_val>0 else np.zeros_like(x), arrays))
    if return_range: return arrays, (min_val, max_val)
    return arrays

def rescale(scaledarray, a_min=0, a_max=1): return (scaledarray*(a_max-a_min))+a_min


def snd(a): return (np.e**(-(a**2)/2))/((2*np.pi)**0.5)  # standard normal distribution # µ==0, var==1
def weights(values): return np.unique(values, return_counts=True)[1]
def rng(chances, seed=None):
    if is_integer(chances, False) or is_float(chances, False): chances = [1,chances]
    a = np.cumsum(chances)
    if seed is None: f = np.random.random(1)
    elif is_integer(seed): f = np.random.default_rng(seed).random(1)
    else: f = seed.random(1)
    return (a<(f*a[-1])[0]).sum()
def rng_bulk(chances, n=1, count=False, seed=None):
    if is_integer(chances, False) or is_float(chances, False): chances = [1,chances]
    a = np.expand_dims(np.cumsum(chances), axis=0)
    if seed is None: f = np.random.random((n,1))
    elif is_integer(seed): f = np.random.default_rng(seed).random((n,1))
    else: f = seed.random((n,1))
    b = f*a.max()
    c = np.sum((a>b).astype(np.int64), axis=1)
    if c.any(): c = len(chances)-c
    if count:
        observedchances = np.zeros(a.size)
        for i in range(a.size): observedchances[i] += np.sum(c==i)
        return c, observedchances
    return c


def deviant(value, deviancy=.5, seed=None):
    if deviancy!=0:
        if seed is None: f = np.random.random()
        elif is_integer(seed): f = np.random.default_rng(seed).random()
        else: f = seed.random()
        return value+value*deviancy*(f-.5)*2
    return value
def deviants(value, deviancy=.5, n=1, seed=None):
    n = max(int(abs(n)), 1)
    if deviancy!=0:
        if seed is None: f = np.random.random(n)
        elif is_integer(seed): f = np.random.default_rng(seed).random(n)
        else: f = seed.random(n)
        return value+value*deviancy*(f-.5)*2
    return np.ones(amount)*value


def floatrng(a, acc=2): return rng([int((1-a)*10**acc),int(a*10**acc)])
def dictrng(d, amount=1): # treat a dict of floats/integers as a chance list
    return (np.array(list(d.keys()))[rng_bulk(list(d.values()), amount)]).tolist()


def rngseries(series, fails=0): # floatrng every float in list/tuple else passthrough
    passed = []
    for x in series:
        if is_float(x):
            if not floatrng(x):
                if fails==0: break
                else: fails -= 1
        else: passed.append(x)
    return passed

def peakrng(flatness, steps=101):
    # flatness == 1 -> average == 0.5
    # flatness == 2 -> average == 0.66..
    # flatness == 3 -> average == 0.72..
    # flatness == 10 -> average == 0.85..
    # flatness == 100 -> average == 0.95..
    a = np.random.randint(1, steps+1, flatness).sum()
    return 1-abs((a-flatness)/(steps*flatness-flatness)-.5)*2 # 0...1

def findr(points, space, array): # translate points in 2d space (upleft)->(bottomright) into elements in a 2D array
    indexes = np.multiply(np.divide(points, np.expand_dims(space, axis=0)), np.expand_dims(array.shape, axis=0))
    indexes = np.clip(indexes.astype(np.int64), a_min=0, a_max=np.subtract(array.shape, 1))
    return array[*indexes.swapaxes(0, 1)]


#
def preferarray(v1, v2): # 1D
    if type(v1) in [list,tuple]: v1 = np.asarray(v1)
    if type(v2) in [list,tuple]: v2 = np.asarray(v2)
    if type(v1) == np.ndarray:
        if type(v2) in [int,float]: v2 = np.asarray([v2])
        if v1.shape != v2.shape and v1.size>v2.size:
            v2 = exptoshape(v2, v1.shape)
    if type(v2) == np.ndarray:
        if type(v1) in [int,float]: v1 = np.asarray([v1])
        if v2.shape != v1.shape and v2.size>v1.size:
            v1 = exptoshape(v1, v2.shape)
    return v1, v2

def exptoshape(a, newshape, sides=[2]): # sides == iterable of 0/1/2 == start/end/both(end first)
    if len(togglelist:=list(sides)) < a.ndim:
        for i in range(a.ndim-len(togglelist)): togglelist.append(2)
    if not len(newshape) == a.ndim: return a
    for i in range(a.ndim):
        if togglelist[i] == 2: toggle = True
        else: toggle = False
        amount = newshape[i]-a.shape[i]
        if amount > 0: # add more lines/rows/colums/...
            for ii in range(amount):
                l = [1 for iii in range(a.shape[i]-1)]
                if togglelist[i]:
                    l.append(2)
                    if toggle: togglelist[i] = 0
                else:
                    l.insert(0, 2)
                    if toggle: togglelist[i] = 1
                a = np.repeat(a, l, axis=i)
        elif amount < 0:
            for ii in range(abs(amount)):
                if togglelist[i]:
                    a = np.delete(a, -1, axis=i)
                    if toggle: togglelist[i] = 0
                else:
                    a = np.delete(a, 0, axis=i)
                    if toggle: togglelist[i] = 1
    return a
    

def gaussianredistribution(valuelist): # suffles the list according to item commodity & variates the length
    valuelist = list(totuple(valuelist, -1) if type(valuelist)==np.ndarray else valuelist)
    w = list(map(lambda x: valuelist.count(x), valuelist)) # valuelist.count(x)
    r = round(max(2, len(valuelist)/2))
    return [valuelist[rng(w)] for _ in range(len(valuelist)-r+rng((np.hanning(r)*r).astype(np.uint64)+1))]

def weightshift(a, i, power): # ([0.5, 1 , 2.4], 1, 2) -> [0.25, 2 , 1.2]
    a = np.array(a, dtype=float)
    ma = np.ma.array(a, mask=False)
    ma.mask[i] = True
    if power > 1:
        ma /= power
        a[i] *= power
    elif 0 < power < 1: a[i] *= power # reduce the index only
    elif power <= 0: a **= 0.5
    return a



def squarearray(dist=3, hor=True, ver=True, diag=True):
    if dist%2==0: dist += 1
    dist = max(dist, 3)
    s = np.ones((dist,dist), dtype=int)
    m = int(dist/2)
    if not hor:
        s[m,:m] = 0
        s[m,m+1:] = 0
    if not ver:
        s[:m,m] = 0
        s[m+1:,m] = 0
    if not diag:
        s[:m,:m] = 0
        s[:m,m+1:] = 0
        s[m+1:,:m] = 0
        s[m+1:,m+1:] = 0
    return s


# 
def getsubarea(array, i, r=2):
    i = np.mod(conformarray(i, array.ndim), array.shape)
    r = conformarray(totuple(r), array.ndim, r)
    imin = np.clip(np.add(i, -r), a_min=0, a_max=array.shape)
    imax = np.clip(np.add(i, r+1), a_min=0, a_max=array.shape)
    return tuple([slice(imin[x],imax[x]) for x in range(array.ndim)]), i, r


def getsurroundings(array, i, r=2, pad=True):
    i = np.array(i)
    ashape = np.array(array.shape)
    i = np.mod(i, ashape[:len(i)])
    r = conformarray(totuple(r), len(i), r)
    imin = np.clip(np.add(i, -r), a_min=0, a_max=ashape[:len(i)])
    imax = np.clip(np.add(i, r+1), a_min=0, a_max=ashape[:len(i)])
    sub = tuple([slice(imin[x],imax[x]) for x in range(len(i))])
    a = array[sub]
    if pad:
        asdf = np.concatenate([np.expand_dims(np.clip(np.add(-i, r), a_min=0, a_max=r), -1), np.expand_dims(np.clip(i+r+1-ashape[:len(i)], a_min=0, a_max=r), -1)], axis=-1)
        return np.pad(a, np.pad(asdf, ((0,array.ndim-len(i)),(0,0))))
    return a


def mesh(radius:int):
    x = np.append(np.linspace(0,1,radius+1), np.linspace(1,0,radius+1)[1:])
    return np.multiply(*np.meshgrid(x, x))

### ANGLES
def heading_absolute(x): # -inf...inf -> 0...360
    if is_array(x):
        x = np.mod(x, 360)
        x[x<0] = x[x<0]+360
    else: x = x%360
    return x
def heading_normal(x): # -inf...inf -> -180...180
    if is_array(x):
        x = np.mod(x, 360)
        x[x>180] = x[x>180]-360
        x[x<=-180] = x[x<=-180]+360
    else:
        x = x%360
        if x>180: x -= 360
    return x

def heading_offset(h):
    if is_array(h):
        A = np.radians(h)
        o = np.concatenate([np.expand_dims(np.sin(A), axis=1), np.expand_dims(np.cos(A), axis=1)], axis=1)
    else:
        A = math.radians(h)
        o = (math.sin(A), math.cos(A))
    return o
def offset_heading(x):
    if is_array(x) and x.ndim==2:
        A = np.arctan2(x[:,0], x[:,1])
        h = np.degrees(A)
    else:
        A = math.atan2(x[0], x[1])
        h = math.degrees(A)
    return heading_absolute(h)


def cardinal_heading(x):
    x = x.lower()
    if x in ["north","n"]: return 0
    if x in ["east","e"]: return 90
    if x in ["south","s"]: return 180
    if x in ["west","w"]: return 270
    if x in ["northwest","nw"]: return 315
    if x in ["northeast","ne"]: return 45
    if x in ["southeast","se"]: return 135
    if x in ["southwest","sw"]: return 225
def heading_cardinal(h):
    i = nearestvalue_i([0,90,180,270,315,45,135,225,360], heading_absolute(h))
    return ["north","east","south","west","northwest","northeast","southeast","southwest","north"][i]




def odh(x, y): # offset, distance, heading (positive North & positive East)
    o = np.subtract(y, x)
    d = np.linalg.norm(o, axis=-1)
    try: h = offset_heading(o)
    except ValueError: h = None
    return o, d, h



def conformarray(a, shape, value=0, inverse=False):
    a = np.array(a)
    na0 = np.reshape(np.subtract(shape, a.shape), (-1,1))
    na0[na0<0] = 0
    return np.pad(a, np.concatenate((np.zeros(na0.shape), na0), axis=1).astype(np.int32), constant_values=value)
def conformarrays(listofarrays, fill=0):
    shape = np.ones(1)
    for x in listofarrays: shape = np.maximum(shape, np.array(x).shape)
    return list(map(lambda x:conformarray(x, shape, fill), listofarrays)), tuple(shape.astype(np.int16))


def totuple(a, ndim=1):
    if ndim:
        if type(a) == np.ndarray:
            ndim = a.ndim
            a = a.tolist()
        if type(a) == list: a = map(lambda x: totuple(x, ndim-1) if type(x)==list else x, a)
        elif type(a) != tuple: return (a,)
        return tuple(a)
    return a







def crossndenumerate(a, odd=False):
    return [(idx,x) for idx,x in np.ndenumerate(a) if (sum(idx)+1 if odd else sum(idx))%2==0]

def overlap(anchor_a, anchor_b, array_a, array_b, return_values=False): # overlap_slices
    # get overlapping slices/values from two arrays
    overlap_left = np.minimum(anchor_a, anchor_b)
    overlap_right = np.minimum(np.subtract(array_a.shape, anchor_a), np.subtract(array_b.shape, anchor_b))
    array_a_slices = tuple([slice(anchor_a[x]-overlap_left[x],anchor_a[x]+overlap_right[x]) for x in range(array_a.ndim)])
    array_b_slices = tuple([slice(anchor_b[x]-overlap_left[x],anchor_b[x]+overlap_right[x]) for x in range(array_b.ndim)])
    if return_values:
        preferredshape = np.minimum(array_a.shape, array_b.shape)
        i = acenter(preferredshape)
        overlap_left = np.add(i, -overlap_left)
        overlap_right = np.add(i, -overlap_right+(preferredshape%2==0).astype(np.int64)+1)
        overlap_left[overlap_left<0] = 0
        overlap_right[overlap_right<0] = 0
        pad = np.concatenate([overlap_left.reshape(-1,1), overlap_right.reshape(-1,1)], axis=1)
        return np.pad(array_a[*array_a_slices], pad), np.pad(array_b[*array_b_slices], pad)
    return array_a_slices, array_b_slices

def shapemultipliers(shape0, shape1, sigma=0.5):
    z = np.divide(shape0, shape1)
    b = (z**(-1))**sigma-1 # 
    b[2:] = 0
    b[z>=1] = 0
    return z, b













uint16max = np.iinfo(np.uint16).max
class keyhierarchy(): # store links between keys == gives best alternatives to a key
    def __init__(self):
        self.words = {"_": 0} # word : index
        self.kwords = np.asarray(["_"], dtype=np.str_)
        self.powers = np.zeros((1,1)).astype(np.uint16) # positive 2D array; key link power to other keys

    def reset(self): self.powers *= 0

    def inject(self, x):
        if not x in self.words:
            self.words[x] = self.powers.shape[0]
            self.kwords = np.concatenate([self.kwords, [x]], axis=0)
            self.powers = np.pad(self.powers, ((0, 1), (0, 1)), constant_values=0)

    def reinforce(self, x, y, r=1, t=1):
        if x in self.words and y in self.words: self.powers[(self.words[x], self.words[y])] += r
            
    def add(self, source, subjects, r=1.):
        r = max((self.powers.shape[0]**(1/2)/self.powers.shape[0])*r, 0.001)
        if type(subjects) == list: # multiple subjects
            if source is not None: self.inject(source)
            for x in subjects: self.inject(x)
        elif source is not None and type(subjects) == str: # one source, one subject
            self.inject(source)
            self.inject(subjects)
            
        self.powers = scale(self.powers)
        
        if type(subjects) == list: # multiple subjects
            if source is not None:
                for x in subjects: self.reinforce(x, source, r=r)
            for x in subjects:
                for y in subjects:
                    if x!=y: self.reinforce(x, y, r=r/2)
            if source is not None:
                for x in subjects: self.reinforce(source, x, r=r/4)
        elif source is not None and type(subjects) == str: # one source, one subject
            self.reinforce(subjects, source, r=r)
            self.reinforce(source, subjects, r=r/4)
        self.powers = rescale(scale(self.powers), a_min=0, a_max=uint16max).astype(np.uint16)
            
    def get(self, x, d=0):
        if x in self.words:
            l = [(self.kwords[idx], xx) for idx, xx in list(np.ndenumerate(self.powers[self.words[x]])) if xx>0]
            l = [(k,p,(np.int32(self.powers[self.words[x]][self.words[k]])-np.int32(self.powers[self.words[k]][self.words[x]]))/np.int32(uint16max)) for k,p in l]
            if d>0:
                K, A, I = zip(*l)
                K = list(K)
                A = list(A)
                I = list(I)
                for ii in range(len(l)):
                    for k, a, i in self.get(l[ii][0], d-1):
                        if x != k:
                            if not k in K:
                                K.append(k)
                                A.append(int(a*(l[ii][1]/uint16max)))
                                I.append([i])
                            else:
                                k_index = K.index(k)
                                A[k_index] += a*(l[ii][1]/uint16max)
                                if type(I[k_index])==int: I[k_index] = [I[k_index], i]
                                else: I[k_index].append(i)
                for i,x in enumerate(I):
                    if type(x)==list: I[i] = sum(x)/len(x)
                l = list(zip(K, A, I))
                l.sort(reverse=True, key=lambda x:x[1])
            return l

    def multiget(self, keywords, d=0): # get common
        K = []
        A = []
        I = []
        for x in keywords:
            for k,a,i in self.get(x, d): # list(zip(*))
                if not k in K:
                    K.append(k)
                    A.append(int(a))
                    I.append([i])
                else:
                    k_index = K.index(k)
                    A[k_index] += a
                    if type(I[k_index])==int: I[k_index] = [I[k_index], i]
                    else: I[k_index].append(i)

##        I = map(lambda x: sum(x)/len(x) if type(x)==list else x, I) # untested
        for i,x in enumerate(I):
            if type(x)==list: I[i] = sum(x)/len(x)
        l = list(zip(K, A, I))
        l.sort(reverse=True, key=lambda x:x[1])
        return l




def is_prime(x): # check for prime integers
    if is_array(x):
        x = np.int_(x)
        notvalid = np.logical_or(x<=1, np.logical_or(np.mod(x, 2)==0,np.mod(x, 3)==0))
        i = 5
        while not (i**2>x[~notvalid]).all():
            notvalid[~notvalid] = np.logical_or(x[~notvalid]%i==0, x[~notvalid]%(i+2)==0)
            i += 6
        return np.logical_or(np.logical_or(~notvalid, x==2), x==3)
    x = int(x)
    if x<=3: return x>1
    elif x%2==0 or x%3==0: return False
    i = 5
    while i**2<=x:
        if x%i==0 or x%(i+2)==0: return False
        i += 6
    return True

def prev_prime(start):
    if 2>=start: return 2
    start -= (1-start%2)
    i = 0
    while not is_prime(start+i): i -= 2
    return start+i

def next_prime(start):
    if 2>=start: return 2
    start += (1-start%2)
    i = 0
    while not is_prime(start+i): i += 2
    return start+i

def nearest_prime(start):
    if 2>=start: return 2
    start += (1-start%2)
    i = 0
    while 1:
        if is_prime(start-i): return start-i
        i += 2
        if is_prime(start+i): return start+i

def find_primes(low:int, high:int=0, step:int=1, limit:int=0) -> int:
    low += (1-low%2)
    if 2>=low: low = 2
    i = 0
    count = 0
    while (high<1 or (low+i)<=high) and (limit<1 or limit>count):
        if is_prime(low+i):
            if not count%step: yield low+i
            count += 1
        i += 2





def power(a, e):
    # infinity == 0
    # negative values handled like positives
    if is_array(a):
        zeros = a!=0
        if is_array(e) and e.shape[0]==a.shape[0]: e = e[zeros]
        a[zeros] = np.sign(a[zeros])*np.abs(a[zeros])**e
    elif a!=0: a = np.sign(a)*abs(a)**e
    return a
def sigmoid(x): # max == 36
    return 1/(1+np.exp(-x))
def isigmoid(x):
    if x<1: return -np.log((1/x)-1)
    else: return 37

def loss(a, b): return np.average(np.diff([a, b], axis=0))





####
def odd_cycle(x, y):
    return int(x/(y/2))%2
def odd_cycles(x, y):
    return np.mod((x/(y/2)).astype(np.int8), 2)

def linear_cycle(x, y, p=1): # 0...1...0...-1...0|0..
    if y!=0:
        y = y/4
        out = (x%y)/y
        a = odd_cycle(x, y*2)
        out = out*-(a*2-1)+a
        out = out*-(odd_cycle(x, y*4)*2-1)
        return power(out, p)
    return 0
def linear_cycles(x, y, p=1): # linear_cycles
    x = np.array(x)
    if is_array(y) or y!=0:
        y = y/4
        out = np.mod(x, y)/y
        a = odd_cycles(x, y*2)
        out = out*-(a*2-1)+a
        out = out*-(odd_cycles(x, y*4)*2-1)
        return power(out, p)
    return np.zeros(x.shape)

def sin_cycle(x, y, p=1): # 0...1...0...-1...0|
    if y!=0: return math.sin(math.pi*2*(x/y))**p
    return 0
def sin_cycles(x, y, p=1): # arraysin
    x = np.array(x)
    if is_array(y) or y!=0: return power(np.sin(np.pi*2*(x/y)), p)
    return np.zeros(x.shape)

def cos_cycle(x, y, p=1): # 1...0...-1...0...1|
    if y!=0: return math.cos(math.pi*2*(x/y))**p
    return 0
def cos_cycles(x, y, p=1):
    x = np.array(x)
    if is_array(y) or y!=0: return power(np.cos(np.pi*2*(x/y)), p)
    return np.zeros(x.shape)

def tan_cycle(x, y, p=1): # 0...inf-inf...0|0...inf-inf...0|
    if y!=0: return math.tan(math.pi*2*(x/y))**p
    return 0
def tan_cycles(x, y, p=1):
    x = np.array(x)
    if is_array(y) or y!=0:
        out = np.tan(np.pi*2*(x/y))
        return power(out, p)
    return np.zeros(x.shape)
####



def phobia(xa, scare, r): # calculate force(-0..-1 or 1..0) from the repelling 'scare' value
##    if abs(scare-x)<=r:
##        if scare == x: return 1.
##        a = (scare-x)/r
##        a = (1+a) if scare<x else (1-a)
##        return a if scare<x else -a
##    return 0.
    inrange = np.absolute(scare-xa)<=r
    ones = xa==scare
    valid = np.logical_and(inrange, ~ones)
    a = np.zeros(xa.shape)
    a[valid] = (scare-xa[valid])/r
    validpositive = np.logical_and(valid, scare<xa)
    a[validpositive] = (1+a[validpositive])
    validnegative = np.logical_and(valid, scare>xa)
    a[validnegative] = (1-a[validnegative])
    a[validnegative] *= -1
    return a


def elementroom(a): # amount of space per element
    dist = np.zeros(a.shape)
    pada = np.pad(a, (1,1), mode="edge")
    for idx,x in np.ndenumerate(pada):
        if 0<idx[0]<a.shape[0]+1: dist[idx[0]-1] = pada[idx[0]+1]-pada[idx[0]-1]
    return dist
def valuephobia(a, values):
    valueroom = elementroom(values)
    values = np.concatenate([values.reshape(-1, 1), valueroom.reshape(-1, 1)], axis=1)
    totalphobia = np.zeros(a.shape)
    for scare,r in values: totalphobia += phobia(a, scare, r)
    return totalphobia





def cluster(a, n=3): # pair and mean values until only n+1 left
    def asd(a):
        dist = np.zeros(a.shape)
        pada = np.pad(a, (1,1), mode="edge")
        for idx,x in np.ndenumerate(pada):
            if 0<idx[0]<=a.shape[0]: dist[idx[0]-1] = pada[idx[0]+1]-pada[idx[0]]
        return dist[:-1]
    b = a.copy()
    while len(dist:=asd(b))>n:
        i = dist.argmin()
        b[i+1] = np.mean(b[i:i+2])
        b = np.concatenate([b[:i], b[i+1:]], axis=0)
    return b




def construct(d, parts):
    return np.sum(np.multiply(d, np.expand_dims(parts, 0)), axis=-1)
def destruct(x, parts): # turn an array/integer to smaller parts
    partsarray = np.asarray(parts)
    if is_array(x):
        x = x.copy()
        d = np.zeros((partsarray.size, x.size))
    else: d = np.zeros(partsarray.size)
    for idx,xx in np.ndenumerate(parts):
        xxx = np.int_(x/xx)
        d[idx] = xxx
        x = x-xx*xxx
    return (np.swapaxes(d, 0, 1) if d.ndim>1 else d).astype(np.uint64)





def spacerng(a, n=10): # place a fitting value in the largest gap n-times # fill "missing" values
    olen = len(a)
    while len(a)<(olen+n):
        i = np.argmax(np.diff(a))
        a = np.insert(a, i+1, np.random.uniform(a[i], a[i+1])) # .astype(a.dtype)
    return a[1:-1]







def diffblur(a):
    b = np.zeros(a.shape)
    c = np.ones(a.shape)*a.ndim*2
    axisslices = [slice(0, a.shape[i]) for i in range(a.ndim)]
    for i in range(a.ndim):
        slices = axisslices.copy()
        for ii in range(a.shape[i]):
            slices[i] = ii
            c[tuple(slices)][0] -= 1
            c[tuple(slices)][-1] -= 1
    for i in range(a.ndim):
        slices = axisslices.copy()
        for ii in range(a.shape[i]-1):
            slices[i] = ii
            aa0 = a[tuple(slices)]
            slices[i] = ii+1
            aa1 = a[tuple(slices)]
            
            slices[i] = ii
            bb = b[tuple(slices)]
            bb += ((aa0+aa1)/c[tuple(slices)])/2
            
            slices[i] = ii+1
            bb = b[tuple(slices)]
            bb += ((aa0+aa1)/c[tuple(slices)])/2
    return (b).astype(a.dtype)



def bestmatch(a, b, value=True): # where loss is smallest for (a in b) # 1D
    if a.ndim!=1 or b.ndim!=1 or a.size>b.size: return b
    i = 0
    minval = min(a.min(), b.min())
    maxval = max(a.max(), b.max())
    loss = np.abs(maxval*a.size-minval*a.size) # max loss that can happen
    result_i = 0
    while b[i:i+a.size].size==a.size:
        l = np.sum(np.abs(a-b[i:i+a.size]))
        if loss>l: result_i, loss = i, l
        i += 1
    if value: return b[result_i:result_i+a.size] # -> return value
    return slice(result_i, result_i+a.size) # -> return slice

def matchloss(u, i): # arrays
    und, ind = abs(max(u.ndim, i.ndim)), abs(min(u.ndim, i.ndim))
    ush, ish = np.abs(np.maximum(u.shape[:ind], i.shape[:ind])), np.abs(np.minimum(u.shape[:ind], i.shape[:ind]))

    while u.ndim!=i.ndim:
        if u.ndim>ind: u = np.mean(u, axis=ind) # ind
        if i.ndim>ind: i = np.mean(i, axis=ind)
        
    ndim_loss = np.abs(und-ind) # dimension mismatch == times np.mean() was used
    shape_loss = np.sum(np.abs(ush-ish)) # shape mismatch == amount of endvalues ignored
    
    ish = np.pad(np.expand_dims(ish, axis=1), ((0,0),(1,0))) # [:ind]
    ish = tuple([slice(*d) for d in ish])
    value_loss = np.sum(np.abs(u[ish]-i[ish])) # absolute value mismatch
    return ndim_loss, shape_loss, value_loss








def heading_as_index(size, h):
    i_size = np.subtract(size, 1)
    i_size_min = i_size.argmin()
    s = i_size[i_size_min]
    delta = max(i_size)-s
    h += 45
    nh = heading_normal(h)
    asd = (heading_absolute(h)%90)/90
    if 0<=nh<90: a, b = 0, asd # 0
    elif 90<=nh<180: a, b = asd, 1 # 1
    elif -180<=nh<-90 or nh==180: a, b = 1, 1-asd # 0
    elif -90<=nh<0: a, b = 1-asd, 0 # 1
    a = a*(s+delta if i_size_min else s)
    b = b*(s+delta if not i_size_min else s)
    return int(a), int(b)


def mask_line_2(start, end):
    o, d, h = odh(start, end)
    return mask_line(np.clip(np.abs(o), a_min=1, a_max=None), h+90)

def arrayline(size, h):
    return heading_as_index(size, h), heading_as_index(size, 180+h)


def mask_line(size, h):
    mask = np.zeros(size).astype(np.uint8)
    start, end = arrayline(size, h)
    max_len = max(size)
    x = around(np.linspace(start[0], end[0], max_len)).astype(np.uint64)
    y = around(np.linspace(start[1], end[1], max_len)).astype(np.uint64)
    if mask.size: mask[x,y] = 1
    return mask.astype(np.bool_)


def mask_split(size, h, ratio=0.5):
    h += 180
    mask = mask_line(size, h).astype(np.int8)
    if (np.max(mask, axis=1)==0).any(): # use top side
        argmax = np.argmax(mask, axis=0)
        for i in range(size[1]):
            m = mask[argmax[i]:,i]
            m[m==0] = 2
    else: # use left side
        argmax = np.argmax(mask, axis=1)
        for i in range(size[0]):
            m = mask[i,:argmax[i]]
            m[m==0] = 2
    i = heading_as_index(size, 90+h) # check that this side is filled
    if not mask[i]:
        zr = np.logical_or(mask==1, mask==2)
        mask[mask==0] = 1
        mask[zr] = 0
    else:
        mask[mask==1] = 0 # hide the line
        mask[mask==2] = 1
    # move rows to achieve wanted ratio
    if i[1]==0:# filled side is left
        rows_to_move = round((ratio-0.5)*size[1])
        if rows_to_move>0:
            mask = np.pad(mask[:,:-rows_to_move], ((0,0),(rows_to_move,0)), constant_values=1)
        elif rows_to_move<0:
            mask = np.pad(mask[:,-rows_to_move:], ((0,0),(0,-rows_to_move)), constant_values=0)
    elif i[1]==size[1]-1: # filled side is right
        rows_to_move = round((ratio-0.5)*size[1])
        if rows_to_move>0:
            mask = np.pad(mask[:,rows_to_move:], ((0,0),(0,rows_to_move)), constant_values=1)
        elif rows_to_move<0:
            mask = np.pad(mask[:,:rows_to_move], ((0,0),(-rows_to_move,0)), constant_values=0)
    elif i[0]==0: # filled side is top
        rows_to_move = round((ratio-0.5)*size[0])
        if rows_to_move>0:
            mask = np.pad(mask[:-rows_to_move], ((rows_to_move,0),(0,0)), constant_values=1)
        elif rows_to_move<0:
            mask = np.pad(mask[-rows_to_move:], ((0,-rows_to_move),(0,0)), constant_values=0)
    elif i[0]==size[0]-1: # filled side is bottom
        rows_to_move = round((ratio-0.5)*size[0])
        if rows_to_move>0:
            mask = np.pad(mask[rows_to_move:], ((0,rows_to_move),(0,0)), constant_values=1)
        elif rows_to_move<0:
            mask = np.pad(mask[:rows_to_move], ((-rows_to_move,0),(0,0)), constant_values=0)
    return mask.astype(np.bool_)












    


def centermass(a): # index for a multidimensional array's center(ish) value
    s = np.ones(a.shape)
    def asd(x): return sin_cycles(np.linspace(0, 1, x), 2, 2)
    ss = [asd(x) for x in a.shape]
    for i,x in enumerate(ss):
        for ii in range(i): x = np.expand_dims(x, axis=0)
        for ii in range(len(ss)-1-i): x = np.expand_dims(x, axis=-1)
        s = s*x
    a = a*(s+1)
    start = np.zeros(a.ndim, dtype=np.int64)
    end = np.array(a.shape, dtype=np.int64)
    mid = np.multiply(end, 0.2).astype(np.int64)
    sections_n = 2**a.ndim
    while 1:
        slices = []
        sections = []
        for ii in range(sections_n):
            asd = 2**np.arange(a.ndim)
            binary = ii/asd
            binary[1:] = np.mod(binary[1:], asd[1:])
            binary[:1] = np.mod(binary[:1], 2)
            b = binary.astype(np.int16)
            s = [(start[iii]+(mid[iii] if not b[iii] else 0), end[iii]-(mid[iii] if b[iii] else 0)) for iii in range(a.ndim)]
            slices.append(s)
            sections.append(np.sum(a[*[slice(*x) for x in s]]))

        s = slices[np.argmax(sections)]
        aa = a[*[slice(*x) for x in s]]
        s = list(zip(*s))
        index = s[0]
        if aa.size<=1: break
        start = s[0]
        end = s[1]
        mid = np.subtract(end, start)*0.2
        mid[mid%1>0] += 1
        mid = mid.astype(np.int64)
    return index





def arc(offset, degrees):
    radius = arc_radius(degrees)
    h = offset_heading(offset)
    offset = np.multiply(heading_offset(h-degrees/2+90), radius)
    return radius, offset
def arc_radius(degrees):
    if degrees%360: return 1/np.sin(np.radians(degrees/2))
    return np.inf







def trimvalues(a, edgevalue=0):
##    print(a, edgevalue)
    if a.ndim==1: a = np.expand_dims(a, axis=0)

    if not (is_float(edgevalue) or is_integer(edgevalue)):
        edgevalue = np.array(edgevalue)
        dims = a.ndim-edgevalue.ndim
    else: dims = a.ndim
    slicebase = [slice(0,a.shape[d]) for d in range(dims)]
    trims = []
    for d in range(dims):
        trims.append([0, 0]) # amount of line trimmed per dimension (start,end)
        s = slicebase.copy()
        s[d] = 0
        while 1: # start
##            print(a[tuple(s)]==edgevalue)
            if (a[tuple(s)]==edgevalue).all():
                s0 = slicebase.copy()
                s0[d] = slice(1,a.shape[d])
                a = a[tuple(s0)]
                trims[-1][0] += 1
            else: break
        s = slicebase.copy()
        s[d] = -1
        while 1: # end
##            print(a[tuple(s)]==edgevalue)
            if (a[tuple(s)]==edgevalue).all():
                s1 = slicebase.copy()
                s1[d] = slice(0,-1)
                a = a[tuple(s1)]
                trims[-1][1] += 1
            else: break
    return a, trims


























class curvegen():
    dtype = np.float64
    def __init__(self):
        self.px = np.zeros(1) # x position
        self.py = np.zeros(1) # y position
        self.vx = np.zeros(1) # x velocity
        self.vy = np.zeros(1) # y velocity
        self.ax = np.zeros(1) # x acceleration
        self.ay = np.zeros(1) # y acceleration
    
    def __len__(self): return max([len(getattr(self, x)) for x in ["px","vx","ax"]])
    
    def _sin(self, t, r=1, p=1): return sin_cycles((np.arange(t+1))*r, t, p)
    def _lin(self, t, r=1, p=1): return linear_cycles((np.arange(t+1))*r, t, p)
    def _series(self, length, n=1, weight=1, lin=False): return (self._lin if lin else self._sin)(length, n/4, weight)
    
    def _add(self, what, value, l, offset=0):
        x = getattr(self, what)
        poff = max(offset, 0)
        missing = max(len(l)+poff-len(x), 0)
        if missing>0: x = np.pad(x, (0,missing), mode="edge")
        if offset<0: l = l[abs(offset):]
        x[poff:len(l)+poff] += value*l
        setattr(self, what, x)
        
    def _set(self, what, value, l, offset=0):
        x = getattr(self, what)
        poff = max(offset, 0)
        missing = max(len(l)+poff-len(x), 0)
        if missing>0: x = np.pad(x, (0,missing), mode="edge")
        l = (1-l)*x[poff]+value*l
        if offset<0: l = l[abs(offset):]
        setattr(self, what, np.append(x[:poff], l))
        
    def _pop(self, what, n):
        x = getattr(self, what)
        missing = max(n-len(x), 0)
        if missing: x = np.pad(x, (0,missing), mode="edge")
        xx, x = x[:n], x[n:]
        setattr(self, what, x if x.size else xx[-1:])
        return xx

    def add(self, *args, **kwargs):
        kwargs["add"] = True
        self.update(*args, **kwargs)
    def set(self, *args, **kwargs):
        kwargs["add"] = False
        self.update(*args, **kwargs)
        
    def update(self, size, p=None, v=None, a=None, h=0, n=1, weight=1, lin=True, offset=0, add=True):
        method = self._add if add else self._set
        if size>0: l = abs(self._series(max(size, 1), n, weight, lin))
        else: l = np.ones(1)
        direction = heading_offset(heading_absolute(h))
        if p is not None:
            xy = np.multiply(direction, p)
            method("px", xy[0], l, offset)
            method("py", xy[1], l, offset)
        if v is not None:
            xy = np.multiply(direction, v)
            method("vx", xy[0], l, offset)
            method("vy", xy[1], l, offset)
        if a is not None:
            xy = np.multiply(direction, a)
            method("ax", xy[0], l, offset)
            method("ay", xy[1], l, offset)
            
    def _get(self, n, flat=False, scale=True, **kwargs):
        px = self._pop("px", n)
        py = self._pop("py", n)
        vx = self._pop("vx", n)
        vy = self._pop("vy", n)
        ax = self._pop("ax", n)
        ay = self._pop("ay", n)
        p = np.stack([px,py], axis=1)
        v = np.stack([vx,vy], axis=1)/n
        a = np.stack([ax,ay], axis=1)/n**2
        v += np.cumsum(a, axis=0)
        p += np.cumsum(v, axis=0)
        if flat: p = self._flatten(p)
        elif scale: p = self._scale(p)
        return p.astype(self.dtype)

    def _flatten(self, p):
        p = p[:,1]+np.linspace(0, np.diff([p[-1][1],p[0][1]])[0], len(p))
        p_max = np.abs(p).max()
        return p/p_max

    def _scale(self, p):
        p[:,0] /= np.abs(p[:,0]).max()
        p[:,1] /= np.abs(p[:,1]).max()
        return p
        
    def get(self, size=None, *args, **kwargs): return self._get(len(self) if not size else size, *args, **kwargs)



def join_curve(x, y):
    if np.sign(np.diff(y[:2]))!=np.sign(np.diff(x[-2:])): y = -y
    return np.append(x, y)

def reangle_curve(curve, angle):
    _,d,h = odh((0,0), curve)
    d_angle = h[-1]-angle
    offsets = heading_offset(h-d_angle)
    new_curve = np.expand_dims(d, axis=1)*offsets
    return new_curve/np.abs(new_curve).max()
    
def rotate_curve(curve, rotation):
    _,d,h = odh((0,0), curve)
    offsets = heading_offset(h+rotation)
    new_curve = np.expand_dims(d, axis=1)*offsets
    return new_curve/np.abs(new_curve).max()
















def circle_mask(r, inner=0, border=0):
    d = int(r*2+1)
    ogrid_y, ogrid_x = np.ogrid[:d, :d]
    dist_from_center = np.sqrt((ogrid_x-r)**2 + (ogrid_y-r)**2)
    mask = dist_from_center<=r
    if inner>0: mask[dist_from_center<=inner] = False
    elif inner<0: mask[dist_from_center<=r+inner] = False
    return np.pad(mask, (border, border))


def line_gradient(shape, degrees=0, offset=0, width=0):
    o = heading_offset(degrees)
    o_abs = np.abs(o).astype(np.float16)
    width_abs = abs(width)
    
    if o_abs[0]<o_abs[1]:
##        print("vertical", o)
        x = shape[1]*o[0]
        arange = np.arange(shape[0]).astype(np.float32)-(shape[0]-1)/2-offset
        mask = np.stack([arange.copy()+x*(i/max(shape[1]-1, 1)-.5) for i in range(shape[1])]).astype(np.float16)
        if o[1]<0: mask = np.flip(mask, axis=1)
        under = mask<-width_abs
        over = mask>width_abs
        good = np.logical_and(~under, ~over)
        mask[under] = 1
        if width!=0: mask[good] = np.clip(1-(mask[good]+width)/(width*2), a_min=0, a_max=1)
        else: mask[good] = 2
        mask[over] = 0
    else:
        if o_abs[0]==o_abs[1] and width==0:
##            print("diagonal", o)
            if o[0]>0:
                mask = np.stack([np.concatenate([np.ones(shape[0]-i), np.zeros(i)]) for i in range(shape[1])])
                if o[1]<0: mask = np.flip(mask, axis=1)
            else:
                mask = np.stack([np.concatenate([np.zeros(i),np.ones(shape[0]-i)]) for i in range(shape[1]-1,-1,-1)])
                if o[1]>0: mask = np.flip(mask, axis=1)
        else:
##            print("horizontal", o)
            y = shape[0]*o[1]
            arange = np.arange(shape[1]).astype(np.float32)-(shape[1]-1)/2-offset
            mask = np.stack([arange.copy()+y*(i/max(shape[0]-1, 1)-.5) for i in range(shape[0])]).astype(np.float16)
            mask = mask.swapaxes(0, 1)
            if o[0]<0: mask = np.flip(mask, axis=0)
            under = mask<-width_abs
            over = mask>width_abs
            good = np.logical_and(~under, ~over)
            mask[under] = 1
            if width!=0: mask[good] = np.clip(1-(mask[good]+width)/(width*2), a_min=0, a_max=1)
            else: mask[good] = 2
            mask[over] = 0
    return np.abs(mask)


##def mask_split(shape, degrees, ratio=.5): # WIP
##    d = max(shape)#*2**.5
##    return box_gradient(shape, degrees, d*ratio-d/2, width=0).astype(np.bool_)

def mask_blur(mask, n=1):
    mask = mask.astype(np.float16)
    
    y_d = mask[n:,:].copy()
    y_u = mask[:-n,:].copy()
    x_l = mask[:,:-n].copy()
    x_r = mask[:,n:].copy()
    
    mask[:-n,:] += y_d
    mask[n:,:] += y_u
    mask[:,n:] += x_l
    mask[:,:-n] += x_r
    if mask.size and mask.any():
        mask /= mask.max()
    return np.clip(mask, a_min=0, a_max=1)













# TODO: symmetric distribution functions (expected value, variance)
def normal(value, expected, variance): #   normal / Gauss
    return (1/(2*np.pi)**.5)*np.e**(-.5*((value-expected)/variance**.5)**2)
#   logistic
#   laplace

# TODO: unsymmetric distribution functions (expected value, variance)
#   exponent
#   gamma
#   weibull
#   log-normal
#   generalized gamma


##if __name__ == "__main__":
##    pass

