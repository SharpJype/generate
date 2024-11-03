import os
import ospck
from arrayprc import *

from PIL import Image as PILimg
import imageio.v2 as imageio


def diffuse(a):
    if a.ndim==3: return [np.reshape(x, (x.shape[:2])) for x in np.split(a, a.shape[2], axis=2)]
    return []
def fuse(l):
    for x in l: x = imageexpand(x, 1)
    if len(l)>0: return np.stack(l, axis=2)
    
def imageexpand(a, c=3):
    if a.ndim == 2: a = np.repeat(np.expand_dims(a, axis=-1), c, axis=2)
    elif a.ndim == 3:
        if a.shape[-1]<c: a = np.pad(a, ((0,0),(0,0),(0,c-a.shape[-1])), mode="edge")
    return a

def subimage(array, size=1., point=(0.5, 0.5)): # partial array around a point
    r = np.clip(np.int_(np.multiply(array.shape[:2], size)), a_min=1, a_max=None)
    refpixel = np.int_(np.multiply(array.shape[:2], point))-arraycenter(r)
    botright = refpixel+r
    return array[refpixel[0]:botright[0], refpixel[1]:botright[1]]


def shapenormalize(origshape, newshape, aspect=True, keepsmaller=True, keeplarger=True, hitwidth=True, hitheight=True, even=True):
    s0 = np.array(origshape)
    s1 = newshape
    if keepsmaller:
        if aspect: s1 = s0*np.min(np.minimum(s0, s1)/s0) # change size to keep within size + keep the aspect ratio
        else: s1 = np.minimum(s0, s1) # squeeze to keep within size
    elif keeplarger:
        if aspect: s1 = s0*np.max(np.maximum(s0, s1)/s0)
        else: s1 = np.maximum(s0, s1)
    elif hitwidth and hitheight:
        if aspect: s1 = s0*np.max(s1/s0) # change size to hit all sides + keep the aspect ratio
        #else: s1 = s0
    elif hitheight:
        if aspect: s1 = s0*np.max(s1[1]/s0[1]) # change size to hit height side + keep the aspect ratio
        else: s1 = (s0[0], s1[1])
    elif hitwidth:
        if aspect: s1 = s0*np.max(s1[0]/s0[0]) # change size to hit width side + keep the aspect ratio
        else: s1 = (s1[0], s0[1]) # change size to hit width side
    else: s1 = s0
    s1 = np.round(s1)
    if even:
        a = s1%2!=0
        s1[a] = s1[a]+1
    return tuple(s1.astype(np.uint16))

def validifyimage(a, colors=True, alpha=True):
    if colors: a = imageexpand(a, 4 if alpha else 3)
    elif alpha: a = imageexpand(a, 2)
    if not colors and a.shape[2]>=3: a[:,:,0] = np.mean(a[:,:,:3], axis=2)
    if colors and not alpha: a = a[:,:,:3]
    elif not colors and not alpha: a = a[:,:,0]
    elif not colors and alpha: a = a[:,:,:2]
    return a


def imageresize(path, out, shape):
    dirs, name, ext = ospck.explode(path)
    try: img = PILimg.open(path)
    except: return False
    shape = shapenormalize(img.size, shape)
    if getattr(img, "is_animated", False):
        imgs = []
        img0 = img.resize(shape)
        while 1:
            try: img.seek(img.tell()+1)
            except: break
            imgs.append(img.resize(shape))
        img0.save(out, save_all=True, append_images=imgs, optimize=True, interlace=False, **img.info)
    else:
        img = img.resize(shape)
        if ext=="webp":
            ext = "jpg"
            out = out.replace(".webp", ".jpg")
        if ext in ["jpg","jpeg"]: img.save(out, quality=95, optimize=True, progressive=True)
        else: img.save(out, optimize=True)
    return True


def gifsave(path, array, **info): return imageio.mimwrite(path, array, **info)
def save(path, a, **info):
    if a.ndim==4: return gifsave(path, a, **info) 
    dirs, name, ext = ospck.explode(path)
    os.makedirs(ospck.implode(dirs), exist_ok=True)
    mode = info.get("mode", "RGBA")
    if not ext: ext = info.get("ext", "png")
    path = ospck.implode(dirs, name, ext)
    a = validifyimage(a.copy(), colors=("RGB" in mode), alpha=("A" in mode))
    if ext=="jpg": PILimg.fromarray(a).save(path, quality=95, optimize=True, progressive=True)
    elif ext=="webp": PILimg.fromarray(a).save(path, format=ext, quality=100, lossless=True, exact=True)
    else: PILimg.fromarray(a).save(path, optimize=True)

def gifload(path, img=None):
    if os.path.isfile(path):
        if not img: img = PILimg.open(path)
        info = img.info
        img.close()
        l = imageio.mimread(path)
        mode = l[0].shape[-1]
        for i,a in enumerate(l):
            if a.shape[-1]>mode: l[i] = a[:,:,:mode]
        return np.asarray(l), info
    return np.zeros(0, dtype=np.uint8), {}
def load(path):
    if os.path.isfile(path):
        dirs, name, ext = ospck.explode(path)
        try: img = PILimg.open(path)
        except: return np.zeros(0, dtype=np.uint8), {}
        mode = img.mode
        if getattr(img, "is_animated", False): return gifload(path, img)
        if ext in ["tif","tiff"]: ext = "tif"
        elif ext in ["png","tga"]: ext = "png"
        elif ext in ["jpeg","jpg","webp"]: ext = "jpg" # rgb
        elif ext in ["pgm","xbm"]: ext = "jpg" # 8/16-bit greyscale
        elif ext in ["pbm"]: # 1-bit greyscale
            mode = "1"
            ext = "png"
        elif ext in ["ppm"]: # 24-bit
            mode = "RGB"
            ext = "bmp"
        else: return np.zeros(0, dtype=np.uint8), {}
        if mode=="P":
            mode = "RGBA" if img.has_transparency_data else "RGB"
            img = img.convert(mode)
        elif mode=="PA":
            mode = "RGBA"
            img = img.convert(mode)
        if mode in ["LA","RGBA"] and ext == "jpg": ext = "png"
        a = np.asarray(img, dtype=np.uint8)
        info = {
            "mode": mode,
            "ext": ext,
            }
        return a, info
    return np.zeros(0, dtype=np.uint8), {}




def stretchtoshape(a, shape):
    dshape = np.divide(shape, a.shape)
    dshape[dshape<1] = 1
    for x in range(a.ndim): a = np.repeat(a, dshape[x], axis=x)
    pw = np.subtract(shape, a.shape)/2
    pw[pw<0] = 0
    ow = pw.astype("int")
    pw = ((pw-ow)*2).astype("int")
    return np.pad(a, tuple([(ow[0]+pw[0],ow[0]),(ow[1]+pw[1],ow[1])]+[(0,0) for _ in range(max(0, a.ndim-2))]), mode="edge")





def value_label(a, n=3):
    focusvalues = cluster(np.sort(np.unique(a)), n)
    d = np.abs(np.subtract(np.expand_dims(a, axis=-1), np.expand_dims(focusvalues, axis=0))) # distances to focusvalues
    return np.argmin(d, axis=-1)
def size_label(a, n=3): # 
    uvalues = np.unique(a)
    b = np.zeros(a.shape)
    uvaluesizes = []
    for u in uvalues:
        s = a[a==u].size
        b[a==u] = s
        uvaluesizes.append(s)
    focusvalues = cluster(np.sort(uvaluesizes), n) # np.flip(np.sort(uvaluesizes))
    d = np.abs(np.subtract(np.expand_dims(b, axis=-1), np.expand_dims(focusvalues, axis=0))) # distances to uvaluesizes
    return np.argmin(d, axis=-1)


def imagetrimmer(a, edgevalue=None): # autofind & trim edgevalues from a 2D/3D array
    if a.ndim <2: return None
    elif a.ndim >2:
        if is_iterable(edgevalue):
            if len(edgevalue) != a.shape[2]: edgevalue = None
            else: edgevalue = totuple(edgevalue)
        else: edgevalue = None
    elif a.ndim ==2 and not (is_int(edgevalue) or is_float(edgevalue)): edgevalue = None # type(edgevalue) in [int,float]
    if edgevalue==None: # search for the edgevalue
        slicebase = [slice(0,a.shape[d]) for d in range(2)]
        alledgevalues = []
        alledgecounts = []
        for d in range(2):
            s0 = slicebase.copy()
            s0[d] = 0
            s1 = slicebase.copy()
            s1[d] = a.shape[d]-1
            values, counts = np.unique(np.concatenate([a[tuple(s0)], a[tuple(s1)]]), axis=None if a.ndim<3 else 0, return_counts=True)
            values = totuple(values)
            for x in range(len(values)):
                if values[x] in alledgevalues:
                    alledgecounts[alledgevalues.index(values[x])] += counts[x]
                else:
                    alledgevalues.append(values[x])
                    alledgecounts.append(counts[x])
        l = list(zip(alledgevalues, alledgecounts))
        l.sort(reverse=True, key=lambda x:x[1])
        edgevalue = l[0][0]
    # trim
    a, trims = trimvalues(a, edgevalue)
    return a



def boxblue3D(array, size=3): # boxblur per channel
    if array.ndim == 3:
        channels = [np.expand_dims(boxblur(array[:,:,c], size), axis=-1) for c in range(array.shape[-1])]
        return np.concatenate(channels, axis=-1)
    return array
def boxblur(array, size=3): # simple 2D blur
    blurarray = np.zeros(array.shape)
    for idx,x in np.ndenumerate(array):
        a = getsurroundings(array, idx, size, pad=True)
        h = hanning2D(a.shape)
        slices = smallestrect(a>0)
        m = (a[slices]*h[slices]).mean()
        r = m/array.mean()
        blurarray[idx] += x*(1-r)+m*r
    return blurarray.astype(array.dtype)
def hanning2D(shape):
    asd = []
    for d in range(len(shape)):
        l = []
        for n in shape[:d]+shape[d+1:]:
            for i in range(n): l.append(np.hanning(shape[d]))
        asd.append(np.stack(l))
    return asd[0]*asd[1]#.transpose()
def smallestrect(a): # trim Falses from bool array and return slices for the smallest possible rectangle
    b, trims = trimvalues(a.astype("bool"), False)
    return tuple([slice(trims[d][0], a.shape[d]-trims[d][1]) for d in range(a.ndim)])




if __name__ == "__main__":
    pass
    
