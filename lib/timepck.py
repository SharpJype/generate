from datetime import datetime as dtdt
from datetime import date as dtd
from datetime import time as dtt
from datetime import timedelta as dttd
from datetime import timezone as tz
from time import perf_counter as pec
from time import perf_counter_ns as nspec # nanoseconds
from time import sleep
import re


def func_timer_decor(func):
    def wrapper(*args, **kwargs):
        t_start = nspec()
        out = func(*args, **kwargs)
        t_stop = nspec()
        print(func, f"{(t_stop-t_start)/1e6} ms")
        return out
    return wrapper

def datetime(ms=False):
    date, time = utcnow(True).split("T")
    return date, time[:8+4*ms]
def datetime_print(*args, depth=0, **kwargs): return print(*datetime(), "|"+" "*int(depth), *args, **kwargs)

def utcnow(iso=False):
    if iso: return dtdt.now(tz.utc).isoformat()
    return dtdt.now(tz.utc)

def dtdtformat(x):
    if type(x)==dtdt: return x
    return dtdt.fromisoformat(x)
def isoformat(x):
    if type(x)==dtdt: return x.isoformat()
    return x
def utcadd(t, years=0, months=0, weeks=0, days=0, hours=0, minutes=0, seconds=0, microseconds=0):
    iso = type(t)==str
    if not iso: t = t.isoformat()
    a, m, t = t.split("-", 2)
    m = int(m)+months
    intm = int(m/13)
    t = str(min(max(int(a)+years+intm, 1), 9999)).rjust(4, "0")+"-"+str(max(m-intm*12, 1)).rjust(2, "0")+"-"+t
    t = dtdtformat(t)+dttd(days=days+weeks*7, hours=hours, minutes=minutes, seconds=seconds, microseconds=microseconds)
    if iso: return t.isoformat()
    return t

def utcdur(a, b=None): # duration between utcnow()s
    if not b: return utcnow(False)-dtdtformat(a)
    return dtdtformat(a)-dtdtformat(b) # timedelta object
def utcdelta(deltaobj, years=False, days=False, hours=False, minutes=False, seconds=False, microseconds=False, string=False): # oneway
    b = [years, days, hours, minutes, seconds, microseconds]
    s = "adhmsµ"
    d = [0,0,0,0,0,0] # 0a 1d 2h 3m 4s 5µ
    d[1] = deltaobj.days
    d[4] = deltaobj.seconds
    if microseconds: d[5] = deltaobj.microseconds
    if years: # 365 days
        d[0] = int(d[1]/365)
        d[1] = d[1]%365
    if hours: # 3600 seconds
        d[2] = int(d[4]/3600)
        d[4] = d[4]%3600
    if minutes: # 60 seconds
        d[3] = int(d[4]/60)
        d[4] = d[4]%60
    if not days:
        if hours: d[2] += d[1]*24
        elif minutes: d[3] += d[1]*1440
        elif seconds: d[4] += d[1]*86400
        elif microseconds: d[5] += d[1]*86400000000
        d[1] = 0
    if not seconds:
        if microseconds: d[5] += d[4]*1000000
        d[4] = 0
    if string: return "".join([str(x)+s[i] for i,x in enumerate(d) if b[i]])
    return [x for i,x in enumerate(d) if b[i]]


def utcpath(x): # turn datetime object to a path friendly string or reverse it
    if type(x)==dtdt: return x.strftime("%Ya%mm%dd_%Hh%Mm%Ss")
    elif type(x)==str and re.match(r'(?:(\d+)a)(?:(\d+)m)(?:(\d+)d)_(?:(\d+)h)(?:(\d+)m)(?:(\d+)s)', x): # reverse
        return dtdt(tzinfo=tz.utc, year=int(found.group(1)), month=int(found.group(2)), day=int(found.group(3)), hour=int(found.group(4)), minute=int(found.group(5)), second=int(found.group(6)))

class stopwatch(): # nanoseconds
    def __init__(self):
        self.now = self.start = nspec() # ns integer
        self.total = self.lap = 0
    def reset(self): self.now = self.starttime = nspec()
    def __call__(self):
        now = nspec()
        self.total = now-self.start
        self.lap = now-self.now
        self.now = now

def ticks(seconds=10, rate=1):
    ns_rate = int(rate*1e9)
    f_acc = str(seconds)
    if "." in f_acc: f_acc = len(f_acc[f_acc.index(".")+1:])+1
    else: f_acc = 1
    sw = stopwatch()
    while 0<seconds:
        sw()
        yield round(seconds,f_acc) if f_acc>1 else round(seconds)
        t_delta = (ns_rate-sw.lap)/1e9
        if t_delta>0: sleep(t_delta)
        sw()
        seconds -= sw.lap/1e9



if __name__ == "__main__":
##    t_start = nspec()
##    n = 10
##    for t in ticks(n, .5): datetime_print("-"*int(t)+str(t), depth=0)
##    t_stop = (nspec()-t_start)/1e6
##    print(t_stop, "ms")
    
##    ticks(10, 0.5)
##    print(utcpath(utcnowtime()))
    
##    x = utcadd(utcnow(False), years=10000, months=-100000)
##    print(type(x))
    
##    t = utcnow(True)
##    print(t)
##    print(utcadd(t, days=2))
    ##
##    s = utcnow()
##    print(s)
##    print(dtdtformat(s))
##    e = utcnow(False)
##    print(e)
##    es = utcpath(e)+"___"
##    print(es)
##    print(utcpath(es))
##    d = utcdur(e, s)
##    print(str(d))

##    d = dttd(days=365)-dttd(seconds=5000)
##    d = utcdelta(d, years=True, hours=True, minutes=True, seconds=True, microseconds=True, string=True)
##    print(d)
    
##    durstring = "{0}a{1}h{2}min".format(*d)
##    print(durstring) # , years=False, days=True, hours=False, minutes=False, seconds=True, microseconds=True
##    print(utcpath(d))
##    print(utcpath(utcnow(False)))
    
    ## stopwatch
##    sw = stopwatch()
##    sw.reset()
##    for x in range(20): sw()
##    sw.reset()
##    for x in range(20): sw()
##    print(sw.time)

    ## clock
##    ms = 4.#66666666666666666666
##    c = clock(ms) #
##    pc = time.Clock()
##    c.start()
##    i = 0
##    starttime = nspec()
##    for x in range(100):
####        print(i, c.stopwatch.sincecall/10**9)
##        i = c()
####        print(round(1000/ms))
####        pc.tick(round(1000/ms))
##    t = nspec()-starttime
##    print(t)

    # synctime
##    x = synctime()
##    print(x)
##    print(synctime_i(x))
    
    pass
