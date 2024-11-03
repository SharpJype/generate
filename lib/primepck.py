import numpy as np
import random
def is_array(x): return type(x)==np.ndarray

def is_prime(x): # check for primes
    if is_array(x):
        x = np.int_(x)
        notvalid = np.logical_or(x<=1, np.logical_or(np.mod(x, 2)==0,np.mod(x, 3)==0))
        i = 5
        while not (i**2>x[~notvalid]).all():
            notvalid[~notvalid] = np.logical_or(x[~notvalid]%i==0, x[~notvalid]%(i+2)==0)
            i += 6
        return np.logical_or(np.logical_or(~notvalid, x==2), x==3)
    if x<=3: return x>1
    elif x%2==0 or x%3==0: return False
    i = 5
    while i**2<=x:
        if x%i==0 or x%(i+2)==0: return False
        i += 6
    return True

def prev_prime(start):
    if is_array(start):
        start = np.int_(start)
        start[start<=2] = 2
        start = start+(1-np.mod(start, 2))
        i = np.zeros(start.shape, dtype=start.dtype)
        searching = np.ones(start.shape, dtype=np.bool_)
        while searching.any():
            searching[searching] = ~is_prime(start[searching]+i[searching])
            i[searching] -= 2
        return start+i
        
    if 2>=start: return 2
    start -= (1-start%2)
    i = 0
    while not is_prime(start+i): i -= 2
    return start+i

def next_prime(start):
    if is_array(start):
        start = np.int_(start)
        start[start<=2] = 2
        start = start+(1-np.mod(start, 2))
        i = np.zeros(start.shape, dtype=start.dtype)
        searching = np.ones(start.shape, dtype=np.bool_)
        while searching.any():
            searching[searching] = ~is_prime(start[searching]+i[searching])
            i[searching] += 2
        return start+i
    
    if 2>=start: return 2
    start += (1-start%2)
    i = 0
    while not is_prime(start+i): i += 2
    return start+i

def nearest_prime(start, reverse=False):
    lambda_func = (lambda x:x-2) if reverse else (lambda x:x+2)
    if is_array(start):
        start = np.int_(start)
        start[start<=2] = 2
        start = start+(1-np.mod(start, 2))
        i = np.zeros(start.shape, dtype=start.dtype)
        searching = np.ones(start.shape, dtype=np.bool_)
        while searching.any():
            searching[searching] = ~is_prime(start[searching]-i[searching])
            i[searching] = lambda_func(i[searching])
            searching[searching] = ~is_prime(start[searching]+i[searching])
        return start+i
    
    if 2>=start: return 2
    start += (1-(start%2))*(1-reverse*2)
    i = 0
    while 1:
        if is_prime(start-i): return start-i
        i = lambda_func(i)
        if is_prime(start+i): return start+i

def random_prime(low, high, shape=None):
    low, high = next_prime(int(low)), prev_prime(int(high))
    if low>=high: return
    if shape is not None:
        return nearest_prime(np.random.randint(low, high, shape))
    return nearest_prime(random.randint(low, high))

def find_primes(low:int, high:int=0, step:int=1, limit:int=0) -> int:
    low += (1-low%2)
    if 2>=low:
        yield 2
        low = 3
    i = 0
    count = 0
    while (high<1 or (low+i)<=high) and (limit<1 or limit>count):
        if is_prime(low+i):
            if not count%step: yield low+i
            count += 1
        i += 2


if __name__ == "__main__":
    pass
