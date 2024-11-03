import os, sys
import pyperclip
import secrets

sys.path.insert(0, "lib")
import ospck

##def randompassword(dictionaryorlist, length=5):
##    x = ""
##    while not x or x in dictionaryorlist:
##        x = textprc.artificialword(length, 0, 0, 0)[:length]
##        x = np.array(list(x))
##        b = np.random.randint(0,2, x.size).astype("bool")
##        x[b] = np.random.randint(0,10, x.size)[b]
##        x = "".join(x.tolist())
##        length += 1
##    return x


def randomascii(l): # ascii
    def func(x, y): return str(2+round(x/51)%10)+hex(round(y/16)%10)[2:]
    
    x = os.urandom(l*2)
    x = "".join(["."+func(x[i], x[i+1]) for i in range(0, l*2, 2)]).replace("20", "21").replace("7f", "61").replace(".", "")
    return "".join([chr(int(x[i:i+2], 16)) for i in range(0, l*2, 2)])

def randomsimple(l):
    special_char = secrets.choice(".,-_?!=+/*")
    return ospck.randomint(l//3)+ospck.randomstr(l//3)+ospck.randomstr(l-(l//3)*2).upper()+special_char

def segments(l, partlen=4, sep=" "):
    funcs = [ospck.randomstr,ospck.randomhex]
    
    seg_n = l//partlen
    x = [secrets.choice(funcs)(partlen) for i in range(seg_n)]
    
    x = [xx if os.urandom(1)[0]%2 else xx.upper() for xx in x]
    x = sep.join(x)[:l]
    if x[-1]==sep: x = x[:-1]+secrets.choice(funcs)(1)[0]
    return x

def main_loop(length, blacklist, genfunction):
    x = ""
    while not x or x in blacklist:
        x = genfunction(length)
    return x
    
    

def main(length=14, passwordfile="passwords.txt", func="ascii"):
    if os.path.isfile(passwordfile):
        passwords = open(passwordfile, "r", encoding="utf8").read().split("\u000A")
    else: passwords = []
    
    while "" in passwords: passwords.remove("")
    match func:
        case "ascii": func = randomascii
        case "segments": func = segments
        case _: func = randomsimple
    rngname = main_loop(length, passwords, func)
    
    pyperclip.copy(rngname)
    open(passwordfile, "a", encoding="utf8").write(rngname+"\n")
    


if __name__ == "__main__":
    main(func="segments")
    pass

