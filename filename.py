import sys, os
import pyperclip
sys.path.insert(0, 'lib')
import ospck

if __name__ == "__main__":
    if os.path.isfile("filenames.txt"):
        usednames = open("filenames.txt", "r", encoding="ascii").read().split("\n")
    else: usednames = []
    l = 6
    while 1:
        rngname = ospck.randomhex(l, usednames)
        if rngname in usednames: l += 1
        else: break
    pyperclip.copy(rngname)
    open("filenames.txt", "a", encoding="ascii").write(rngname+"\n")
    pass

