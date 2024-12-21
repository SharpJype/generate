import sys, os
import pyperclip
sys.path.insert(0, 'lib')
import timepck

def main():
    file = "filename.txt"
    date = timepck.cleandate()
    count = 0
    if os.path.isfile(file):
        with open(file, "r", encoding="ascii") as f:
            olddate, count = f.read().rsplit("-", 1)
            if olddate==date: count = int(count)+1
            else: count = 0
    name = date+"-"+str(count)
    pyperclip.copy(name)
    with open(file, "w", encoding="ascii") as f:
        f.write(name)


if __name__ == "__main__":
    main()
    pass

