import sys
sys.path.insert(0, 'lib')
import textprc

def loop(lan="en"):
    total = 0
    while 1:
        for l in range(1,6):
            total += 1
            if lan=="fi": word = textprc.fantasyfinnish(textprc.randomword(l+1, lan))
            else: word = textprc.randomword(l+1, lan)
            print(f" {total}.", word.title())
        x = input("more? [Y/n]")
        if x.lower()=="n": break

def main():
    while 1:
        print("options: en/fi/fr/es/eo/hu/nl/da/de/is/se/pt/pl/it/cs/la/ja/jah/jak")
        lan = input("language: ").lower()
        try: loop(lan)
        except KeyError:
            print("invalid language")
        print("")

if __name__ == "__main__":
    main()

