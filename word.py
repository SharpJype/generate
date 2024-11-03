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
        x = input()

if __name__ == "__main__":
    print("language options:")
    print("\ten/fi/fr/es/eo/hu/nl/da/de/is/se/pt/pl/it/cs/la/ja/jah/jak")
    while 1:
        lan = input("choose language: ")
        try: loop(lan)
        except KeyError: print("invalid language")

