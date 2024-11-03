import sys
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, 'lib')
import textprc


# get a random wikipedia article
banned = ['Wikipedia',' is a stub.', 'This article', 'lähde?', 'source?']
def get_random_wikipedia_article(language="en"):
    if language=="fi": url = 'https://fi.wikipedia.org/wiki/Toiminnot:Satunnainen_sivu'
    else: url = f'https://{language}.wikipedia.org/wiki/Special:Random'
    
    resp = requests.get(url)
    sentences = []
    if resp.status_code==200:
        soup = BeautifulSoup(resp.text,'html.parser')
        l = soup.find("div",{"class":"mw-parser-output"})
        
        for p in l.findAll("p"):
            if p.text: sentences += textprc.sentences(p.text)
        for b in banned:
            bans = 0
            for i,s in enumerate(sentences.copy()):
                if b in s:
                    sentences.pop(i-bans)
                    bans += 1
    return sentences

if __name__ == "__main__":
    while 1:
        for x in get_random_wikipedia_article("en"):
            if x: print(x, "\n")
        input("\nget another?")
    pass
