import os
import re
import random
import numpy as np
from io import FileIO
import textprc_syllables as syllables

## from arrayprc
def is_float(x, array=True): return type(x)==float or isinstance(x, np.floating) or (array and is_array(x) and is_float(getattr(np, str(x.dtype))(1)))
def is_integer(x, array=True): return type(x)==int or isinstance(x, np.integer) or (array and is_array(x) and is_integer(getattr(np, str(x.dtype))(1)))

def floatrng(a, acc=2): return rng([int((1-a)*10**acc),int(a*10**acc)])

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

def dictrng(d, amount=1): # treat a dict of floats/integers as a chance list
    return (np.array(list(d.keys()))[rng_bulk(list(d.values()), amount)]).tolist()
## arrayprc ends





linebreaks = [u"\u000A",u"\u000B",u"\u000C",u"\u000D"] # standard newline, vertical tab, form feed, carriage return
whitespaces = [
    u"\u0009",u"\u00A0",u"\u1680",u"\u2000",u"\u2001",u"\u2002",u"\u2003",u"\u2004",u"\u2005",u"\u2006",
    u"\u2007",u"\u2008",u"\u2009",u"\u200A",u"\u2003",u"\u2003",u"\u202F",u"\u205F",u"\u3000",
    u"\u180E",u"\u200B",u"\u200C",u"\u200D",u"\u2060",u"\uFEFF",u"\u2800",u"\u3164",u"\u115F",u"\uFFA0"]
def pathfix(string, x="_"): # replace any non windows path friendly with x
    return re.sub(r"[:\|\\\*/\?><]", x, string).replace("\"", "'")
def spacefix(string, x=" "): # replace any whitespace-like character with x
    return re.sub("["+"".join(whitespaces)+"]", x, string)
def linefix(string, x="\n"): # replace linebreaks with x
    return re.sub("["+"".join(linebreaks)+"]", x, string)

def is_a_value(x:str): # is_number
    return bool(re.match(r'^\s*(?:\d+\.?\d*|\d*\.?\d+)\s*$', x))



##
def c2u(x): # character or string as unicode character codes
    x = re.sub('.', lambda x: r'%04X' % ord(x.group()), x)
    if len(x)!=4 and x in linebreaks: x = "000"+"abcd"[linebreaks.index(x)]
    return x
def urand(e=4): return hex(random.randint(0,16**min(e, 4)))[2:].rjust(4, "0")
def u2c(x): # inverse
    try: return eval("u'\\u"+x+"'")
    except: pass











re_abr = {
    "en": r'(?:[A-Z])(?:(?:[a-z])|(?:rs))?\.',
    "fi": r'((?:mm|jne|ao|eaa|eKr|em|eo|esim|huom|jaa|jKr|jms|k|ka|ko|ks|l|ma|ml|mrd|n|nk|ns|os|o\.s|oto|pl|po|puh|pvm|s|tjsp|tm|tms|tmv|ts|v|va|vrt|vs|vt|yht|ym|yms|yo)\.)(?:\s)',
    }
def sentences(string):
    string = string.replace("\n", "")
    for x in re.findall(r"\[(?:\d+])|(?:citation needed])|(?:vague])", string): string = string.replace(x, "")
    for x in re.findall(r"\s+", string): string = string.replace(x, " ")
    for x in re.findall(r"\D+\s+([IVXLCDM]{2,})\W", string): string = string.replace(x, str(romannumerals(x)))
    for x in re.findall("\\d+(?:[\\s,]\\d+)+", string): string = string.replace(x, x.replace(",", "").replace(" ", ""))
    for x in re.findall("\\d*\\.\\d+", string): string = string.replace(x, x.replace(".", ","))
    for x in re.findall(r'(\d+?\.)(?:\s[a-z]+)', string): string = string.replace(x, x.replace(".", ""))
    for x in re.findall(re_abr["en"], string): string = string.replace(x, x.replace(".", ""))
    for x in re.findall(re_abr["fi"], string): string = string.replace(x, x.replace(".", ""))
    dots = []
    for i,x in enumerate(re.findall(r'(\.{3,})(?:\s)', string)):
        string = string.replace(x, "d{"+str(i)+"}")
        dots.append("...") # [:-1]
    brackets = []
    for i,x in enumerate(re.findall(r'\([^\(]*\)', string)):
        string = string.replace(x, "b{"+str(i)+"}")
        brackets.append(x)
    sents = re.findall('(?:[\'\"]?[A-Z][^\\.!?]+)(?:[\\.!?]|$)', string, re.M)
    if sents:
        for i,x in enumerate(brackets):
            for ii,xx in enumerate(sents): sents[ii] = sents[ii].replace("b{"+str(i)+"}", x)
        for i,x in enumerate(dots):
            for ii,xx in enumerate(sents): sents[ii] = sents[ii].replace("d{"+str(i)+"}", x)
        return sents
    else:
        for i,x in enumerate(brackets): string = string.replace("b{"+str(i)+"}", x)
        for i,x in enumerate(dots): string = string.replace("d{"+str(i)+"}", x)
        return [string]




default_soft = "aeiouy"
default_hard = "bdfghjklmnprstvqz"
def listpick(l, n=1):
    picks = []
    while len(l)>0 and len(picks)<n: picks.append(l.pop(random.randint(0,len(l)-1)))
    return picks
def artificialword_settings(soft=default_soft, hard=default_hard, special=default_hard):
    if not soft: soft = default_soft
    elif type(soft)==int: soft = "".join(listpick(list(default_soft), min(soft, len(default_soft))))
    if not hard: hard = default_hard
    elif type(hard)==int: hard = "".join(listpick(list(default_hard), min(hard, len(default_hard))))
    if not special: special = default_hard
    elif type(special)==int: special = "".join(listpick(list(default_hard), min(special, len(default_hard))))
    return soft, hard, special
def artificialword(segments, soft=default_soft, hard=default_hard, special=default_hard):
    soft, hard, special = artificialword_settings(soft, hard, special)
    soft, hard, special = soft.lower(), hard.lower(), special.lower()
    def segment():
        x = ""
        e = random.randint(0, 1)
        if random.randint(0, 1):
            x += random.choice(hard) if hard else ""
            if e and random.randint(0, 1): x += random.choice(special) if special else ""
        x += random.choice(soft) if soft else ""
        if random.randint(0, 1):
            if not e and random.randint(0, 1): x += random.choice(special) if special else ""
            x += random.choice(hard) if hard else ""
        return x
    return "".join([segment() for i in range(segments)])





vowels = "aeiouyàâáåäãąæœ"+"ëêèéęě"+"îìíïı"+"òöôóőõø"+"ùúûŭüý"
consonants = "bcdfghjklmnprstvqz"+"çĉćčďð"+"ĝğĥ"+"ĵłľñńň"+"řŝşśšßťþ"+"źżž"
letterfrequencies = {}
letterfrequencies["en"] = [('a', 8167), ('b', 1492), ('c', 2782), ('d', 4253), ('e', 12702), ('f', 2228), ('g', 2015), ('h', 6094), ('i', 6966), ('j', 153), ('k', 772), ('l', 4025), ('m', 2406), ('n', 6749), ('o', 7507), ('p', 1929), ('q', 95), ('r', 5987), ('s', 6327), ('t', 9056), ('u', 2758), ('v', 978), ('w', 2360), ('x', 150), ('y', 1974), ('z', 74)]
letterfrequencies["fr"] = [('a', 7636), ('b', 901), ('c', 3260), ('d', 3669), ('e', 14715), ('f', 1066), ('g', 866), ('h', 737), ('i', 7529), ('j', 613), ('k', 74), ('l', 5456), ('m', 2968), ('n', 7095), ('o', 5796), ('p', 2521), ('q', 1362), ('r', 6693), ('s', 7948), ('t', 7244), ('u', 6311), ('v', 1838), ('w', 49), ('x', 427), ('y', 128), ('z', 326), ('à', 486), ('â', 51), ('œ', 18), ('ç', 85), ('è', 271), ('é', 1504), ('ê', 218), ('ë', 8), ('î', 45), ('ï', 5), ('ô', 23), ('ù', 58), ('û', 60)]
letterfrequencies["de"] = [('a', 6516), ('b', 1886), ('c', 2732), ('d', 5076), ('e', 16396), ('f', 1656), ('g', 3009), ('h', 4577), ('i', 6550), ('j', 268), ('k', 1417), ('l', 3437), ('m', 2534), ('n', 9776), ('o', 2594), ('p', 670), ('q', 18), ('r', 7003), ('s', 7270), ('t', 6154), ('u', 4166), ('v', 846), ('w', 1921), ('x', 34), ('y', 39), ('z', 1134), ('ä', 578), ('ö', 443), ('ß', 307), ('ü', 995)]
letterfrequencies["es"] = [('a', 11525), ('b', 2215), ('c', 4019), ('d', 5010), ('e', 12181), ('f', 692), ('g', 1768), ('h', 703), ('i', 6247), ('j', 493), ('k', 11), ('l', 4967), ('m', 3157), ('n', 6712), ('o', 8683), ('p', 2510), ('q', 877), ('r', 6871), ('s', 7977), ('t', 4632), ('u', 2927), ('v', 1138), ('w', 17), ('x', 215), ('y', 1008), ('z', 467), ('á', 502), ('é', 433), ('í', 725), ('ñ', 311), ('ó', 827), ('ú', 168), ('ü', 12)]
letterfrequencies["pt"] = [('a', 14634), ('b', 1043), ('c', 3882), ('d', 4992), ('e', 12570), ('f', 1022), ('g', 1303), ('h', 781), ('i', 6186), ('j', 397), ('k', 15), ('l', 2779), ('m', 4738), ('n', 4446), ('o', 9735), ('p', 2523), ('q', 1204), ('r', 6530), ('s', 6805), ('t', 4336), ('u', 3639), ('v', 1575), ('w', 37), ('x', 253), ('y', 6), ('z', 470), ('à', 72), ('â', 562), ('á', 118), ('ã', 733), ('ç', 530), ('é', 337), ('ê', 450), ('í', 132), ('ô', 635), ('ó', 296), ('õ', 40), ('ú', 207), ('ü', 26)]
letterfrequencies["eo"] = [('a', 12117), ('b', 980), ('c', 776), ('d', 3044), ('e', 8995), ('f', 1037), ('g', 1171), ('h', 384), ('i', 10012), ('j', 3501), ('k', 4163), ('l', 6104), ('m', 2994), ('n', 7955), ('o', 8779), ('p', 2755), ('r', 5914), ('s', 6092), ('t', 5276), ('u', 3183), ('v', 1904), ('z', 494), ('ĉ', 657), ('ĝ', 691), ('ĥ', 22), ('ĵ', 55), ('ŝ', 385), ('ŭ', 520)]
letterfrequencies["it"] = [('a', 11745), ('b', 927), ('c', 4501), ('d', 3736), ('e', 11792), ('f', 1153), ('g', 1644), ('h', 636), ('i', 10143), ('j', 11), ('k', 9), ('l', 6510), ('m', 2512), ('n', 6883), ('o', 9832), ('p', 3056), ('q', 505), ('r', 6367), ('s', 4981), ('t', 5623), ('u', 3011), ('v', 2097), ('w', 33), ('x', 3), ('y', 20), ('z', 1181), ('à', 635), ('è', 263), ('ì', 30), ('í', 30), ('ò', 2), ('ù', 166), ('ú', 166)]
letterfrequencies["tr"] = [('a', 11920), ('b', 2844), ('c', 963), ('d', 4706), ('e', 8912), ('f', 461), ('g', 1253), ('h', 1212), ('i', 8600), ('j', 34), ('k', 4683), ('l', 5922), ('m', 3752), ('n', 7487), ('o', 2476), ('p', 886), ('r', 6722), ('s', 3014), ('t', 3314), ('u', 3235), ('v', 959), ('y', 3336), ('z', 1500), ('ç', 1156), ('ğ', 1125), ('ı', 5114), ('ö', 777), ('ş', 1780), ('ü', 1854)]
letterfrequencies["se"] = [('a', 9383), ('b', 1535), ('c', 1486), ('d', 4702), ('e', 10149), ('f', 2027), ('g', 2862), ('h', 2090), ('i', 5817), ('j', 614), ('k', 3140), ('l', 5275), ('m', 3471), ('n', 8542), ('o', 4482), ('p', 1839), ('q', 20), ('r', 8431), ('s', 6590), ('t', 7691), ('u', 1919), ('v', 2415), ('w', 142), ('x', 159), ('y', 708), ('z', 70), ('å', 1340), ('ä', 1800), ('ö', 1310)]
letterfrequencies["pl"] = [('a', 8965), ('b', 1482), ('c', 3988), ('d', 3293), ('e', 7921), ('f', 312), ('g', 1377), ('h', 1072), ('i', 8286), ('j', 2343), ('k', 3411), ('l', 2136), ('m', 2911), ('n', 5600), ('o', 7590), ('p', 3101), ('q', 3), ('r', 4571), ('s', 4263), ('t', 3966), ('u', 2347), ('v', 34), ('w', 4549), ('x', 19), ('y', 3857), ('z', 5620), ('ą', 1020), ('ć', 448), ('ę', 1131), ('ł', 1746), ('ń', 185), ('ó', 823), ('ś', 683), ('ź', 61), ('ż', 885)]
letterfrequencies["nl"] = [('a', 7490), ('b', 1580), ('c', 1240), ('d', 5930), ('e', 18910), ('f', 810), ('g', 3400), ('h', 2380), ('i', 6500), ('j', 1460), ('k', 2250), ('l', 3570), ('m', 2210), ('n', 10030), ('o', 6060), ('p', 1570), ('q', 9), ('r', 6410), ('s', 3730), ('t', 6790), ('u', 1990), ('v', 2850), ('w', 1520), ('x', 36), ('y', 35), ('z', 1390)]
letterfrequencies["da"] = [('a', 6025), ('b', 2000), ('c', 565), ('d', 5858), ('e', 15453), ('f', 2406), ('g', 4077), ('h', 1621), ('i', 6000), ('j', 730), ('k', 3395), ('l', 5229), ('m', 3237), ('n', 7240), ('o', 4636), ('p', 1756), ('q', 7), ('r', 8956), ('s', 5805), ('t', 6862), ('u', 1979), ('v', 2332), ('w', 69), ('x', 28), ('y', 698), ('z', 34), ('å', 1190), ('æ', 872), ('ø', 939)]
letterfrequencies["is"] = [('a', 10110), ('b', 1043), ('d', 1575), ('e', 6418), ('f', 3013), ('g', 4241), ('h', 1871), ('i', 7578), ('j', 1144), ('k', 3314), ('l', 4532), ('m', 4041), ('n', 7711), ('o', 2166), ('p', 789), ('r', 8581), ('s', 5630), ('t', 4953), ('u', 4562), ('v', 2437), ('x', 46), ('y', 900), ('á', 1799), ('æ', 867), ('ð', 4393), ('é', 647), ('í', 1570), ('ö', 777), ('ó', 994), ('þ', 1455), ('ú', 613), ('ý', 228)]
letterfrequencies["fi"] = [('a', 12217), ('b', 281), ('c', 281), ('d', 1043), ('e', 7968), ('f', 194), ('g', 392), ('h', 1851), ('i', 10817), ('j', 2041), ('k', 4973), ('l', 5761), ('m', 3202), ('n', 8826), ('o', 5614), ('p', 1842), ('q', 13), ('r', 2872), ('s', 7862), ('t', 8750), ('u', 5008), ('v', 2250), ('w', 94), ('x', 31), ('y', 1745), ('z', 51), ('å', 3), ('ä', 3577), ('ö', 444)]
letterfrequencies["cs"] = [('a', 8421), ('b', 822), ('c', 740), ('d', 3475), ('e', 7562), ('f', 84), ('g', 92), ('h', 1356), ('i', 6073), ('j', 1433), ('k', 2894), ('l', 3802), ('m', 2446), ('n', 6468), ('o', 6695), ('p', 1906), ('q', 1), ('r', 4799), ('s', 5212), ('t', 5727), ('u', 2160), ('v', 5344), ('w', 16), ('x', 27), ('y', 1043), ('z', 1599), ('á', 867), ('č', 462), ('ď', 15), ('é', 633), ('ě', 1222), ('í', 1643), ('ň', 7), ('ó', 24), ('ř', 380), ('š', 688), ('ť', 6), ('ú', 45), ('ů', 204), ('ý', 995), ('ž', 721)]
letterfrequencies["hu"] = [('a', 10778), ('b', 2647), ('c', 924), ('d', 2410), ('e', 11926), ('f', 1221), ('g', 3650), ('h', 1568), ('i', 5343), ('j', 1321), ('k', 5939), ('l', 7464), ('m', 3621), ('n', 6472), ('o', 4620), ('p', 1573), ('q', 14), ('r', 5188), ('s', 7016), ('t', 9184), ('u', 1224), ('v', 2032), ('w', 72), ('x', 115), ('y', 2319), ('z', 5251), ('á', 4208), ('é', 4072), ('í', 710), ('ö', 1251), ('ó', 1145), ('ő', 1105), ('ú', 320), ('ü', 683), ('ű', 145)]
letterfrequencies["la"] = [('a', 889), ('b', 158), ('c', 399), ('d', 277), ('e', 1138), ('f', 93), ('g', 121), ('h', 69), ('i', 1144), ('k', 1), ('l', 315), ('m', 538), ('n', 628), ('o', 540), ('p', 303), ('q', 151), ('r', 667), ('s', 760), ('t', 800), ('u', 846), ('v', 96), ('x', 60), ('y', 7), ('z', 1)]


ja_sounds = []
base = [("a",u"\u3042",u"\u30A2"),("i",u"\u3044",u"\u30A4"),("u",u"\u3046",u"\u30A6"),("e",u"\u3048",u"\u30A8"),("o",u"\u304A",u"\u30AA")]
ja_sounds.extend(base)
ja_sounds.extend([("ka",u"\u304B",u"\u30AB"),("ki",u"\u304D",u"\u30AD"),("ku",u"\u304F",u"\u30AF"),("ke",u"\u3051",u"\u30B1"),("ko",u"\u3053",u"\u30B3")])
ja_sounds.extend([("ga",u"\u304C",u"\u30AC"),("gi",u"\u304E",u"\u30AE"),("gu",u"\u3050",u"\u30B0"),("ge",u"\u3052",u"\u30B2"),("go",u"\u3054",u"\u30B4")])

ja_sounds.extend([("sa",u"\u3055",u"\u30B5"),("shi",u"\u3057",u"\u30B7"),("su",u"\u3059",u"\u30B9"),("se",u"\u305B",u"\u30BB"),("so",u"\u305D",u"\u30BD")])
ja_sounds.extend([("za",u"\u3056",u"\u30B6"),("ji",u"\u3058",u"\u30B8"),("zu",u"\u305A",u"\u30BA"),("ze",u"\u305C",u"\u30BC"),("zo",u"\u305E",u"\u30BE")])

ja_sounds.extend([("ta",u"\u305F",u"\u30BF"),("chi",u"\u3061",u"\u30C1"),("tsu",u"\u3064",u"\u30C4"),("te",u"\u3066",u"\u30C6"),("to",u"\u3068",u"\u30C8")])
ja_sounds.extend([("da",u"\u3060",u"\u30C0"),("de",u"\u3067",u"\u30C7"),("do",u"\u3069",u"\u30C9")])

ja_sounds.extend([("na",u"\u306A",u"\u30CA"),("ni",u"\u306B",u"\u30CB"),("nu",u"\u306C",u"\u30CC"),("ne",u"\u306D",u"\u30CD"),("no",u"\u306E",u"\u30CE")])
ja_sounds.extend([("ha",u"\u306F",u"\u30CF"),("hi",u"\u3072",u"\u30D2"),("fu",u"\u3075",u"\u30D5"),("he",u"\u3078",u"\u30D8"),("ho",u"\u307B",u"\u30DB")])
ja_sounds.extend([("ba",u"\u3070",u"\u30D0"),("bi",u"\u3073",u"\u30D3"),("bu",u"\u3076",u"\u30D6"),("be",u"\u3079",u"\u30D9"),("bo",u"\u307C",u"\u30DC")])
ja_sounds.extend([("pa",u"\u3071",u"\u30D1"),("pi",u"\u3074",u"\u30D4"),("pu",u"\u3077",u"\u30D7"),("pe",u"\u307A",u"\u30DA"),("po",u"\u307D",u"\u30DD")])

ja_sounds.extend([("ma",u"\u307E",u"\u30DE"),("mi",u"\u307F",u"\u30DF"),("mu",u"\u3080",u"\u30E0"),("me",u"\u3081",u"\u30E1"),("mo",u"\u3082",u"\u30E2")])
ja_sounds.extend([("ya",u"\u3084",u"\u30E4"),("yu",u"\u3086",u"\u30E6"),("yo",u"\u3088",u"\u30E8")])
ja_sounds.extend([("ra",u"\u3089",u"\u30E9"),("ri",u"\u308A",u"\u30EA"),("ru",u"\u308B",u"\u30EB"),("re",u"\u308C",u"\u30EC"),("ro",u"\u308D",u"\u30ED")])
ja_sounds.extend([("wa",u"\u308F",u"\u30EF"),("wi",u"\u3090",u"\u30F0"),("we",u"\u3091",u"\u30F1"),("wo",u"\u3092",u"\u30F2")])

ja_combosounds = [("kya",u"\u304D\u3083",u"\u30AD\u30E3"),("kyu",u"\u304D\u3086",u"\u30AD\u30E6"),("kyo",u"\u304D\u3088",u"\u30AD\u30E8")]
ja_combosounds.extend([("gya",u"\u304E\u3083",u"\u30AE\u30E3"),("gyu",u"\u304E\u3086",u"\u30AE\u30E6"),("gyo",u"\u304E\u3088",u"\u30AE\u30E8")])
ja_combosounds.extend([("sha",u"\u3057\u3083",u"\u30B7\u30E3"),("shu",u"\u3057\u3086",u"\u30B7\u30E6"),("sho",u"\u3057\u3088",u"\u30B7\u30E8")])
ja_combosounds.extend([("cha",u"\u3061\u3083",u"\u30C1\u30E3"),("chu",u"\u3061\u3086",u"\u30C1\u30E6"),("cho",u"\u3061\u3088",u"\u30C1\u30E8")])
ja_combosounds.extend([("nya",u"\u306B\u3083",u"\u30CB\u30E3"),("nyu",u"\u306B\u3086",u"\u30CB\u30E6"),("nyo",u"\u306B\u3088",u"\u30CB\u30E8")])
ja_combosounds.extend([("hya",u"\u3072\u3083",u"\u30D2\u30E3"),("hyu",u"\u3072\u3086",u"\u30D2\u30E6"),("hyo",u"\u3072\u3088",u"\u30D2\u30E8")])
ja_combosounds.extend([("bya",u"\u3073\u3083",u"\u30D3\u30E3"),("byu",u"\u3073\u3086",u"\u30D3\u30E6"),("byo",u"\u3073\u3088",u"\u30D3\u30E8")])
ja_combosounds.extend([("pya",u"\u3074\u3083",u"\u30D4\u30E3"),("pyu",u"\u3074\u3086",u"\u30D4\u30E6"),("pyo",u"\u3074\u3088",u"\u30D4\u30E8")])
ja_combosounds.extend([("mya",u"\u307F\u3083",u"\u30DF\u30E3"),("myu",u"\u307F\u3086",u"\u30DF\u30E6"),("myo",u"\u307F\u3088",u"\u30DF\u30E8")])
ja_combosounds.extend([("rya",u"\u308A\u3083",u"\u30EA\u30E3"),("ryu",u"\u308A\u3086",u"\u30EA\u30E6"),("ryo",u"\u308A\u3088",u"\u30EA\u30E8")])
ja_combosounds.extend([("ja",u"\u3058\u3083",u"\u30B8\u30E3"),("ju",u"\u3058\u3086",u"\u30B8\u30E6"),("jo",u"\u3058\u3088",u"\u30B8\u30E8")])

def randomword_ja(l=5):
    def modif(x, double, n):
        if floatrng(double) and x[0] in "stkpnm": x = x[0]+x
        if floatrng(n): x = x+"n"
        elif floatrng(n): x = x+x[-1]
        return x
    def choice(): return random.choice(ja_combosounds if floatrng(0.2) else ja_sounds)
    word = choice()[0]
    for i in range(l): word += modif(choice()[0], 0.1, 0.1)
    return word
def randomword_jah(l=5):
    def modif(x, double, n):
        if floatrng(double):
            if x[0] in "stkp": x = u"\u3063"+x
            elif x[0]=="n": x = u"\u3093"+x # n
        if floatrng(n): x = x+u"\u3093"
        elif floatrng(n) and not x[-1] in u"\u3083\u3086\u3088": x = x+x[-1]
        elif floatrng(n): x = x+u"\u30FC" # prolonged sound
        return x
    def choice(): return random.choice(ja_combosounds if floatrng(0.2) else ja_sounds)
    word = choice()[1]
    for i in range(l): word += modif(choice()[1], 0.1, 0.1)
    return word
def randomword_jak(l=5):
    def modif(x, double, n):
        if floatrng(double):
            if x[0] in "stkp": x = u"\u30C3"+x # small tsu
            elif x[0]=="n": x = u"\u30F3"+x # n
            elif x[0]=="m": x = u"\u30E0"+x # mu
            elif x[0]=="w": x = u"\u30A5"+x # small u
        elif floatrng(n): x = x+u"\u30F3"
        elif floatrng(n) and not x[-1] in u"\u30E3\u30E6\u30E8": x = x+x[-1]
        elif floatrng(n): x = x+u"\u30FC" # prolonged sound
        return x
    def choice(): return random.choice(ja_combosounds if floatrng(0.2) else ja_sounds)
    word = choice()[2]
    for i in range(l): word += modif(choice()[2], 0.1, 0.1)
    return word

def japanese_latin(string):
    for x,hx,kx in ja_combosounds+ja_sounds: string = string.replace(hx, x).replace(kx, x)
    findings = re.findall(u"(?:[\u30C3\u3063][stkp])|(?:\u30A5w)|(?:\u30E0m)|(?:\u30F3n)", string)
    for x in findings: string = string.replace(x, x[1]*2)
    findings = re.findall(u"[aiueo]\u30FC", string)
    for x in findings: string = string.replace(x, x[0]*2)
    return string.replace(u"\u3093", "n").replace(u"\u30F3", "n")
def latin_hiragana(string):
    l = ja_sounds[5:]
    for x,hx,kx in ja_combosounds+l:
        if x[1] in "ie":
            string = string.replace(x[0]+x+x[1], u"\u3063"+hx+u"\u3044")
            string = string.replace(x+x[1], hx+u"\u3044")
        elif x[1] in "ou":
            string = string.replace(x[0]+x+x[1], u"\u3063"+hx+u"\u3046")
            string = string.replace(x+x[1], hx+u"\u3046")
        if x[0] in "stkp": string = string.replace(x[0]+x, u"\u3063"+hx)
        elif x[0]=="n": string = string.replace(x[0]+x, u"\u3093"+hx) # n
        string = string.replace(x, hx)
    for x,hx,kx in ja_sounds[:5]: string = string.replace(x, hx)
    return string.replace("n", u"\u3093")
def latin_katakana(string):
    findings = re.findall(u"(?:aa)|(?:ii)|(?:uu)|(?:ee)|(?:oo)", string) # [aiueo]
    for x in findings: string = string.replace(x, x[0]+u"\u30FC")
    findings = re.findall(u"(?:ww[aiueo])", string)
    for x in findings: string = string.replace(x, u"\u30A5"+x[1:])
    findings = re.findall(u"(?:ss)(?:tt)(?:pp)(?:kk)[aiueo]", string) # [stpk]
    for x in findings: string = string.replace(x, u"\u3063"+x[1:])
    findings = re.findall(u"(?:nn[aiueo])|(?:mm[aiueo])", string)
    for x in findings: string = string.replace(x, u"\u30F3"+x[1:])
    l = ja_sounds[5:]
    for x,hx,kx in ja_combosounds+l: string = string.replace(x, kx)
    for x,hx,kx in ja_sounds[:5]: string = string.replace(x, kx)
    return string
def randomstring_word(l=5, lan="en"): # word-like string
    l = max(l, 1)
    if lan=="ja": word = randomword_ja(l-1)
    elif lan=="jah": word = randomword_jah(l-1)
    elif lan=="jak": word = randomword_jak(l-1)
    else:
        soft = list(zip(*[x for x in letterfrequencies[lan] if x[0] in vowels]))
        soft = "".join(np.array(soft[0])[rng_bulk(soft[1], 100)].tolist())
        hard = list(zip(*[x for x in letterfrequencies[lan] if x[0] in consonants]))
        hard = "".join(np.array(hard[0])[rng_bulk(hard[1], 100)].tolist())
        if 0: # lan in additionalconsonants:
            special = list(zip(*[x for x in letterfrequencies[lan] if x[0] in consonants]))
            special = "".join(np.array(special[0])[rng(special[1], 100)].tolist())
        else: special = ""
        word = artificialword(l, soft, hard, special)
        for x in re.findall("[^"+vowels+"]*(["+vowels+"]{3,})[^"+vowels+"]*", word): word = word.replace(x, x[:2]) # squeeze triple vowels
    return word
def randomstring(dictionaryorlist, word=False, l=5, lan="en"):
    x = ""
    while not x or x in dictionaryorlist:
        if word: x = randomword(l, lan)
        else: x = hex(random.randint(16**l, 16**(l+1)-1))[2:]
        l += 1
    return x


def fantasyfinnish(word): # modification
    if 1:
        vowels = "aeuioöäy"
        vow_not = "[^"+vowels+"]"
        word = re.sub("sj", "s", word)
        word = re.sub("dt|td", "tt", word)
        word = re.sub("ds", "ts", word)
        word = re.sub("dp|pd", "pp", word)
        word = re.sub("pk|kp|kq|kg|qq|gg|qk|gk", "kk", word)
        word = re.sub("gs|sg", random.choice(["sk","ks"]), word)
        word = re.sub("hh|hn|hm", "h", word)
        word = re.sub("nm", "m", word)
        word = re.sub("tn|gn", "n", word)
        word = re.sub("q"+vow_not, "g", word)
        word = re.sub("m$|nn$", "nn"+random.choice("aeui"), word)
        word = re.sub("s(?:"+vow_not+"|$)", "s"+random.choice(vowels), word)
        word = re.sub("v(?:"+vow_not+"|$)", "v"+random.choice(vowels), word)
        word = re.sub(vow_not+"v", "v", word)
        word = re.sub("j(?:"+vow_not+"|$)", "j"+random.choice(vowels), word)
        word = re.sub("r(?:"+vow_not+"|$)", "r"+random.choice(vowels), word)
        word = re.sub("z(?:"+vow_not+"|$)", "s"+random.choice(vowels), word)
        word = re.sub("k$", "k"+random.choice(vowels), word)
        word = re.sub("l$", "l"+random.choice(vowels), word)
        word = re.sub("h$", "h"+random.choice(vowels), word)
        word = re.sub("it$", "it"+random.choice(vowels), word)
        word = re.sub("[pb](?:"+vow_not+"|$)", "p"+random.choice(vowels), word)
        if re.match("^"+vow_not+"{2,}.*", word):
            for x in re.findall("^"+vow_not+"{2,}", word): word = word.replace(x, x[0])
        if re.match(".*"+vow_not+"{3,}.*", word):
            for x in re.findall(""+vow_not+"{3,}", word): word = word.replace(x, x[:2])

        if re.match(vow_not+"*["+vowels+"][mn]["+vowels+"]"+vow_not+"*", word):
            for x in re.findall(vow_not+"*(["+vowels+"][mn]["+vowels+"])"+vow_not+"*", word):
                word = word.replace(x, random.choice([x[0]+x, x[0]+x[1]+x[1:]]))
                
        if re.match("a"+vow_not+"ö|o"+vow_not+"ä", word):
            for x in re.findall("a"+vow_not+"ö|o"+vow_not+"ä", word): word = word.replace(x, str("ä" if x[0]=="a" else "ö")+x[1:])
        if re.match("ö"+vow_not+"[ao]|ä"+vow_not+"[ao]", word):
            for x in re.findall("ö"+vow_not+"o|ä"+vow_not+"a", word): word = word.replace(x, x[1:]+str("ä" if x[-1]=="a" else "ö"))

        word = re.sub("iji|jj", "j", word)
        word = re.sub("ia", "iä", word)
        word = re.sub("eö", "eo", word)
        word = re.sub("o[eö]", "oo", word)
        word = re.sub("ö[aeo]", "öö", word)
        word = re.sub("oä", "oi", word)
        word = re.sub("aä|äa", random.choice(["ää","aa"]), word)
        word = re.sub("äö|öä", random.choice(["öö","ää"]), word)

        pitch = 0 # total
        for x in word: # -1 ouae, +1 iöäy
            if x in "ouae": pitch -= 1
            elif x in "iöäy": pitch += 1
        if pitch<-1: word = word.replace("ö", "o").replace("ä", "a")
        elif pitch>1: word = word.replace("o", "ö").replace("a", "ä")
        
    for x in re.findall("[^"+vowels+"]*(["+vowels+"]{3,})[^"+vowels+"]*", word): word = word.replace(x, x[:2]) # squeeze triple vowels
    return word



def randomword(l, lan="en"):
    l = max(int(l), 1) if l else 1
    def get_syllables():
        if l>0 and (s:=getattr(syllables, lan, None)): return "".join(dictrng(s, l))
        return ""
    if lan=="fi": return fantasyfinnish(get_syllables())
    if lan=="en": return get_syllables()
    return randomstring_word(l, lan)






def adj_adv(x): # adverb from adjective
    if re.match(".*ic$", x): return re.sub("ic$", "ically", x)
    if re.match(".*used$", x): return x
    if re.match(".*y$", x): return re.sub("y$", "ily", x)
    if re.match(".*[bt]le$", x): return re.sub("[bt]le$", "bly", x) # acceptable -> acceptably
    if re.match(".*ll$", x): return None
    return x+"ly" # ous/ing/ane
def adv_adj(x): # adjective from adverb
    if re.match(".*ically$", x): return re.sub("ically$", "ic", x)
    if re.match(".*used$", x): return x
    if re.match(".*ily$", x): return re.sub("ily$", "y", x)
    if re.match(".*[bt]ly$", x): return re.sub("[bt]ly$", "ble", x)
    if re.match(".*ll$", x): return None
    return re.sub("ly$", "", x)




def imperial_to_SIvalues(string):
    im_len = r"(?:th|mil)|(?:in|\″)|hh|(?:ft|\′)|yd|ch|fur|mi|lea|ftm|nmi"
    im_len_conversion = { # -> meters
        "th": 0.0000254,
        "mil": 0.0000254,
        "in": 0.0254,
        "″": 0.0254,
        "hh": 0.1016,
        "ft": 0.3048,
        "′": 0.3048,
        "yd": 0.9144,
        "ch": 20.1168,
        "fur": 201.168,
        "mi": 1609.344,
        "lea": 4828.032,
        }
    im_vol = r"fl\soz|gi|pt|qt|gal"
    im_vol_conversion = { # -> liters
        "fl oz": .0284130625,
        "gil": 0.1420653125,
        "pt": 0.56826125,
        "qt": 1.1365225,
        "gal": 4.54609,
        }
    im_mass = r"gr|dr|oz|lb|st|qr|qtr|cwt|t"
    im_mass_conversion = { # -> grams
        "gr": 0.06479891,
        "dr": 1.7718451953125,
        "oz": 28.349523125,
        "lb": 453.59237,
        "st": 6350.29318,
        "qt": 12700.58636,
        "qtr": 12700.58636,
        "t": 1016046.9088,
        }
    im_re = rf"(((?:(?:\d*)\.(?:\d*))?(?:\d*\.\d*)|(?:\d+))\s?({im_len}|{im_vol}|{im_mass}))"
    fs = re.findall(im_re, string)
    newstring = string
    for x in fs:
        x, value, unit = x
        value = float(value)
        meters = value*im_len_conversion.get(unit, 0)
        if meters: newstring = newstring.replace(x, f"{meters}m", 1)
        else:
            liters = value*im_vol_conversion.get(unit, 0)
            if liters: newstring = newstring.replace(x, f"{liters}l", 1)
            else:
                grams = value*im_mass_conversion.get(unit, 0)
                if grams: newstring = newstring.replace(x, f"{grams}g", 1)
    return newstring

def SIvalues(string):
    string = imperial_to_SIvalues(string)
    si_re = r'((?:(?:\d*)\.(?:\d*))?(?:\d*\.\d*)|(?:\d+))\s?(?:([YZETGMkhdcmμnpfazy]|da|\s|-)?((?i:rad|sr|Hz|Pa|Wb|lm|lx|Bq|Gy|Sv|kat|-C|mol|cd|[NJWCVFΩSTH]|[smglKA])\b))'
##    fs = 
    units = {}
    if fs:=re.findall(si_re, string):
        p0, p1, p2, p3 = "dc", "mμnpfazy", ["da","h"], "kMGTEZY"
        for x in fs:
            value, prefix, unit = x
            if prefix:
                if prefix in p0: prefix = 10**(-(p0.index(prefix)+1))
                elif prefix in p1: prefix = 10**(-3*(p1.index(prefix)+1))
                elif prefix in p2: prefix = 10**((p2.index(prefix))+1)
                elif prefix in p3: prefix = 10**(3*(p3.index(prefix)+1))
                else: prefix = 1
            else: prefix = 1
            if not unit in units: units[unit] = []
            units[unit].append(float(value)*prefix)
    return units

def seconds(x):
    if type(x)==str:
        t_re = r"((?:(?:\d*)\.(?:\d*))?(?:\d*\.\d*)|(?:\d+))\s?([adhsw]|ms?)"
        t = 0
        for value, unit in re.findall(t_re, x):
            m = 1
            match unit:
                case "a": m = 3600*24*365
                case "w": m = 3600*24*7
                case "d": m = 3600*24
                case "h": m = 3600
                case "m": m = 60
                case "ms": m = 1/1000
                case "µs": m = 1/1000000
            t += float(value)*m
        return t
    elif type(x) in [float,int]: return f"{x}s"




def progress(x, y, l=10, chars="#=- "):
    scale = max(y/l, 1)
    h = int(x/scale)
    i = max(int(len(chars)*(1-(x/scale-h)))-1, 0)
    return str(chars[0]*h+chars[i%len(chars)]+chars[-1]*(int(y/scale)-h))[:l]
def int_progress(x, y, l=10, chars="x "):
    scale = max(y/l, 1)
    h = int(x/scale)
    i = int(10*(x/scale-h))
    return str(chars[0]*h+str(i if i else chars[-1])+chars[-1]*(int(y/scale)-h))[:l]





def romannumerals(x): # VL -> 45 -> VL
    if type(x)==int:
        t = ""
        for char,val in [("M", 1000),("D", 500),("C", 100),("L", 50),("X", 10),("V", 5),("I", 1)]:
            i = int(x/val)
            x -= i*val
            t += char*i
        # shortening -> XXXXV = VL
        t = t.replace("C"*4, "CD")
        t = t.replace("X"*4+"V", "VL")
        t = t.replace("X"*4, "XL")
        t = t.replace("I"*4, "IV")
        return t
    elif type(x)==str:
        d = {
            "M": 1000,
            "D": 500,
            "C": 100,
            "L": 50,
            "X": 10,
            "V": 5,
            "I": 1,
            }
        do = list(d.keys())
        p = 0
        v = [0 for _ in range(len(do))]
        for xx in list(x):
            if not xx in do: return False
            newp = do.index(xx)
            v[newp] += d[xx]
            if p > newp: 
                v[newp] -= v[p]
                v[p] = 0
            p = newp
        return sum(v)




def SIinteger(x, l=3, unit="", space=True):
    if type(x) in [int,float]:
        if abs(x)>1:
            y = len(str(round(x)))-1
            if x<0:
                y -= 1
                l += 1
            lx = min(int(y/3), 7)
            if lx>=1:
                x = x*10**(-3*lx) # 10 000 -> 10k
                sx = str(round(x))
                if len(sx) > (l+1 if x < 0 else l): lx = min(lx+1, 7)
                elif len(sx) < (l if x<0 else l-1): sx = str(round(x,1))
                return sx+" "*space+"kMGTEZY"[lx-1]+unit
        elif m:=re.match(r"(?:\d*\.\d+e\-(\d+))|(?:\d*\.(0*)[^0e]+)", str(x)):
            if m.group(1): lx = int((int(m.group(1))-1)/3)
            else: lx = int(len(m.group(2))/3)
            lx = min(lx, 3)
            x = x*10**(3*(lx+1))
            sx = str(round(x))
            if len(sx)>(l+1 if x<0 else l): lx = min(lx+1, 7)
            elif len(sx)<(l if x<0 else l-1): sx = str(round(x,1))
            return sx+" "*space+"mµnp"[lx]+unit
        return str(round(x))+unit
    elif type(x)==str:
        if m:=re.fullmatch(r'(?:(?:\d*\.\d*)?(\d*\.\d*)|(\d+))(?:(?:\s?([kMGTEZY]))|(?:\s?([mµnp])))?.*', x):
            if m.group(1): x = float(m.group(1))
            if m.group(2): x = int(m.group(2))
            if m.group(3): x *= 10**(3*(list("kMGTEZY").index(m.group(3))+1))
            if m.group(4): x *= 10**(-3*(list("mµnp").index(m.group(4))+1))
        elif h:=re.fullmatch(r'0x(\d|[abcdef])+', x): x = int(h.group(0), 16)
    return x

def integer(x, l=3, space=False):
    if type(x) in [int,float]:
        y = len(str(round(x)))-1
        if x<0:
            y -= 1
            l += 1
        lx = min(int(y/l), 4)
        if lx>=1:
            x = x*10**(-3*lx) # 10 000 -> 10K
            sx = str(round(x))
            if len(sx)>(l+1 if x < 0 else l): lx = min(lx+1, 7)
            elif len(sx)<(l if x<0 else l-1): sx = str(round(x,1))
            return sx+" "*space+"KMBT"[min(lx-1, 3)]
        return str(round(x))
    elif type(x)==str:
        if m:=re.fullmatch(r'(?:(?:\d*\.\d*)?(\d*\.\d*)|(\d+))(?:\s?([KMBT]))?.*', x):
            if m.group(1): x = float(m.group(1))
            if m.group(2): x = int(m.group(2))
            if m.group(3): x *= 10**(3*(list("KMBT").index(m.group(3))+1))
        elif h:=re.fullmatch(r'0x(\d|[abcdef])+', x): x = int(h.group(0), 16)
        x = int(x)
    return x




def fraction(x, y):
    # 2070, 00B9, 00B2, 00B3, 2074, 2075, 2076, 2077, 2078, 2079
    # 2080, ...
    def upper(xx): return (r"\u00B9" if xx=="1" else r"\u00B"+xx) if xx in "123" else r"\u207"+xx
    x = "".join([upper(xx) for xx in str(x)])
    y = "".join([r"\u208"+xx for xx in str(y)])
    x = x+r'\u2044'+y
    return eval("u'"+x+"'")





ak_re_booleans = r"(True|False)"
ak_re_strings = r"(?:\"([^\"]+)\")|(?:\'([^\']+)\')|(?:([^\s\d]+\.\S*|\S*\.[^\s\d]+))"
ak_re_unknowns = r"([^\s,]+)"
ak_re_floats = r"(\-?(?:(?:\d+\.\d*)|(?:\d*\.\d+))(?:e[\-\+]\d+)?)"
ak_re_integers = r"(\-?\d+)\b" # [\s,$\(\)] (?:[\s,]|\D|[A-Z])? # (?:[\s,$\(\)])
ak_re_ec = r"(_{\d+})"
ak_re_args = re.compile(r"(?:,\s?|\s)?(?:"+ak_re_ec+r")|(?:"+ak_re_booleans+r")|(?:"+ak_re_strings+r")|(?:"+ak_re_floats+r")|(?:"+ak_re_integers+r")|(None)|(?:"+ak_re_unknowns+r")(?:\s*$|,\s?|\s)")
ak_re_strings = re.compile(r"(?:\"[^\"]+\")|(?:\'[^\']+\')")
ak_re_tuples = re.compile(r"(?:[^\(\)]*)(\([^\(\)]*\))(?:[^\(\)]*)")
ak_re_kwargs = re.compile(r"(,\s?|\s)?([^\=\s]+\=[^\=]+)(?:\s*$|,\s?|\s)")
ak_re_floats = re.compile(ak_re_floats)
ak_re_integers = re.compile(r"[^\.\d]?(\-?\d+)[^\.\d]") # [^\.\d]\b

def args_kwargs(t, **embedded):
    # solve strings from t
    while 1:
        t_i = len(embedded)
        if not (l:=ak_re_strings.findall(t)): break # deepest first
        for i,x in enumerate(l):
            if x:
                embedded_code = "_{"+str(t_i)+str(i)+"}"
                t = t.replace(x, embedded_code, 1)
                embedded[embedded_code] = x[1:-1]
    # solve tuples from t
    while 1:
        t_i = len(embedded)
        if not (l:=ak_re_tuples.findall(t)): break # deepest first
        for i,x in enumerate(l):
            if x:
                embedded_code = "_{"+str(t_i)+str(i)+"}"
                t = t.replace(x, embedded_code, 1)
                embedded[embedded_code] = args_kwargs(x[1:-1], **embedded)[0]
    kwargs = {}
    keys = ''
    values = ''
    for x in ak_re_kwargs.findall(t): # (?:\s[^\=\s]+\=.+)
        k,v = x[1].split("=", 1)
        keys += " '"+k+"'"
        values += " "+v
        t = t.replace(x[0]+x[1], "", 1)
    else:
        if keys and values:
            keys = args_kwargs(keys, **embedded)[0]
            values = args_kwargs(values, **embedded)[0]
            for i,k in enumerate(keys): kwargs[k] = values[i]
    args = []
    for ec,b,sss,ss,s,f,i,nan,u in ak_re_args.findall(t):
        if ec and ec in embedded:
            d = embedded[ec]
            del embedded[ec]
            args.append(d)
        elif b: args.append(b=="True")
        elif f: args.append(float(f))
        elif i: args.append(int(i))
        elif nan: args.append(None)
        elif u: args.append(str(u))
        else: args.append(s+ss+sss)
    return tuple(args), kwargs




def float_accuracy(string):
    accuracy = None
    for f in ak_re_floats.findall(string):
        acc = len(f.split(".", 1)[1])
        if accuracy!=None: accuracy = min(accuracy, acc)
        else: accuracy = acc
    return accuracy

def int_accuracy(string):
    accuracy = None
    for i in ak_re_integers.findall(string):
        # remove trailing zeros and count meaningful digits
        acc = len(i.rstrip("0"))-1
        if accuracy!=None: accuracy = min(accuracy, acc)
        else: accuracy = acc
    return accuracy









def read_morse(t):
    def decode(s):
        l = ""
        for i,x in enumerate(s):
            if not x in ".-": break
            x = x=="."
            if not l: l = "e" if x else "t"
            else:
                match l:
                    case "e": l = "i" if x else "a"
                    case "i": l = "s" if x else "u"
                    case "s": l = "h" if x else "v"
                    case "u": l = "f" if x else ""
                    case "a": l = "r" if x else "w"
                    case "r": l = "l" if x else ""
                    case "w": l = "p" if x else "j"
                    case "t": l = "n" if x else "m"
                    case "n": l = "d" if x else "k"
                    case "d": l = "b" if x else "x"
                    case "k": l = "c" if x else "y"
                    case "m": l = "g" if x else "o"
                    case "g": l = "z" if x else "q"
                    case _: break
        return l.upper()
    tt = ""
    mem = ""
    for x in t:
        if x in ".-": mem += x
        else: # save
            tt += decode(mem)
            mem = ""
    else: tt += decode(mem)
    return tt

def write_morse(t):
    def encode(x):
        match x.upper():
            case "E": return "."
            case "I": return ".."
            case "U": return "..-"
            case "F": return "..-."
            case "S": return "..."
            case "V": return "...-"
            case "H": return "...."
            case "A": return ".-"
            case "R": return ".-."
            case "L": return ".-.."
            case "W": return ".--"
            case "P": return ".--."
            case "J": return ".---"
            case "T": return "-"
            case "N": return "-."
            case "D": return "-.."
            case "B": return "-..."
            case "X": return "-..-"
            case "K": return "-.-"
            case "C": return "-.-."
            case "Y": return "-.--"
            case "M": return "--"
            case "G": return "--."
            case "O": return "---"
            case "Q": return "--.-"
            case "Z": return "--.."
    l = []
    for x in t:
        x = encode(x)
        if x: l.append(x)
    return " ".join(l)






def base_email(email:str):
    if "@" in email:
        name, domain = email.split("@", 1)
        if "." in domain:
            name = re.match(r"^([^\+]+)(\+.*)?$", name).group(1)
            return "@".join((name.replace(".", ""), domain))


if __name__ == "__main__":
##    for x in ["a",1,"1","2.1.3","5.123"]:
##        print(x)
##        print(is_number(x))
##        print(is_a_value(str(x)), end="\n"*2)
    
##    print(base_email("jyj..o+ol10@gmail.com"))
    
##    t = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
##    print(t)
##    t = write_morse(t)
##    print(t)
##    t = read_morse(t)
##    print(t)
##    t = write_morse(t)
##    print(t)
    
##    f = open("textprc_syllables.txt", "w")
##    f.write("finnish = {\n")
##    sylls = syllableload("textprc_syllables/fi.txt")
##    for k,v in sylls.items(): f.write(f"\t'{k}': {v},\n")
##    f.write("\t}\n")
##    
##    f.write("english = {\n")
##    sylls = syllableload("textprc_syllables/en.txt")
##    for k,v in sylls.items(): f.write(f"\t'{k}': {v},\n")
##    f.write("\t}\n")
##
##    f.close()

    
##    print(randomword(3, "la"))
    
##    print(args_kwargs("asd.asd .20"))
    
##    print(int_accuracy("2102000*11100.10000010"))
##    print(float_accuracy("2*1.22"))
##    print(randomword(3, "fi"))
    
##    print(SIinteger(0.0552, 4, "B"))

##    string = ".1 True string (\"a('asd')sd\") 2.244e-9 ((0,0), 'asdasd') 'asd' z=True a=0 b=(90,50) c=2."
##    string = "00b62465177623264325432746c5448563365544a4c5531464765545a3553454652636c46506133683259544e556330784465456c3657567032617a303d0a"
##    print(string)
##    args, kwargs = args_kwargs(string)
##    print(args, kwargs)

    
##    x = integer(51210, 4)
##    y = integer(30000000)
##    print(x, y)
##    x = integer(x, 4)
##    y = integer(y)
##    print(x, y)
##    x = integer(x, 4)
##    y = integer(y)
##    print(x, y)
    
##    y = romannumerals(45)
##    print(y)
    
##    for i in range(10):
##        x = int(2000*random.random()+1)
##        y = romannumerals(x)
##        print(x, y, romannumerals(y))
    
##    for i in range(101):
##         print(int_progress(i,1000,50,"x "))
##    string = "10mi+1le--->1gal"
##    string = imperial_to_SIvalues(string)
##    print(seconds("50a"))
    
##    while 1:
##        for i in range(1,6):
##            print(randomsyllables(5, "en"))
##            pass
##        print(randomsentence())
##        print(randomadjective()) # "^a.{6,}$"
##        print(randomverb()) # "^a.{6,}$"
        
    
##    t = randomword(8, "ja")#"awwaemmeinnioppoukku"# # manawweeku
##    print(t)
##    t = alpha2hiragana(t)
####    t = alpha2katakana(t)
##    print(t)
##    print(japanese2alpha(t))
    
    # unicode
####    while 1:
####        x = urand(2)
####        x = u2c(x)
####        y = c2u(x)
####        z = u2c(y)
####        print(x, y, z)
        
##    print(u"asd\u001Eas\u001Ed")
        
##    print(artificialword(5))
##    for i in range(5): print(seededstring("asd    ", 5))
    
##    print(sentences("This is the first sentence, I guess... I hope. What was that"))

    
##    v = 0
##    for i in range(100):
##        v += random.randint(50, 200)
##        print(valuestring(v))

##    x = SIvalues("90m asd 5Mg 90 cHz")
##    print(x)
    
    pass
