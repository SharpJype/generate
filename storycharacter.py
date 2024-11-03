import sys, os
import random



whats = [
    "search","time",
    "errand","summoning",
    "invitation","whisper",
    "shout","talk","words",
    "interruption","chase",
    "fall","stress",
    "climb","catch",
    "release","argument",
    "negotiation","plot",
    "plead","reward",
    "threat","gratitude",
    "praise","scolding",
    "persuasion","ridicule",
    "scare","flight",
    "retreat","approach",
    "fight","witness",
    "crime","suspicion",
    "command","lead",
    "trade","break",
    "attack","defence",
    "accussation","comfort",
    "dismissal","expectation",
    "lesson","promise",
    "obligation","void",
    "betrayal","help",
    "gift","theft",
    "pain","injury",
    "find","loss",
    "meeting","gaze",
    "judgement","decision",
    "approval","rejection",
    "noise","feeling",
    "touch","rest",
    "dream","thought",
    "unaware","point",
    "cut","force",
    "life","revenge",
    "death","paranoia",
    "bravery","wrath",
    "pride","sloth",
    "greed","lust",
    "indecision","freedom",
    "connection","envy",
    "thrill","sacrifice",
    "journey","triumph",
    "fighter","crack",
    "twist","wound",
    "crush","advantage",
    "stare","watch",
    "scrape","hollow",
    "damage","trickery",
    "attempt","protection",
    "honour","smear",
    "intent","accident",
    ]
prepositions = [ # other connecting words
    "over",
    "above",
    "across",
    "at",
    "behind",
    "below",
    "between",
    "by",
    "beside",
    "next to",
    "from",
    "in",
    "into",
    "in front of",
    "near",
    "on",
    "onto",
    "through",
    "to",
    "towards",
    "under",
    "before",
    "after",
    "from",
    "during",
    "for",
    "until",
    "since",
    "and",
    ]
wheres = [
    "cave",
    "cliff",
    "coast",
    "island",
    "peninsula",
    "beach",
    "ocean",
    "lagoon",
    "atoll",
    "sound",
    "marsh",
    "channel",
    "waterfall",
    "jungle",
    "isthmus",
    "strait",
    "basin",
    "bay",
    "sea",
    "cape",
    "lake",
    "forest",
    "valley",
    "hill",
    "field",
    "plain",
    "prairie",
    "river",
    "dune",
    "mesa",
    "oasis",
    "geyser",
    "glacier",
    "tundra",
    "volcano",
    "archipelago",
    "plateau",
    "canyon",
    "delta",
    "butte",
    "fjord",
    "gulf",
    "swamp",
    "desert",
    "mountain",
    "peak",
    "sinkhole",
    "crater",
    "caldera",
    "boulder",
    ]
wheres_prepos = [
    "over",
    "above",
    "across",
    "at",
    "behind",
    "below",
    "between",
    "by",
##    "beside",
    "next to",
##    "from",
##    "into",
    "in front of",
    "near",
    "on",
##    "onto",
    "through",
    "to",
    "towards",
    "under"
    ]
wheres_room = [ # ??? is in a room -> preposition + where
    "furniture",
    "window",
    "door",
    "wall",
    "ceiling",
    "floor",
    "corner",
    ]
wheres_outside = [ # ??? is outside -> preposition + where
    "forest",
    "field",
    "desert",
    "hill",
    "mountain",
    "water",
    ]

expressions = [
    "in a gathering",
    "in private",
    "in public",
    "in the open",
    "while travelling",
    ]

def title(n=3):
    def prepos():
        choices = whats+feeling+target+wheres_room+wheres_outside+expressions
        pre = random.choice(prepositions)
        if pre == "between": where = random.choice(choices)+" and "+random.choice(choices)
        else: where = random.choice(choices)
        return pre+" "+where
    
    return " -> ".join([random.choice(whats)+" "+prepos() for _ in range(n)])


def title(n=3):
    def prepos():
        choices = whats+feeling+target+wheres_room+wheres_outside+expressions
        pre = random.choice(prepositions)
        if pre == "between": where = random.choice(choices)+" and "+random.choice(choices)
        else: where = random.choice(choices)
        return pre+" "+where
    
    return "".join(["\n\t"+random.choice(whats)+" "+prepos() for _ in range(n)])





def character_traits(n=3):
    ck3traits = [("Brave","Craven"),("Calm","Wrathful"),("Chaste","Lustful"),("Content","Ambitious"),("Diligent","Lazy"),
                 ("Fickle","Stubborn"),("Forgiving","Vengeful"),("Generous","Greedy"),("Gregarious","Shy"),("Honest","Deceitful"),
                 ("Humble","Arrogant"),("Just","Arbitrary"),("Patient","Impatient"),("Temperate","Gluttonous"),("Trusting","Paranoid"),
                 ("Zealous","Cynical"),("Compassionate","Callous","Sadistic")]
    t = {}
    usedck3 = []
    for i in range(n):
        if random.randint(0,1):
            xx = None
            while xx is None or xx in usedck3: xx = random.randint(0, len(ck3traits)-1)
            usedck3.append(xx)
            x = random.choice(ck3traits[xx]).lower()
        else: x = random.choice(["athletic","attractive","charming","intellectual","honorable","wealthy","strongwilled"])
        t[x] = t.get(x, 0)+random.randint(0,1)*2-1
    return "".join(["\n\t"+f"extremely "*(abs(v)>1)+"not "*(v<0)+k for k,v in t.items()])

















##whos = [xx+" "+x for xx in ["physical","romantic","social","economical","intellectual","spiritual"] for x in ["superior","equal","inferior"]]
##whos2 = [x+" "+xx for xx in ["physique","features","charm","intellect","honor","wealth","willpower"] for x in ["superior","equal","inferior"]] # ,"rights"
target = ["self","foreigner","family","relative","stranger","acquintance","friend","bestfriend","ally","adversary","nemesis","academian",
          "superior","mentor","parent","sibling","employee","outcast","outlaw","miscreant","child","teen","elder","politician","worker",
          "manager","enforcer","government","organization","group","neighbour","cheater","manipulator","worshipper","icon"]
feeling = [ # feeling toward an event or a character
    "anger",
    "disgust",
    "joy",
    "sadness",
    "surprise",
    "fear",
    "terror",
    "outrage", # anger+disgust, 
    "desperate", # anger+fear, nothing to lose
    "cruelty", # anger+joy, 
    "betrayal", # anger+sadness, 
    "horror", # disgust+fear, 
    "playful disgust", # disgust+joy,
    "embarrassment", # disgust+sadness, 
    "shock", # disgust+surprise, 
    "desperate hope", # fear+joy, 
    "devastation", # fear+sadness, 
    "faint hope", # joy+sadness, 
    "amazement", # joy+surprise,
    "disappointment", # sadness+surprise,
    "confusion",
    "empathy",
    "indifference",
    "pride",
    "attraction",
    "shame",
    "superiority",
    "jealousy",
    "envy",
    "love",
    ]

def character_feeling(n=3):
    return "".join(["\n\t"+random.choice(feeling).upper()+" -> "+random.choice(target) for _ in range(n)])



##def single_event():
##    a = random.choice(["after","during","before"])
####    b = random.choice(["physical","familial","romantic","rightful","economical","intellectual","spiritual"])
##    c = random.choice(["persuasion","approach","conflict","defeat","success"])
####    c = random.choice(hows)
##    return a+" "+c # +" "+b


def stage(n=3):
    def prepos():
        pre = random.choice(wheres_prepos)
        if pre == "between": where = random.choice(wheres_outside+wheres_room)+" and "+random.choice(wheres_outside+wheres_room)
        else: where = random.choice(wheres_outside+wheres_room)
        return pre+" "+where
    return "".join(["\n\t"+prepos() for _ in range(n)])






if __name__ == "__main__":
    while 1:
        print("TRAIT:", character_traits(3).lower())
        print("FEELS:", character_feeling(3))
        #print("WORDS:", title(3))
        x = input()
    
##    while 1:
##        print("characters:")
##        for x in range(1,2): print(f" {x}.", char(3)) # .title()
##        print("\nfeelings:")
##        for x in range(1,2): print(f" {x}.", feel(1).title())
##        print("\nconcepts:")
##        for x in range(1,2): print(f" {x}.", title(1).title())
##        x = input()
##    multiple_char(50)
##    multiple_title(50)
##    multiple_feel(50)
    pass
