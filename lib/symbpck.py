import multiprocessing as mp

def single_send(pipe, f, *args, **kwargs): pipe.send(f(*args, **kwargs))
class single():
    def __init__(self):
        self.pipe = None
        self.process = None
        self.active = False
        self.done = False
        self.output = None
    def start(self, f, *args, **kwargs): # over a second setup time
        self.close()
        self.active = True
        self.done = False
        self.pipe, pipe = mp.Pipe()
        self.process = mp.Process(target=single_send, args=(pipe, f, *args), kwargs=kwargs)
        self.process.start()
    def poll(self):
        if self.active and self.pipe.poll():
            self.output = self.pipe.recv()
            self.done = True
            self.close()
    def close(self):
        if self.active:
            self.active = False
            self.pipe.close()
            self.process.terminate() # alot faster than .join()
    def waitclose(self): # wait to complete
        if self.active: self.output = self.pipe.recv()
        self.active = False
        self.process.join()
        self.pipe.close()

def multi_client(f, pipe, breakcode):
    done = False
    while not done:
        out = []
        l = pipe.recv()
        if l==breakcode: break
        for x in l:
            if x==breakcode:
                done = True
                break
            attr, args, kwargs = x
            ff = f
            if hasattr(ff, attr):
                for a in attr.split("."):
                    if a: ff = getattr(ff, a)
            o = ff(*args, **kwargs)
            out.append(o)
        pipe.send(out)
    if hasattr(f, "close"): f.close()
    pipe.close()
class multi():
    class process():
        process = None
        pipe = None
        breakcode = "exit"
        busy = False
        def __init__(self):
            self.inputs = []
            self.input_attrs = []
            self.outputs = []
    def __init__(self): self.processes = {}
        
    def new(self, name, inputclass, breakcode="exit"):
        prc = self.processes.get(name)
        if prc: prc.process.terminate()
        prc = self.process()
        p0, p1 = mp.Pipe()
        prc.process = mp.Process(target=multi_client, args=(inputclass, p1, breakcode))
        prc.process.start()
        prc.pipe = p0
        prc.breakcode = breakcode
        prc.busy = False
        self.processes[name] = prc

    def isempty(self, name): return not self.processes[name].inputs
    def isbusy(self, name): return self.processes[name].busy

    def add(self, name, attr, *args, **kwargs):
        self.processes[name].inputs.append((attr, args, kwargs))
        self.processes[name].input_attrs.append(attr)
    def get(self, name):
        prc = self.processes[name]
        if prc.outputs:
            out = prc.outputs.pop(0)
            attrs = prc.input_attrs[:len(out)]
            prc.input_attrs = prc.input_attrs[len(out):]
            return list(zip(attrs, out))
        
    def update(self, k=None):
        if k==None:
            for k in self.processes.keys(): self.update(k)
        elif k in self.processes:
            prc = self.processes[k]
            if prc.process.exitcode==None:
                if prc.busy and prc.pipe.poll():
                    prc.outputs.append(prc.pipe.recv())
                    prc.busy = False
                if prc.inputs and not (prc.busy or prc.pipe.poll()):
                    prc.pipe.send(prc.inputs)
                    prc.inputs.clear()
                    prc.busy = True
    
    def join(self, k=None):
        if k==None:
            for k,prc in list(self.processes.items()): self.join(k)
        elif k in self.processes:
            prc = self.processes[k]
            if prc.pipe.poll(): prc.pipe.recv()
            prc.pipe.send(prc.breakcode)
            prc.process.terminate()
            del self.processes[k]
            print(k, "terminated")

class asd():
    def asdasd(self, x): return x*2
    def asdasd2(self, x): return x*.5






if __name__ == "__main__":
    
    mp.freeze_support() # so that exes wont freeze up

##    m = multi()
##    m.new("asd", asd())
##    m.add("asd", "asdasd", 2)
##    m.add("asd", "asdasd", 3)
##    m.add("asd", "asdasd", 5)
##    l = []
##    while len(l)<10:
##        m.update()
##        x = m.get("asd")
##        if x:
##            for xx in x: m.add("asd", "asdasd2", xx)
##            l.append(x)
##    m.join()
    
##    m = multi()
##    m.new("asd", asd())
##    m.add("asd", "asdasd", 2)
##    m.update()
##    m.join("asd")
    
##    pipeend0, pipeend1 = mp.Pipe()
##    print(pipeend1.poll())
##    pipeend0.send(2)
##    print(pipeend1.poll())
##    print(pipeend1.recv())
##    print(pipeend1.poll())

    pass
