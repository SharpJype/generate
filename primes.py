import sys, os
import numpy as np

sys.path.insert(0, 'lib')

import primepck
import symbpck
import timepck

def primerun(start, end, folder=".", prefix="", printing=False): # test one by one from n->
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"{prefix}_"*bool(prefix)+"primes_")
    
    count = 0
    logfile_count = 0
    maxcount_per_logfile = int(1e6)
    logfile = open(filepath+f"{logfile_count}.txt","w", encoding="ascii")
    logfile.write(f"range: {start}to{end}\n")

    if printing:
        distance = end-start
        progress_reports = 100
        progress_report_count = 0
    
    for prime in primepck.find_primes(start, end):
        if count>=maxcount_per_logfile:
            logfile.close()
            count = 0
            logfile_count += 1
            logfile = open(filepath+f"{logfile_count}.txt","w", encoding="ascii")
            logfile.write(f"range: {start}to{end}\n")
        logfile.write(str(prime)+"\n")
        if printing:
            if ((prime-start)/distance)>=(progress_report_count+1)/progress_reports:
                progress_report_count += 1
                print(f"{progress_report_count:3d} %", end="\r")
        count += 1
    logfile.close()
    if printing: print("100 %")


def range_to_subranges(start, end, parts):
    diff = end-start
    arange = 1+np.arange(parts)
    a = np.flip((arange*arange+arange)/2)
    array = start+np.cumsum(a*(diff/a.sum()))
    yield int(start), int(array[0])
    for i in range(1, parts-1): yield int(array[i-1]), int(array[i])
    yield int(array[-2]), int(end)


def multi_thread(start, end, threads=4):
    symbpck.mp.freeze_support() # if made into .exe
    if start>=end or end<0: return
    
    basename = __file__.rsplit(os.path.sep, 1)[1]
    filename = basename.rsplit(".", 1)[0]
    
    threads_list = []
    for i in range(threads): threads_list.append(symbpck.single())

    length = (end-start)
    per_thread = length//threads
    
    for i,subrange in enumerate(range_to_subranges(start, end, threads)):
        path = os.path.join(".", filename)
        threads_list[i].start(primerun, subrange[0], subrange[1], path, f"multi{i}")

    i = 0
    is_done = np.zeros(threads, dtype=np.bool_)
    print(f"{0:3d} % {is_done.astype(np.uint8)}", end="\r")
    while 1:
        if not is_done[i]:
            threads_list[i].poll()
            if threads_list[i].done:
                is_done[i] = True
                percent = int((is_done.sum()/is_done.shape[0])*100)
                print(f"{percent:3d} % {is_done.astype(np.uint8)}", end="\r")
        if is_done.all(): break
        i += 1
        if i%threads==0:
            i = 0
            timepck.sleep(1)
    print("100 %")
    
if __name__ == "__main__":
##    for subrange in range_to_subranges(1000, 2000, 2):
##        print(subrange)
    
    a = int(float(input("start number: ")))
    b = int(float(input("end number: ")))
    threads = int(float(input("amount of threads: ")))
    print("")
    print(*timepck.datetime())
    if threads>1: multi_thread(a, b, threads=threads)
    else:
        basename = __file__.rsplit(os.path.sep, 1)[1]
        path = os.path.join(".", basename.rsplit(".", 1)[0])
        primerun(a, b, path, "singlethread", printing=True)
    print(*timepck.datetime())
    input("\ndone!")
    pass
