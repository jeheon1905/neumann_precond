import torch
import torch.multiprocessing as mp
import numpy as np
import time
import threading
import os

from multiprocessing.pool import ThreadPool
from itertools import cycle

def thread_mm2(arg):
    X, A, rank, bsz, world_size = arg
    res = []
    for i,x in enumerate(X.split(bsz, dim=1)):
        x = x.to(f'cuda:{rank}')
        res.append((A@x).to('cpu'))
    res = torch.cat(res, dim=1)
    return res

def thread_mm(X, A=None, rank=0, q=None, bsz=100, world_size=4):
    res = []
    for i,x in enumerate(X.split(bsz, dim=1)):
        if i%world_size == rank:
            x = x.to(f'cuda:{rank}')
            res.append((A@x).to('cpu'))
    res = torch.cat(res, dim=1)
    q.put(res)


def process_mm1(A, q_in, q_out, rank, semaphore):
    
    print('pid :',os.getpid(), 'rank :',rank)
    torch.set_num_threads(4)
    A = A.to(f'cuda:{rank}')
    dev = A.device
    while True:
        idx, x = q_in.get()
        semaphore.release()
        x = x.to(dev)
        y = (A@x).to('cpu')
        q_out.put((idx, y))


def process_mm2(A, q_in, q_out, rank, semaphore, bsz=100):
    
    print('pid :',os.getpid(), 'rank :',rank)
    torch.set_num_threads(4)
    A = A.to(f'cuda:{rank}')
    dev = A.device
    while True:
        idx, X = q_in.get()
        semaphore.release()
        res = []
        for x in X.split(bsz,dim=1):
            x = x.to(dev)
            y = (A@x).to('cpu')
            res.append(y)
        res = torch.cat(res,dim=1)
        q_out.put((idx, res))


def by_multiprocessing2(A, X, num_gpus=4, batch_size=100, shard_size=2500):
    queue_size = 10
    world_size = num_gpus
    semaphore = mp.Semaphore(world_size*queue_size)
    q_in = mp.Queue()
    q_out = mp.Queue()
    procs = []
    
    for rank in range(world_size):
        p = mp.Process(target=process_mm2, 
                       args=(A,q_in,q_out,rank,semaphore),
                       kwargs={'bsz':batch_size})
        procs.append(p)
        p.start()
    
    #start = torch.cuda.Event(enable_timing=True)
    #end = torch.cuda.Event(enable_timing=True)
    #start.record()
    time_ = []
    st = time.time()
    for i in range(10):
        #start.record()
        it = time.time()
        res = []
        shards = X.split(shard_size, dim=1)
        for j, shard in enumerate(shards):
            q_in.put((j, shard))
            semaphore.acquire()
        
        for j in range(len(shards)):
            res.append(q_out.get())
        #end.record()
        #e = start.elapsed_time(end)
        #print(f'({i})\tMM_proc : {e}')
        #time_.append(e)
        time_.append((time.time()-it)*1000)
        print(f'({i})\tMM_processing : {time_[-1]}')

    print(f'MM (mean): {sum(time_[1:])/len(time_[1:])}')
    #print(f'total proc: {sum(time_[1:])/len(time_[1:])}')
    #end.record()
    #print(f'total : {start.elapsed_time(end)}')
    for p in procs:
        p.terminate()


def by_multiprocessing1(A, X, num_gpus=4, batch_size=100):
    queue_size = 10
    world_size = num_gpus
    semaphore = mp.Semaphore(world_size*queue_size)
    q_in = mp.Queue()
    q_out = mp.Queue()
    procs = []
    
    for rank in range(world_size):
        p = mp.Process(target=process_mm1, 
                       args=(A,q_in,q_out,rank,semaphore))
        procs.append(p)
        p.start()
    
    #start = torch.cuda.Event(enable_timing=True)
    #end = torch.cuda.Event(enable_timing=True)
    #start.record()
    time_ = []
    st = time.time()
    for i in range(10):
        #start.record()
        it = time.time()
        res = []
        batches = X.split(batch_size, dim=1)
        for j, x in enumerate(batches):
            q_in.put((j, x))
            semaphore.acquire()
        
        for j in range(len(batches)):
            res.append(q_out.get())
        #end.record()
        #e = start.elapsed_time(end)
        #print(f'({i})\tMM_proc : {e}')
        #time_.append(e)
        time_.append((time.time()-it)*1000)
        print(f'({i})\tMM_processing : {time_[-1]}')

    print(f'MM (mean) : {sum(time_[1:])/len(time_[1:])}')
    #print(f'total proc: {sum(time_[1:])/len(time_[1:])}')
    #end.record()
    #print(f'total : {start.elapsed_time(end)}')
    for p in procs:
        p.terminate()


def by_threading(A, X, num_gpus=4, batch_size=100):
    q_out = mp.Queue()
    ngpu = 4
    time_ = []
    time__ = []
    A_cuda = [A.to(f'cuda:{i}') for i in range(ngpu)]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    e2 = 0
    for j in range(10):  
        threads = []
        for i in range(ngpu):
            threads.append(threading.Thread(
                            target=thread_mm, 
                            args=(X,),
                            kwargs={'A':A_cuda[i], 
                                    'rank':i, 
                                    'q':q_out,
                                    'bsz':batch_size,
                                    'world_size':num_gpus}))
        
        end.record()
        e1 = start.elapsed_time(end)
        print(f'({j})\tinit + queue.get() : ',e1-e2)
        time__.append(e1-e2)
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        end.record()
        e2 = start.elapsed_time(end)
        
        for i in range(len(threads)):
            q_out.get()

        print(f'({j})\tMM_threading : ',e2-e1)
        time_.append(e2-e1)
    
    print(f'MM (mean) : {sum(time_[1:])/len(time_[1:])}')
    print(f'init+get (mean) : {sum(time__[1:])/len(time__[1:])}')


def by_threadpool(A, X, num_gpus=4, batch_size=100, shard_size=2500):
    time_ = []

    A_cuda = [A.to(f'cuda:{i}') for i in range(4)]
    #shards = X.split(shard_size, dim=1)
    shards = X.split(batch_size, dim=1)
    rank = range(num_gpus)
    bsz = [batch_size]*num_gpus
    world_size = [num_gpus]*num_gpus

    for i in range(10):
        it = time.time()
        args = zip(shards, cycle(A_cuda), cycle(rank), cycle(bsz), cycle(world_size))
        with ThreadPool(num_gpus) as p:
            out = p.map(thread_mm2, args)
        
        e = (time.time() - it)*1000
        time_.append(e)
        print(f'({i})\tMM_threadpool : ',e)

    print(f'MM (mean) : {sum(time_[1:])/len(time_[1:])}')




def by_single(A, X, batch_size=100):
    time_ = []
    A_cuda = A.cuda()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    
    for j in range(10):  
        start.record()
        
        res = []
        for x in X.split(batch_size, dim=1):
            x = x.cuda()
            res.append((A_cuda@x).to('cpu'))

        end.record()
        e = start.elapsed_time(end)
        
        print(f'({j})\tMM_single : ',e)
        time_.append(e)
    
    print(f'MM (mean) : {sum(time_[1:])/len(time_[1:])}')


if __name__ == '__main__':
    torch.set_num_threads(16)
    torch.manual_seed(0)
    
    n = 500000
    m = 1000
    batch_size = 250
    sparsity = 0.0001
    idx = torch.randint(n, (2, int(n**2 * sparsity)))
    v = torch.randn(int(n**2 * sparsity))
    A = torch.sparse_coo_tensor(idx, v, (n,n))
    X = torch.randn(n, m)
    import sys

    if sys.argv[1] == 'single':
        by_single(A, X, batch_size=batch_size)
    
    elif sys.argv[1] == 'thread':
        by_threading(A, X, batch_size=batch_size)
    
    elif sys.argv[1] == 'threadpool':
        by_threadpool(A, X, batch_size=batch_size)
    
    elif sys.argv[1] == 'multiproc1':
        mp.set_start_method('spawn')
        by_multiprocessing1(A, X, batch_size=batch_size)

    elif sys.argv[1] == 'multiproc2':
        mp.set_start_method('spawn')
        by_multiprocessing2(A, X, batch_size=batch_size)
