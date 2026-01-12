import os
import torch 
from torch import distributed as dist
import numpy as np
import random 
import time


def print_pass(*args, sep=' ', end='\n', file=None, flush=False):
    """Dummy print function for non-master ranks"""
    pass


class ParallelHelper:
    __local_rank = None
    __rank = None
    __global_size = None
    __backend = None
    __device = None

    @classmethod
    @property
    def local_rank(cls):
        if isinstance(cls.__local_rank, int) :
            return cls.__local_rank
        return 0

    @classmethod
    @property
    def rank(cls):
        if isinstance(cls.__rank, int) :
            return cls.__rank
        return 0

    @classmethod
    @property
    def global_size(cls):
        if isinstance(cls.__global_size, int) :
            return cls.__global_size
        return 1

    @classmethod
    def is_master(cls):
        if isinstance(cls.global_size, int) :
            if (cls.rank==0): return True
            else: return False
        return True

    @classmethod
    def split_size(cls, total_size, return_all=False):
        rank        = cls.rank if isinstance(cls.rank, int) else 0
        global_size = cls.global_size if isinstance(cls.global_size, int) else 1
        list_size = [ total_size//global_size + (1 if i < total_size%global_size else 0) for i in range(global_size) ]
        if(return_all): return list_size
        return list_size[rank]

    @classmethod
    def split(cls, X, dim=-1, return_all=False):
        if isinstance(cls.global_size, int)  and cls.global_size>1:

            local_rank  = cls.local_rank
            rank        = cls.rank
            global_size = cls.global_size
            
            list_size = [ X.size(dim)//global_size + (1 if i < X.size(dim)%global_size else 0) for i in range(global_size) ]
            if(return_all):
                return torch.split(X, list_size, dim = dim )
            return torch.split(X, list_size, dim = dim )[rank]
        else:
            if(return_all):
                return [X]
            return X

    @classmethod
    def merge(cls, X, dim=-1, list_size=None, return_list_size=False, async_op=False):
        if isinstance(cls.global_size, int)  and cls.global_size>1:
            rank        = cls.rank
            global_size = cls.global_size
    
            if list_size is not None:
                assert len(list_size) == global_size
            else:            
                list_size = [ torch.zeros(1, dtype=torch.int64, device=X.device) for _ in range(global_size)  ]
                my_size   = torch.tensor([X.size(dim)], dtype=torch.int64, device=X.device ) 
                dist.all_gather(list_size, my_size)
    
            max_size  = max(list_size)
    
            idx       = []
            for i, size in enumerate(list_size):
                start = i*max_size
                idx+= [start+ i for i in range(size) ] 
    
            if(X.size(dim) < max_size):
                pad_size = list(X.size())
                pad_size[dim] = max_size - X.size(dim)
                pad_X = torch.zeros(pad_size, device=X.device, dtype=X.dtype)
                X_ = torch.cat([X,pad_X], dim=dim ).contiguous()
            else:
                X_ = X.contiguous()
            output_size = list(X.size())
            output_size[dim] = max_size
            output = [torch.zeros(output_size, device=X.device, dtype=X.dtype) for _ in range(global_size) ]
            dist.all_gather(output, X_ , async_op=async_op)
            output = torch.cat(output, dim=dim)
            output= torch.index_select(output, dim, torch.tensor(idx, dtype=torch.int64, device=X.device))
            
            if return_list_size:
                return output, list_size
            return output
        else:
            if return_list_size:
                return X, [ X.size(dim) ]
            return X

    @classmethod
    def all_reduce(cls, X, op=dist.ReduceOp.SUM):
        if isinstance(cls.global_size, int) and cls.global_size>1:
            is_complex = torch.is_complex( X )
            X = X.contiguous()
            if is_complex:
                X_tmp = torch.stack([X.real,X.imag])
                dist.all_reduce(X_tmp, op=op)
                X = torch.complex(X_tmp[0], X_tmp[1])
            else:                
                dist.all_reduce(X, op=op)
        return X

    @classmethod
    def scatter(cls, X, scatter_list=None, src=0):
        """
            X: output
            scatter_list: list that should be sent
            src: rank of processor that has data
        """
        assert len(scatter_list)==cls.global_size
        if isinstance(cls.global_size, int) and cls.global_size>1:
            dist.scatter(X, scatter_list, src)
            return scattered_X
        else: 
            return X

    @classmethod
    def broadcast(cls, X, src=0, async_op=False):
        if isinstance(cls.global_size, int) and cls.global_size>1:
            is_complex = torch.is_complex( X )
            if is_complex:
                X_tmp = torch.stack([X.real,X.imag])
                dist.broadcast(X_tmp, src, async_op=async_op)
                X = torch.complex(X_tmp[0], X_tmp[1])
            else:                
                dist.broadcast(X, src, async_op=async_op)
        return X
    @classmethod
    def reduce_scatter(cls, X, op=dist.ReduceOp.SUM, dim=0):
        if isinstance(cls.global_size, int) and cls.global_size>1:
            if cls.__backend=='nccl' and X.size(dim)%cls.global_size==0:
                X_ = X.transpose(0,dim).contiguous()
                size = list(X_.size())
                size[0]=size[0]//cls.global_size
                output = torch.zeros(tuple(size), dtype=X.dtype, device=X.device)
                dist.reduce_scatter_tensor(output, X_, op=op)
                output = output.transpose(dim,0).contiguous()
            else:
                X_ = X.contiguous()
                cls.all_reduce(X_, op=op)
                output = cls.split(X_, dim)            
            return output
        else:
            return X

#    @classmethod
#    # def redistribute(cls, X, dim0=-1, dim1=-2, list_size0=None):
#    def redistribute(cls, X, dim0=-1, dim1=-2, list_size0=None, print_time=None):
#        """
#        dim0: original spliting dimension
#        dim1: target splitting dimension
#        """
#        if print_time:
#            cls.synchronize()
#            st = time.time()
#
#        if (dim0<0 ): dim0= len(X.size())+dim0
#        if (dim1<0 ): dim1= len(X.size())+dim1
#
#        is_complex = torch.is_complex( X )
#        if isinstance(cls.global_size, int)  and cls.global_size>1:
#            if 'cuda' in str(X.device):
#                torch.cuda.set_device(X.device)
#    
#            rank        = cls.rank
#            global_size = cls.global_size
#    
#            # get size of tensors in all devices
#            if list_size0 is None:
#                list_size0 = [ torch.empty(1, dtype=torch.int64, device=X.device) for _ in range(global_size)  ]
#                #my_size0   = torch.tensor([X.size(dim0)], dtype=torch.int64, device=X.device ) 
#                dist.all_gather(list_size0, torch.tensor([X.size(dim0)], dtype=torch.int64, device=X.device ) )
#            else:
#                assert len(list_size0)==global_size, f'The size of given list_size0 is not matched with global size {global_size} {len(list_size0)}'
#    
#            list_splitted_X = cls.split(X, dim=dim1, return_all=True)
#            if is_complex:
#                list_splitted_X = [ torch.stack([splitted_X.real, splitted_X.imag]) for splitted_X in list_splitted_X ]
#                dim0+=1
#                dim1+=1
#            #construct and recv tensors
#            recv_tensors =[]
#            list_recv_op =[]
#            for idx, size in enumerate(list_size0):
#                if idx==rank: 
#                    recv_tensors.append( list_splitted_X[idx] )
#                    continue
#    
#                tensor_size = list(X.size())
#                if is_complex:
#                    tensor_size = [2] + tensor_size
#                    tensor_size[dim0]=list_size0[idx]
#                    tensor_size[dim1]=list_splitted_X[rank].size(dim1)
#                    recv_tensors.append( torch.empty(tuple(tensor_size), dtype=X.real.dtype, device=X.device) )
#                else:                    
#                    tensor_size[dim0]=list_size0[idx]
#                    tensor_size[dim1]=list_splitted_X[rank].size(dim1)
#                    recv_tensors.append( torch.empty(tuple(tensor_size), dtype=X.dtype, device=X.device) )
#                list_recv_op.append( dist.P2POp(dist.irecv, recv_tensors[idx], idx) )
#    
#            #isend
#            list_send_op =[]
#            for idx, send_tensor in enumerate(list_splitted_X ):
#                if idx==rank: continue
#                list_send_op.append( dist.P2POp(dist.isend, send_tensor.contiguous(), idx) )
#            reqs = dist.batch_isend_irecv(list_send_op+list_recv_op)
#            for req in reqs: req.wait()
#            X = torch.cat(recv_tensors, dim=dim0)
#            if is_complex:
#                X = torch.complex(X[0], X[1])
#        #     return X
#        # else:
#        #     return X
#        if print_time:
#            cls.synchronize()
#            print(f"{print_time}: {time.time()-st} sec", flush=True)
#        return X

    @classmethod
    # def redistribute(cls, X, dim0=-1, dim1=-2, list_size0=None):
    def redistribute(cls, X, dim0=-1, dim1=-2, list_size0=None, print_time=None):
        """
        dim0: original spliting dimension
        dim1: target splitting dimension
        """
        if print_time:
            cls.synchronize()
            st = time.time()
        if isinstance(cls.global_size, int)  and cls.global_size>1:
            
            rank        = cls.rank
            global_size = cls.global_size
    
            # get size of tensors in all devices
            if list_size0 is None:
                list_size0 = [ torch.empty(1, dtype=torch.int64, device=X.device) for _ in range(global_size)  ]
                dist.all_gather(list_size0, torch.tensor([X.size(dim0)], dtype=torch.int64, device=X.device ) )
            else:
                assert len(list_size0)==global_size, f'The size of given list_size0 is not matched with global size {global_size} {len(list_size0)}'
    
            list_splitted_X = cls.split(X, dim=dim1, return_all=True)
            list_splitted_X = [ splitted_X.contiguous() for splitted_X in list_splitted_X]
            #construct and recv tensors
            recv_tensors =[]
            for idx, size in enumerate(list_size0):
                if idx==rank: 
                    recv_tensors.append( list_splitted_X[idx] )
                    continue
    
                tensor_size = list(X.size())
                tensor_size[dim0]=list_size0[idx]
                tensor_size[dim1]=list_splitted_X[rank].size(dim1)
                recv_tensors.append( torch.empty(tuple(tensor_size), dtype=X.dtype, device=X.device) )

            torch.distributed.all_to_all( recv_tensors, [ splitted_X.contiguous() for splitted_X in list_splitted_X])
            X = torch.cat(recv_tensors, dim=dim0)
        if print_time:
            cls.synchronize()
            print(f"{print_time}: {time.time()-st} sec", flush=True)
            
        return X
#    @classmethod
#    def second_norm(X, dim):
#        assert isinstance(cls.__rank, int), "ParallelHelper should be initialized before it is used"
#        return_val = torch.sum(X*X, dim=dim)
#        dist.all_reduce(return_val, op=dist.ReduceOp.SUM)
#        return torch.sqrt(return_val)
    @classmethod
    def barrier(cls):
        if isinstance(cls.global_size, int)  and cls.global_size>1:
            dist.barrier()
        return 

    @classmethod
    def synchronize(cls):
        """
        Synchronize all processes in the group.
        """
        if cls.__device is None:
            raise RuntimeError("ParallelHelper not initialized. Call init_from_env() first.")

        if cls.__device == torch.device("cpu"):
            return
        elif isinstance(cls.global_size, int) and cls.global_size > 1:
            torch.cuda.synchronize(cls.__device)
            dist.barrier()
        else:
            torch.cuda.synchronize(cls.__device)
        return 
                    
    @classmethod
    def device_count(cls):
        return torch.cuda.device_count()

    @classmethod
    def init_from_env(
        cls,
        use_cuda: bool = False,
        debug: bool = False,
    ) -> None:
        """
        Initialize ParallelHelper using environment variables.

        Expected environment variables:
        - LOCAL_RANK
        - RANK
        - WORLD_SIZE

        Args:
            debug (bool): If True, enable debug mode. Default is False.
            use_cuda (bool): If True, use CUDA device. Default is False.
        """
        backend = 'nccl' if dist.is_nccl_available() else 'gloo'
        cls._set(
            int(os.environ.get('LOCAL_RANK', 0)),
            int(os.environ.get('RANK', 0)),
            int(os.environ.get('WORLD_SIZE', 1)),
            backend,
            debug,
            use_cuda,
        )

    @classmethod
    def _set(cls, local_rank, rank, global_size, backend, debug=False, use_cuda=False):
        """
        Core initialization with explicit arguments.

        Args:
            local_rank (int): local GPU index
            rank (int): global rank index
            global_size (int): total number of processes
            backend (str): torch.distributed backend
            debug (bool): whether to allow print in all ranks
            use_cuda (bool): whether to use CUDA device
        """
        import builtins

        # Argument validation
        if not (isinstance(global_size, int) and global_size > 0):
            raise ValueError(f"Invalid global_size={global_size}")
        if not (isinstance(rank, int) and 0 <= rank < global_size):
            raise ValueError(f"Invalid rank={rank} for global_size={global_size}")
        if not (isinstance(local_rank, int) and 0 <= local_rank < global_size):
            raise ValueError(f"Invalid local_rank={local_rank} for global_size={global_size}")

        # Set internal variables
        cls.__local_rank = local_rank
        cls.__rank = rank
        cls.__global_size = global_size
        cls.__backend = backend

        # Set device
        if use_cuda and torch.cuda.is_available():
            assert local_rank < torch.cuda.device_count(), "local_rank exceeds available cuda devices"
            cls.__device = torch.device(f"cuda:{local_rank}")
        else:
            cls.__device = torch.device("cpu")

        # Initialize process group if necessary
        if global_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend=backend)

        # Suppress print for non-master ranks
        if not debug and not cls.is_master():
            builtins.print = print_pass

        # Print initialization summary (master only)
        if cls.is_master():
            lines = [
                "\n======================= [ ParallelHelper ] =========================",
                f"* rank/local_rank : {cls.__rank}/{cls.__local_rank}",
                f"* global_size     : {cls.__global_size}",
                f"* backend         : {cls.__backend}",
                f"* device          : {cls.__device}",
                "====================================================================\n"
            ]
            print("\n".join(lines))

    @classmethod
    def get_device(cls):
        """
        Return current device
        """
        return cls.__device


#def A_mul_B (A, B, 
#             global_index_A=None, global_index_B=None, 
#             split_dim_A=-1, split_dim_B=-1):
#
#    assert len(A.size() )== len(B.size())==2
#    B_= ParallelHelper.merge(B, split_dim_B)
#    if global_index_B is not None:
#        global_index_B = ParallelHelper.merge(global_index_B)
#        B_= torch.index_select( B_, split_dim_B, global_index_B)
#    result = ParallelHelper.merge( A@B_, dim=0)
#    if global_index_A is not None:
#        global_index_A = ParallelHelper.merge(global_index_A)
#        result = torch.index_select( result, split_dim_A, global_index_A)
#    return result 

# NOTE: the following two lines are moved to ParallelHelper.init_from_env()
# backend = 'nccl' if torch.distributed.is_nccl_available() else 'gloo'
# ParallelHelper._set(int(os.environ.get('LOCAL_RANK',0)), int(os.environ.get('RANK',0)), int(os.environ.get('WORLD_SIZE',1) ), backend, int(os.environ.get('GOSPEL_PARALLEL_DEBUG',0))==1 )
#try:
#    backend = 'nccl' if torch.distributed.is_nccl_available() else 'gloo'
#    ParallelHelper._set(int(os.environ.get('LOCAL_RANK',0)), int(os.environ.get('RANK',0)), int(os.environ.get('WORLD_SIZE',1) ), backend, int(os.environ.get('ParallelDebug',0))==1 )
#
#except KeyError:
#    s = "\n======================= [ ParallelHelper ] ========================="
#    s += f"\n* NOT USED"
#    s += "\n====================================================================\n"
#    print(s)
#    pass

################################################
# seed =os.environ.get("GOSPEL_RANDOM_SEED", 1) + ParallelHelper.rank
# seed = int(os.environ.get("GOSPEL_RANDOM_SEED", 1)) + ParallelHelper.rank
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# torch.backends.cudnn.benchmark =False   # potential error source
# torch.backends.cudnn.deterministic=True # potential error source
# random.seed(seed)
################################################
#import socket
#import subprocess
#print(socket.gethostname())
#print(subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip())

def main():
    import torch 
    import random
    import numpy  as np
    import time 

    seed =2
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark =False
    torch.backends.cudnn.deterministic=True
    random.seed(seed)

    ParallelHelper.init_from_env(debug=True, use_cuda=True)

    device = torch.device("cuda:"+str(ParallelHelper.local_rank))
    #print('0) local_rank', ParallelHelper.local_rank )
    A = torch.rand(1000000,400//ParallelHelper.global_size).double()
    A = A.to(device)

    st = time.time()
    for i in range(30):
        A_redist = ParallelHelper.redistribute(A)
    et = time.time()
    print(et-st)
#    exit(-1)
#
#    #A = (A + A.T).to(device) #symmetrize
#    A = ParallelHelper.all_reduce(A) / ParallelHelper.global_size
##    X = torch.rand(100,3).double().to(device)
##    X = ParallelHelper.all_reduce(X) / ParallelHelper.global_size
##    from gospel.Eigensolver.ParallelDavidson2 import davidson as p_davidson 
##    from gospel.Eigensolver.Davidson import davidson as s_davidson
##    #print(A)
##
##    if ParallelHelper.rank==0:
##        try:
##            print( s_davidson( A, X, verbosityLevel=1, maxiter=200) )
##        except AttributeError:
##            pass
##    print( p_davidson( A, X, verbosityLevel=(ParallelHelper.rank==0), maxiter=200) )
#    for i in range(ParallelHelper.global_size):
#        if(ParallelHelper.rank==i):
#            print(A)
#        ParallelHelper.barrier()
#    A_=ParallelHelper.reduce_scatter(A, dim=1)
#    for i in range(ParallelHelper.global_size):
#        if(ParallelHelper.rank==i):
#            print(A_)
#        ParallelHelper.barrier()
if __name__=="__main__":
    main()
#    local_rank  = 1
#    rank        = 1
#    global_size = 4
#    X =torch.randn(10,5)
#    dim=-1
#    list_size = [ X.size(dim)//global_size + (1 if i < X.size(dim)%global_size else 0) for i in range(global_size) ]
#    print([ torch.split(X, list_size, dim = dim )[i].size() for i in range(global_size)] )
