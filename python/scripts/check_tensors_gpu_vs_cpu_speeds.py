import torch
import numpy as np
# torch Tensor on CPU
import time
import numpy as np



def exp_numpy():
    x = np.random.rand(1, 1000)
    y = np.random.rand(5000, 1000)
    #Exp1
    ticks = time.time()
    for i in range(1,1000):
        z = (x * y).sum()
    tacks = time.time()
    diff = tacks - ticks
    print("Exp: Numpy Array: time={}".format(diff))

def exp_pytorch_cpu():
    x = torch.rand(1, 1000)
    y = torch.rand(5000, 1000)
    #Exp1
    ticks = time.time()
    for i in range(1,1000):
        z = (x * y).sum()
    tacks = time.time()
    diff = tacks - ticks
    print("Exp: PyTorch tensor CPU: time={}".format(diff))

def exp_pytorch_gpu_move_in_loop():
    x = torch.rand(1, 1000)
    y = torch.rand(5000, 1000)
    #Exp1
    ticks = time.time()
    for i in range(1,1000):
        x = x.to('cuda')
        y = y.to('cuda')
        z = (x * y).sum()
    tacks = time.time()
    diff = tacks - ticks
    print("Exp: PyTorch tensor GPU moving in loop: time={}".format(diff))


def exp_pytorch_gpu_move_before_loop():
    x = torch.rand(1, 1000).to('cuda')
    y = torch.rand(5000, 1000).to('cuda')
    #Exp1
    ticks = time.time()
    for i in range(1,1000):
        z = (x * y).sum()
    tacks = time.time()
    diff = tacks - ticks
    print("Exp: PyTorch tensor GPU moving in loop: time={}".format(diff))


def exp_pytorch_gpu_compare_tensor_to_tensor_cpu():
    x = torch.rand(1, 1)
    y = torch.rand(1, 1)
    ticks = time.time()
    for i in range(1,1000):
        z = x>y
    tacks = time.time()
    diff = tacks - ticks
    print("Exp: PyTorch tensor GPU compare_tensor_to_tensor_cpu: time={}".format(diff))


def exp_pytorch_gpu_compare_tensor_to_tensor_gpu():
    x = torch.rand(1, 1).to('cuda')
    y = torch.rand(1, 1).to('cuda')
    ticks = time.time()
    for i in range(1,1000):
        z = x>y
    tacks = time.time()
    diff = tacks - ticks
    print("Exp: PyTorch tensor GPU compare_tensor_to_tensor_gpu: time={}".format(diff))


def exp_pytorch_gpu_compare_tensor_to_int():
    x = torch.rand(1, 1)
    ticks = time.time()
    for i in range(1,1000):
        z = x>0.5
    tacks = time.time()
    diff = tacks - ticks
    print("Exp: PyTorch tensor GPU compare_tensor_to_int: time={}".format(diff))


def exp_pytorch_gpu_compare_int_to_int():
    x = np.random.rand(1, 1)
    ticks = time.time()
    for i in range(1,1000):
        z = x>0.5
    tacks = time.time()
    diff = tacks - ticks
    print("Exp: PyTorch tensor GPU compare_int_to_int: time={}".format(diff))

if __name__=="__main__":
    exp_numpy()
    exp_pytorch_cpu()
    exp_pytorch_gpu_move_in_loop()
    exp_pytorch_gpu_move_before_loop()


    exp_pytorch_gpu_compare_tensor_to_int()
    exp_pytorch_gpu_compare_tensor_to_tensor_cpu()
    exp_pytorch_gpu_compare_tensor_to_tensor_gpu()
    exp_pytorch_gpu_compare_int_to_int()