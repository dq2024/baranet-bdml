from __future__ import annotations

import numpy as np
from typing import Optional, Union
import sys; sys.path.append("build")
import bten
from contextlib import contextmanager

_grad_enabled = True
def is_grad_enabled():
    return _grad_enabled
    
@contextmanager
def no_grad():
    global _grad_enabled
    old = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = old
# -------- autograd wrapper --------

class AGTensor:
    """
    Minimal autograd wrapper over bten.TensorF (and/or) bten.TensorU32.
    Forward ops: +, *, -, @, relu, mean, cross_entropy_loss
    non-differentiable ops: ==, argmax
    Backward: autodiff via backward() call.

    data is tensor that AGTensor wraps. It can be either bten.TensorF or bten.TensorU32 or a numpy array.
    If data is a numpy array, it will be converted to bten.TensorF or bten.TensorU32 based on its dtype.

    children is the set of AGTensors that this tensor's gradients should propagate to. Alternatively, you can 
    think of these children as the parents used to compute this AGTensor.

    op is the operation that produced this tensor, for debugging purposes only.

    requires_grad is a boolean flag indicating whether to track gradients for this tensor.

    is_cuda is an optional boolean flag indicating whether to store the data on GPU or CPU if data is a numpy array.
    """
    def __init__(self, data : Union[np.ndarray, bten.TensorF, bten.TensorU32], children=(), op='', requires_grad=True, is_cuda: Optional[bool]=None):
        # Lab-2: add your code to complete initialization of self.data and self.grad by replacing None with appropriate values.
        # You do not need to change other member variables' initialization.
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op

        if isinstance(data, np.ndarray):
            use_cuda = is_cuda if is_cuda is not None else True
            
            if data.dtype == np.uint32:
                self.data = bten.TensorU32(data.shape[0], data.shape[1], use_cuda)
                self.data.copy_from_numpy(data.astype(np.uint32))
            else:
                self.data = bten.TensorF(data.shape[0], data.shape[1], use_cuda)
                self.data.copy_from_numpy(data.astype(np.float32))
        else:
            self.data = data
        
        self.grad = None
        

    @property
    def shape(self):
         return self.data.shape

    @property
    def is_cuda(self): 
        return self.data.is_cuda

    @property
    def T(self):
        """
        Transpose of a 2D tensor. This function is only partially filled. 
        You need to add the backward() function.
        """
        out = AGTensor(self.data.transpose(), (self,), 'T', requires_grad=self.requires_grad and is_grad_enabled())
        if out.requires_grad:
            def _backward():
                # Lab-2: add your code to compute self.grad (dIn) given out.grad (dOut).
                if self.grad is None:
                    self.grad = out.grad.transpose()
                else:
                    self.grad = self.grad + out.grad.transpose()
            out._backward = _backward
        return out

    def __add__(self, other: Union[AGTensor, float, int]):
        """
        Elementwise addition with broadcasting: self + other
        other can be an AGTensor, float, or int.
        """
        # Lab-2: add your code here
        if isinstance(other, (float, int)):
            out = AGTensor(self.data + other, (self,), f'+scalar', requires_grad=self.requires_grad and is_grad_enabled())
            if out.requires_grad:
                def _backward():
                    if self.grad is None:
                        self.grad = out.grad
                    else:
                        self.grad = self.grad + out.grad
                out._backward = _backward
        else:
            out = AGTensor(self.data + other.data, (self, other), f'+', requires_grad=(self.requires_grad or other.requires_grad) and is_grad_enabled())
            if out.requires_grad:
                def _backward():
                    if self.requires_grad:
                        # broadcast along dimensions where self.shape=1 but out.shape is > than 1
                        grad = out.grad
                        if self.shape != out.shape:
                            if self.shape[0] != out.shape[0] and self.shape[0] == 1:
                                grad = grad.sum(axis=0)
                            if self.shape[1] != out.shape[1] and self.shape[1] == 1:
                                grad = grad.sum(axis=1)

                        if self.grad is None:
                            self.grad = grad
                        else:
                            self.grad = self.grad + grad
                    if other.requires_grad:
                        grad = out.grad
                        if other.shape != out.shape:
                            if other.shape[0] != out.shape[0] and other.shape[0] == 1:
                                grad = grad.sum(axis=0)
                            if other.shape[1] != out.shape[1] and other.shape[1] == 1:
                                grad = grad.sum(axis=1)

                        if other.grad is None:
                            other.grad = grad
                        else:
                            other.grad = other.grad + grad
                out._backward = _backward
        
        return out

    def __sub__(self, other: AGTensor):
        """Elementwise subtraction with broadcasting: self - other"""
        #Lab-2: add your code here
        if isinstance(other, (float, int)):
            out = AGTensor(self.data - other, (self,), f'-scalar', requires_grad=self.requires_grad and is_grad_enabled())
            if out.requires_grad:
                def _backward():
                    if self.grad is None:
                        self.grad = out.grad
                    else:
                        self.grad = self.grad + out.grad
                out._backward = _backward
        else:
            out = AGTensor(self.data - other.data, (self, other), f'-', requires_grad=(self.requires_grad or other.requires_grad) and is_grad_enabled())
            if out.requires_grad:
                def _backward():
                    if self.requires_grad:
                        grad = out.grad
                        if self.shape != out.shape:
                            if self.shape[0] != out.shape[0] and self.shape[0] == 1:
                                grad = grad.sum(axis=0)
                            if self.shape[1] != out.shape[1] and self.shape[1] == 1:
                                grad = grad.sum(axis=1)


                        if self.grad is None:
                            self.grad = grad
                        else:
                            self.grad = self.grad + grad
                    if other.requires_grad:
                        grad = out.grad * -1                    
                        if other.shape != out.shape:
                            if other.shape[0] != out.shape[0] and other.shape[0] == 1:
                                grad = grad.sum(axis=0)
                            if other.shape[1] != out.shape[1] and other.shape[1] == 1:
                                grad = grad.sum(axis=1)

                        if other.grad is None:
                            other.grad = grad
                        else:
                            other.grad = other.grad + grad
                out._backward = _backward
        
        return out

    def __mul__(self, other : Union[AGTensor, float, int]):
        """
        Elementwise multiplication with broadcasting: self * other
        other can be an AGTensor, float, or int.
        """
        # Lab-2: add your code here
        if isinstance(other, (float, int)):
            out = AGTensor(self.data * other, (self,), f'*scalar', requires_grad=self.requires_grad and is_grad_enabled())
            if out.requires_grad:
                def _backward():
                    if self.grad is None:
                        self.grad = out.grad * other
                    else:
                        self.grad = self.grad + out.grad * other
                out._backward = _backward
        else:
            out = AGTensor(self.data * other.data, (self, other), f'*', requires_grad=(self.requires_grad or other.requires_grad) and is_grad_enabled())
            if out.requires_grad:
                def _backward():
                    if self.requires_grad:
                        grad = out.grad * other.data
                    
                        if self.shape != out.shape:
                            if self.shape[0] != out.shape[0] and self.shape[0] == 1:
                                grad = grad.sum(axis=0)
                            if self.shape[1] != out.shape[1] and self.shape[1] == 1:
                                grad = grad.sum(axis=1)

                        if self.grad is None:
                            self.grad = grad
                        else:
                            self.grad = self.grad + grad
                    if other.requires_grad:
                        grad = out.grad * self.data
                    
                        if other.shape != out.shape:
                            if other.shape[0] != out.shape[0] and other.shape[0] == 1:
                                grad = grad.sum(axis=0)
                            if other.shape[1] != out.shape[1] and other.shape[1] == 1:
                                grad = grad.sum(axis=1)

                        if other.grad is None:
                            other.grad = grad
                        else:
                            other.grad = other.grad + grad
                out._backward = _backward
        
        return out


    def __matmul__(self, other: AGTensor):
        """Matrix multiplication: self @ other"""
        # Lab-2: add your code here
        out = AGTensor(self.data @ other.data, (self, other), '@', requires_grad=(self.requires_grad or other.requires_grad) and is_grad_enabled())
        
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    grad_self = out.grad @ other.data.transpose()
                    if self.grad is None:
                        self.grad = grad_self
                    else:
                        self.grad = self.grad + grad_self
                
                if other.requires_grad:
                    grad_other = self.data.transpose() @ out.grad
                    if other.grad is None:
                        other.grad = grad_other
                    else:
                        other.grad = other.grad + grad_other
            out._backward = _backward
        
        return out

    def relu(self):
        """
        Elementwise ReLU. 
        Forward: y = max(0, x)
        Backward: uses self.data.relu_back for d/dx.
        """
        # Lab-2: add your code here
        out_data = self.data.relu()
        out = AGTensor(out_data, (self,), 'relu', requires_grad=self.requires_grad and is_grad_enabled())
        
        if out.requires_grad:
            def _backward():
                grad_input = self.data.relu_back(out.grad)
                
                if self.grad is None:
                    self.grad = grad_input
                else:
                    self.grad = self.grad + grad_input
            out._backward = _backward
        
        return out

    def sum(self, axis: Optional[int] = None):
        """
        Sum reduction. If axis is None, sum to scalar (1x1 tensor).
        If axis=0 or 1, reduce over that axis and keepdims.
        """
        if axis is None:
            temp = self.data.sum(axis=0)  
            out_data = temp.sum(axis=1)   
            out = AGTensor(out_data, (self,), 'sum', requires_grad=self.requires_grad and is_grad_enabled())
            
            if out.requires_grad:
                def _backward():
                    grad_self = bten.TensorF(self.shape[0], self.shape[1], self.is_cuda)
                    grad_self.fill(1.0)
                    scalar_grad = out.grad.to_numpy()[0, 0]
                    grad_self = grad_self * scalar_grad
                    
                    if self.grad is None:
                        self.grad = grad_self
                    else:
                        self.grad = self.grad + grad_self
                out._backward = _backward

        else:
            out_data = self.data.sum(axis=axis)
            out = AGTensor(out_data, (self,), f'sum', requires_grad=self.requires_grad and is_grad_enabled())
            
            if out.requires_grad:
                def _backward():
                    if self.grad is None:
                        self.grad = bten.TensorF(self.shape[0], self.shape[1], self.is_cuda)
                        self.grad.fill(0.0)
                        self.grad = self.grad + out.grad 
                    else:
                        self.grad = self.grad + out.grad  
                out._backward = _backward
        return out 
    
    def mean(self):
        """Mean of all elements, returning a scalar AGTensor(1x1 tensor)."""
        # Lab-2: add your code here
        # Hint: It can be implemented using AGTensor's sum and * operation and therefore no need for separate backward logic.
        return self.sum() * (1.0 / (self.shape[0] * self.shape[1]))

    def cross_entropy_loss(self, targets: np.ndarray):
        """
        Calculate cross-entropy loss between self (logit Tensor) and targets (integer label tensor). 
        Returns a scalar AGTensor (1x1).
        """
        # Lab-2: add your code here. You should use self.data.cross_entropy_loss...
        tensor = bten.TensorU32(targets.shape[0], 1, self.is_cuda)
        tensor.copy_from_numpy(targets.astype(np.uint32).reshape(-1, 1))
        
        grad = bten.TensorF(self.shape[0], self.shape[1], self.is_cuda)
        
        loss_value = self.data.cross_entropy_loss(tensor, grad)
        
        loss_tensor = bten.TensorF(1, 1, self.is_cuda)
        loss_tensor.fill(loss_value)
        
        out = AGTensor(loss_tensor, (self,), 'cross_entropy', requires_grad=self.requires_grad and is_grad_enabled())
        
        if out.requires_grad:
            def _backward():
                upstream_grad = out.grad.to_numpy()[0, 0]
                scaled_grad = grad * upstream_grad
                
                if self.grad is None:
                    self.grad = scaled_grad
                else:
                    self.grad = self.grad + scaled_grad
            out._backward = _backward
        
        return out
        
    def argmax(self):
        """
        Per-row argmax, returns an AGTensor of shape (N,1).
        This function is non-differentiable.
        """
        # Lab-2: add your code here"""
        return AGTensor(self.data.argmax(), (), 'argmax', requires_grad=False)
   
    def __eq__(self, other: Union[AGTensor, np.ndarray]):
        """
        Elementwise equality (with broadcasting).
        Returns a AGTensor with the broadcasted output shape whose elements are 1 (True) or 0 (False) 
        at the location where the correspond elements in self and other are equal.
        other can be an AGTensor or a NumPy array.
        This function is non-differentiable.
        """
        if isinstance(other, np.ndarray):
            other = AGTensor(other, requires_grad=False, is_cuda=self.is_cuda)
            
        out = AGTensor(self.data == other.data, (), '==', requires_grad=False)
        return out

  
    #since I re-defined __eq__, I need to re-define __hash__
    __hash__ = object.__hash__


    # ----- backprop driver -----
    def backward(self, grad=None):
        """
        Backpropagate the gradient of the loss through the saved computation graph.
        Call this function on a scalar AGTensor (i.e., shape (1,1)).
        The optional grad argument is the initial gradient to be backpropagated.
        If grad is None, it defaults to a tensor of ones with the same shape as self.
        """
        # Lab-2: add your code here
        if grad is None:
            self.grad = bten.TensorF(self.shape[0], self.shape[1], self.is_cuda)
            self.grad.fill(1.0)
        else:
            self.grad = grad 

        topo = []
        visited = set()
        
        def topological_sort(node):
            if node not in visited and node.requires_grad:
                visited.add(node)
                for child in node._prev:
                    topological_sort(child)
                topo.append(node)
        topological_sort(self)

        for node in reversed(topo):
            node._backward()

         
    # Convenience: numpy view for debugging
    def numpy(self):
        return self.data.to_numpy()

    def __repr__(self):
        dev = "cuda" if self.is_cuda else "cpu"
        return f"Tensor(shape={self.data.shape}, device={dev}, op='{self._op}')"