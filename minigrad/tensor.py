from __future__ import annotations
from numpy.typing import NDArray
from numpy import ones_like, zeros_like, array_str, broadcast_to, repeat, expand_dims, allclose, array, zeros, atleast_1d
from typing import Optional, Generator, Iterable, TypeAlias, Callable
from enum import StrEnum
from collections import deque
import numpy as np

class Op(StrEnum):
  add = '+'
  sum = '∑'
  sub = '−'
  mul = '×'
  div = '÷'
  matmul = ':'
  exp = '**'
  relu = 'ReLU'

Backward: TypeAlias = Callable[[], None]
OpFormat: TypeAlias = Callable[[], str]

class BadTensor:
  data: NDArray
  grad: NDArray
  parents: Iterable[BadTensor]
  op: Optional[Op]
  op_format: OpFormat
  label: Optional[str]
  _backward: Optional[Backward] = None
  train_me: bool
  def __init__(
    self,
    data: NDArray,
    parents: Iterable[BadTensor] = (),
    op: Optional[Op] = None,
    op_format: Optional[OpFormat] = None,
    # grad: Optional[NDArray] = None,
    label: Optional[str] = None,
    train_me: bool = False,
  ) -> None:
    self.data = data
    self.grad = zeros_like(data, dtype=float)# if grad is None else grad
    self.parents = parents
    self.op = op
    self.op_format = op_format or (lambda: self.op)
    self.label = label
    self.train_me = train_me
  
  def sum(self, dim: Optional[int] = None) -> BadTensor:
    sum = BadTensor(atleast_1d(self.data.sum(dim)), parents=(self,), op=Op.sum)
    def _backward() -> None:
      # sum     = self0 + self1 + self2
      # dsum_dself0 = 1
      # dsum_dself1 = 1
      # dsum_dself2 = 1
      # self0.grad += droot_dsum * dsum_dself0
      #             =   sum.grad * dsum_dself0
      #             =   sum.grad

      # dsum_dself = ones_like(self.data)
      self.grad += sum.grad # * dsum_dself
    sum._backward = _backward
    return sum

  # https://www.programiz.com/python-programming/operator-overloading
  def __add__(self, other: BadTensor) -> BadTensor:
    sum = BadTensor(self.data + other.data, parents=(self, other), op=Op.add)
    def _backward() -> None:
      # dsum_dself = ones_like(self.data)
      # dsum_dother = ones_like(other.data)
      self.grad += sum.grad # * dsum_dself
      other.grad += sum.grad # * dsum_dother
    sum._backward = _backward
    return sum

  # https://www.programiz.com/python-programming/operator-overloading
  def __sub__(self, other: BadTensor) -> BadTensor:
    sum = BadTensor(self.data - other.data, parents=(self, other), op=Op.sub)
    def _backward() -> None:
      # dsum_dself = ones_like(self.data)
      # dsum_dother = -ones_like(other.data)
      self.grad += sum.grad # * dsum_dself
      other.grad -= sum.grad
    sum._backward = _backward
    return sum

  def __mul__(self, other: BadTensor) -> BadTensor:
    prod = BadTensor(self.data * other.data, parents=(self, other), op=Op.mul)
    def _backward() -> None:
      dprod_dself = other.data
      dprod_dother = self.data
      self.grad += prod.grad * dprod_dself
      other.grad += prod.grad * dprod_dother
    prod._backward = _backward
    return prod

  def __truediv__(self, other: BadTensor | int | float) -> BadTensor:
    if isinstance(other, BadTensor):
      quot_data: NDArray = self.data / other.data
      quot_parents: Iterable[BadTensor] = (self, other)
    else:
      quot_data: NDArray = self.data / other
      quot_parents: Iterable[BadTensor] = (self,)
    quot = BadTensor(quot_data, parents=quot_parents, op=Op.div)
    def _backward() -> None:
      if isinstance(other, BadTensor):
        dquot_dself: NDArray = other.data**-1
        dquot_dother = self.data * -other.data**-2
        other.grad += quot.grad * dquot_dother
      else:
        dquot_dself: int | float = other**-1
      self.grad += quot.grad * dquot_dself
    quot._backward = _backward
    return quot

  def __pow__(self, other: BadTensor | float | int) -> BadTensor:
    if isinstance(other, BadTensor):
      raise ValueError('exp not yet implemented for tensor')
    exp = BadTensor(self.data ** other, parents=(self,), op=Op.exp, op_format=lambda: f'{Op.exp}{other}')
    def _backward() -> None:
      dexp_dself = self.data * other
      self.grad += exp.grad * dexp_dself
    exp._backward = _backward
    return exp

  def __matmul__(self, other: BadTensor) -> BadTensor:
    # self=input @ other=weights
    mm = BadTensor(self.data @ other.data, parents=(self, other), op=Op.matmul)
    def _backward() -> None:
      # dmm_dsum = broadcast_to(other.data.T, (*self.data.shape[:-1], *other.data.shape[:-1]))
      # dmm_dself = other.data.T.repeat(self.data.shape[0], axis=0)
      # dmm_dother = self.data
      # self_broadcast = expand_dims(self.data, -2).repeat(other.data.shape[-1], axis=-2)
      # other_broadcast = expand_dims(other.data.T, 0).repeat(self.data.shape[0], axis=0)
      # hadamard = self_broadcast * other_broadcast
      # alt_matmul = hadamard.sum(-1)
      # hadamard.sum(0)

      # dhada_dself = other_broadcast
      # dhada_dother = self_broadcast
      # dmm_dhada = ones_like(self_broadcast)

      # print(alt_matmul)
      # print(dhada_dself)
      # print(dhada_dother)
      # print(dmm_dhada)
      # print(mm)
      # pass
      # dsum_dmm = ones_like()
      # dself_dsum = 1
      # TODO: check whether we need to transpose
      # dmm_dself = other.data
      # dmm_dother = self.data
      # other.grad += mm.grad * self.data.sum(0, keepdims=True).repeat(other.data.shape[-1], axis=-2).T
      # self.grad += mm.grad * dmm_dself

      # these are likely to be wrong
      self.grad += mm.grad.sum(-1, keepdims=True) * other.data.sum(-1)
      # other.grad += mm.grad.sum(-2) * self.data.sum(-2, keepdims=True).T
      # other.grad += (mm.grad.repeat(self.data.shape[-1], -1) * self.data).sum(-2, keepdims=True).T
      other.grad += (expand_dims(self.data.T, -1).repeat(other.grad.shape[-1], -1, ) * mm.grad).sum(-2)
    mm._backward = _backward
    return mm

  def relu(self) -> BadTensor:
    gated = BadTensor(self.data.clip(0.), parents=(self), op=Op.relu)
    def _backward() -> None:
      # TODO: needs testing
      dgated_dself = (gated.data > 0.).astype(float)
      self.grad += gated.grad * dgated_dself
    gated._backward = _backward
    return gated
  
  def label_(self, label: str) -> BadTensor:
    self.label = label
    return self
  
  def __repr__(self) -> str:
    data = array_str(self.data)
    grad = array_str(self.grad)
    label_prefix = '' if self.label is None else f'{self.label}: '
    return f'{label_prefix}(data: {data}, grad: {grad})'
  
  def ancestors(self) -> Generator[BadTensor, None, None]:
    to_visit: deque[BadTensor] = deque(self.parents)
    while to_visit:
      tensor: BadTensor = to_visit.popleft()
      yield tensor
      to_visit.extend(tensor.parents)

  def backward(self) -> None:
    self.grad = ones_like(self.data)
    self._backward()
    for ancestor in self.ancestors():
      if next(ancestor.parents.__iter__(), None) is not None:
        ancestor._backward()
