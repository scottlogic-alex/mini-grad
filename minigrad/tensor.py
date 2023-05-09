from __future__ import annotations
from numpy.typing import NDArray
from numpy import ones_like, zeros_like, array_str
from typing import Optional, Generator, Iterable, TypeAlias, Callable
from enum import StrEnum
from collections import deque

class Op(StrEnum):
  add = '+'
  sub = '−'
  mul = '×'
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
    label: Optional[str] = None,
    train_me: bool = False,
  ) -> None:
    self.data = data
    self.grad = zeros_like(data)
    self.parents = parents
    self.op = op
    self.op_format = op_format or (lambda: self.op)
    self.label = label
    self.train_me = train_me

  # https://www.programiz.com/python-programming/operator-overloading
  def __add__(self, other: BadTensor) -> BadTensor:
    sum = BadTensor(self.data + other.data, parents=(self, other), op=Op.add)
    def _backward() -> None:
      # dsum_dself = 1
      # dsum_dother = 1
      self.grad += sum.grad # * dsum_dself
      other.grad += sum.grad # * dsum_dother
    sum._backward = _backward
    return sum

  # https://www.programiz.com/python-programming/operator-overloading
  def __sub__(self, other: BadTensor) -> BadTensor:
    sum = BadTensor(self.data - other.data, parents=(self, other), op=Op.sub)
    def _backward() -> None:
      # dsum_dself = 1
      # dsum_dother = -1
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

  def __pow__(self, other: BadTensor | float | int) -> BadTensor:
    if isinstance(other, BadTensor):
      raise ValueError('exp not yet implemented for tensor')
    exp = BadTensor(self.data ** other, parents=(self,), op=Op.exp, op_format=lambda: f'{Op.exp}{other}')
    def _backward() -> None:
      dexp_dself = self.data * other
      self.grad += exp.grad * dexp_dself
    exp._backward = _backward
    return exp

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
