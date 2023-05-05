from __future__ import annotations
from numpy.typing import NDArray
from typing import Optional, Generator, Iterable
from enum import StrEnum
from collections import deque

class Op(StrEnum):
  add = '+'
  sub = '−'
  mul = '×'

class BadTensor:
  data: NDArray
  # partial derivative
  grad: Optional[NDArray] = None
  parents: Iterable[BadTensor]
  op: Optional[Op]
  label: Optional[str]
  def __init__(
    self,
    data: NDArray,
    parents: Iterable[BadTensor] = (),
    op: Optional[Op] = None,
    label: Optional[str] = None
  ) -> None:
    self.data = data
    self.parents = parents
    self.op = op
    self.label = label

  # https://www.programiz.com/python-programming/operator-overloading
  def __add__(self, other: BadTensor) -> BadTensor:
    return BadTensor(self.data + other.data, parents=(self, other), op=Op.add)

  def __mul__(self, other: BadTensor) -> BadTensor:
    return BadTensor(self.data * other.data, parents=(self, other), op=Op.mul)
  
  def label_(self, label: str) -> BadTensor:
    self.label = label
    return self
  
  def __repr__(self) -> str:
    data = self.data.__repr__()
    label_prefix = '' if self.label is None else f'{self.label}: '
    return f'{label_prefix}{data}'
  
  def ancestors(self) -> Generator[BadTensor, None, None]:
    to_visit: deque[BadTensor] = deque(self.parents)
    while to_visit:
      tensor: BadTensor = to_visit.popleft()
      yield tensor
      to_visit.extend(tensor.parents)
