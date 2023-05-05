from __future__ import annotations
from numpy.typing import NDArray
import numpy as np
from numpy import zeros_like, ones_like
from typing import Optional, Generic, Iterator, Generator
from enum import StrEnum

class Op(StrEnum):
  add = '+'
  sub = '−'
  mul = '×'

class BadTensor:
  data: NDArray
  # partial derivative
  grad: Optional[NDArray] = None
  parent: Optional[BadTensor]
  op: Optional[Op]
  label: Optional[str]
  def __init__(
    self,
    data: NDArray,
    parent: Optional[BadTensor] = None,
    op: Optional[Op] = None,
    label: Optional[str] = None
  ) -> None:
    self.data = data
    self.parent = parent
    self.op = op
    self.label = label

  # https://www.programiz.com/python-programming/operator-overloading
  def __add__(self, other: BadTensor) -> BadTensor:
    dother_dself = ones_like(other.data)
    return BadTensor(self.data + other.data, parent=self, op=Op.add)

  def __mul__(self, other: BadTensor) -> BadTensor:
    return BadTensor(self.data * other.data, parent=self, op=Op.mul)
  
  def label_(self, label: str) -> BadTensor:
    self.label = label
    return self
  
  def __repr__(self) -> str:
    data = self.data.__repr__()
    label_prefix = '' if self.label is None else f'{self.label}: '
    return f'{label_prefix}{data}'
  
  def ancestors(self) -> Generator[BadTensor, None, None]:
    if self.parent is not None:
      yield self.parent
      yield from self.parent.ancestors()
