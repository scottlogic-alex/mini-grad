from minigrad.tensor import BadTensor
# from minigrad.graph import draw_dot
from numpy import r_, array, zeros_like, atleast_2d, set_printoptions
from numpy.random import rand
from typing import List, Generator
# from graphviz import Digraph
from itertools import islice

set_printoptions(suppress=True)
def feet_inches_mat() -> Generator[BadTensor, None, None]:
  inputs = rand(16, 2) * r_[10, 12] # array([[6., 0.], [5., 2.], [4., 5.], [3., 11.], [2., 7.]])
  true_weights = array([[2.54*12*.01, 2.54*.01], [2.54*12, 2.54], [2.54*12*10, 2.54*10]]).T
  minibatch_size = inputs.shape[0]
  lr = 5e-3# * minibatch_size
  feet_inches = BadTensor(inputs, label='input_feet_inches')
  # should converge on true_weights
  feet_inches_to_cm = BadTensor(array([[0., 0.], [0., 0.], [0., 0.]]).T, label='learned_feet_inches_to_m_cm_mm', train_me=True)
  # feet_inches_to_cm = BadTensor(true_weights, label='learned_feet_inches_to_m_cm_mm', train_me=True)
  expected_cm = BadTensor(inputs @ true_weights, label='true_cm_mm')

  while True:
    predicted_cm = (feet_inches @ feet_inches_to_cm).label_('predicted_cm')
    diff = (expected_cm - predicted_cm).label_('diff')
    L2 = (diff ** 2).label_('L2')
    sum = (L2.sum()).label_('sum')
    mse = (sum / L2.data.size).label_('MSE')

    mse.backward()

    yield mse

    trainable: List[BadTensor] = [t for t in L2.ancestors() if t.train_me]
    for t in trainable:
      t.data -= t.grad * lr / minibatch_size

    for t in L2.ancestors():
      t.grad = zeros_like(t.data)
root: BadTensor = next(islice(feet_inches_mat(), 1000, None))
pass