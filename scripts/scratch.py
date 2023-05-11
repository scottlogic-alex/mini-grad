from minigrad.tensor import BadTensor
# from minigrad.graph import draw_dot
from numpy import r_, array, zeros_like, atleast_2d, set_printoptions
from numpy.random import rand
from typing import List, Generator
# from graphviz import Digraph
from itertools import islice

set_printoptions(suppress=True)
def feet_inches_mat() -> Generator[BadTensor, None, None]:
  # inputs = rand(16, 2) * r_[10, 12] # array([[6., 0.], [5., 2.], [4., 5.], [3., 11.], [2., 7.]])
  # inputs = rand(1, 2) * r_[10, 12]
  inputs = rand(1, 1) * r_[10]
  # true_weights = array([[2.54*12*.01, 2.54*.01], [2.54*12, 2.54], [2.54*12*10, 2.54*10]]).T
  # true_weights = array([[2.54*12, 2.54]]).T
  true_weights = array([[2.54*12]]).T
  minibatch_size = inputs.shape[0]
  lr = 5e-3# * minibatch_size
  feet_inches = BadTensor(inputs, label='input_feet_inches')
  # should converge on true_weights
  # feet_inches_to_cm = BadTensor(array([[0., 0.], [0., 0.], [0., 0.]]).T, label='learned_feet_inches_to_m_cm_mm', train_me=True)
  # feet_inches_to_cm = BadTensor(array([[0., 0.]]).T, label='learned_feet_inches_to_cm', train_me=True)
  feet_inches_to_cm = BadTensor(array([[0.]]).T, label='learned_feet_to_cm', train_me=True)
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

    trainable: List[BadTensor] = [t for t in mse.ancestors() if t.train_me]
    for t in trainable:
      t.data -= t.grad * lr / minibatch_size

    for t in mse.ancestors():
      t.grad = zeros_like(t.data)

def simple_sum() -> Generator[BadTensor, None, None]:
  input_arr = rand(1) * r_[10]
  minibatch_size = input_arr.shape[0]
  lr = 5e-2# * minibatch_size
  input = BadTensor(input_arr, label='input')
  true_bias = r_[5.]
  learned_bias = BadTensor(r_[0.], label='learned_bias', train_me=True)
  true_sum = BadTensor(input.data + true_bias, label='true_sum')

  while True:
    calc_sum = (input + learned_bias).label_('calc_sum')

    diff = (true_sum - calc_sum).label_('diff')
    L2 = (diff ** 2).label_('L2')
    sum = (L2.sum()).label_('sum')
    mse = (sum / L2.data.size).label_('MSE')

    mse.backward()

    yield mse

    trainable: List[BadTensor] = [t for t in mse.ancestors() if t.train_me]
    for t in trainable:
      t.data -= t.grad * lr / minibatch_size

    for t in mse.ancestors():
      t.grad = zeros_like(t.data)

it = simple_sum()

# root: BadTensor = next(islice(feet_inches_mat(), 100, None))
for step, _ in zip(it, range(100)):
  print(f'loss: {step.data[0]}')
needle: BadTensor = next((t for t in step.ancestors() if t.label == 'learned_bias'))
print(needle)
pass # somewhere to put a breakpoint