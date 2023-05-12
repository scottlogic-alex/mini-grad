from minigrad.tensor import BadTensor
# from minigrad.graph import draw_dot
from numpy import r_, array, zeros_like, atleast_2d, set_printoptions, zeros, arange, expand_dims
from numpy.random import rand
from numpy.typing import NDArray
from typing import List, Generator, Dict
# from graphviz import Digraph
from itertools import islice

set_printoptions(suppress=True)
def feet_inches_mat() -> Generator[BadTensor, None, None]:
  minibatch_size = 10
  # true_weights = array([[2.54*12*.01, 2.54*.01], [2.54*12, 2.54], [2.54*12*10, 2.54*10]]).T
  # true_weights = array([[2.54*12, 2.54], [2.54*12*10, 2.54*10]]).T
  true_weights = array([[2.54*12, 2.54]]).T
  # true_weights = array([[2.54*12]]).T
  # lr = 1e-5 * minibatch_size
  # lr = 5e-3# * minibatch_size
  # lr = 5e-3# * minibatch_size
  # lr = 1e-3# * minibatch_size
  lr = 1e-2 * minibatch_size # known-good for batch-of-10 feet_in->cm
  # lr = 1e-2# * minibatch_size
  # lr = 1e-1# * minibatch_size
  # should converge on true_weights
  # feet_inches_to_cm = BadTensor(zeros_like(true_weights), label='learned_feet_inches_to_m_cm_mm', train_me=True)
  # feet_inches_to_cm = BadTensor(zeros_like(true_weights), label='learned_feet_inches_to_cm_mm', train_me=True)
  feet_inches_to_cm = BadTensor(zeros_like(true_weights), label='learned_feet_inches_to_cm', train_me=True)
  # feet_inches_to_cm = BadTensor(zeros_like(true_weights), label='learned_feet_to_cm', train_me=True)
  # feet_inches_to_cm = BadTensor(true_weights, label='learned_feet_inches_to_m_cm_mm', train_me=True)

  ema_history_depth = 10
  ema_decay = 0.9
  ema_history: Dict[BadTensor, NDArray] = {}

  while True:
    inputs = rand(minibatch_size, 2) * r_[10, 12]
    # inputs = rand(minibatch_size, 1) * r_[10]
    input_feet_inches = BadTensor(inputs, label='input_feet_inches')
    # input_feet_inches = BadTensor(inputs, label='input_feet')
    # expected_cm = BadTensor(inputs @ true_weights, label='true_cm_mm')
    expected_cm = BadTensor(inputs @ true_weights, label='true_cm')

    predicted_cm = (input_feet_inches @ feet_inches_to_cm).label_('predicted_cm')
    diff = (expected_cm - predicted_cm).label_('diff')
    L2 = (diff ** 2).label_('L2')
    sum = (L2.sum()).label_('sum')
    mse = (sum / L2.data.size).label_('MSE')

    if mse.data[0] < 1:
      pass

    mse.backward()

    yield mse

    trainable: List[BadTensor] = [t for t in mse.ancestors() if t.train_me]
    for t in trainable:
      proposed_new_weights = t.data - t.grad * lr / minibatch_size
      # t.data -= t.grad * lr / minibatch_size
      if t not in ema_history:
        ema_history[t] = zeros((ema_history_depth, *t.grad.shape), dtype=t.grad.dtype)
      # let's avoid a big copy
      # ema_history[t] = ema_history[t].roll(ema_history[t], 1, axis=0)
      # prefer in-place roll
      ema_history[t][1:] = (1 - ema_decay) * ema_history[t][:-1]
      ema_history[t][0] = ema_decay * proposed_new_weights
      t.data = ema_history[t].sum(0)

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

# it, needle_label = simple_sum(), 'learned_bias'
# it, needle_label = feet_inches_mat(), 'learned_feet_to_cm'
it, needle_label = feet_inches_mat(), 'learned_feet_inches_to_cm'
# it, needle_label = feet_inches_mat(), 'learned_feet_inches_to_cm_mm'
# it, needle_label = feet_inches_mat(), 'learned_feet_inches_to_m_cm_mm'

for step, _ in zip(it, range(100)):
  print(f'loss: {step.data[0]}')
needle: BadTensor = next((t for t in step.ancestors() if t.label == needle_label))
print(needle.data)
pass # somewhere to put a breakpoint