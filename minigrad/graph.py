from minigrad.tensor import BadTensor
from graphviz import Digraph
from typing import NamedTuple, Set, Generic, TypeVar, List, Generator, Iterable, Dict, TypedDict
from numpy.typing import NDArray

TNode = TypeVar('TNode')

class Edge(NamedTuple):
  from_: TNode
  to: TNode
class EdgeGeneric(Edge, Generic[TNode]): pass

class GraphModel(NamedTuple):
  nodes: Set[TNode]
  edges: Set[EdgeGeneric[TNode]]
class GraphModelGeneric(GraphModel, Generic[TNode]): pass

# https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/micrograd/micrograd_lecture_first_half_roughly.ipynb

def trace(root: BadTensor) -> GraphModelGeneric[BadTensor]:
  # builds a set of all nodes and edges in a graph
  nodes: Set[BadTensor] = set()
  edges: Set[EdgeGeneric[BadTensor]] = set()
  def build(v: BadTensor):
    if v not in nodes:
      nodes.add(v)
      for child in v.parents:
        edge: EdgeGeneric[BadTensor] = EdgeGeneric(from_=child, to=v)
        edges.add(edge)
        build(child)
  build(root)
  graph: GraphModelGeneric[BadTensor] = GraphModelGeneric(nodes, edges)
  return graph

def elem_strs(arr: Iterable[NDArray]) -> Generator[str, None, None]:
  for elem in arr:
    yield '%.2f' % (elem,)

def join_elems(arr: List[str]) -> str:
  return '{ %s }' % (' | '.join(arr),)

def make_label(data: NDArray) -> str:
  if data.ndim == 1:
    return join_elems(tuple(elem_strs(data)))
  as_2d = data.reshape(-1, data.shape[-1])
  return join_elems([make_label(row) for row in as_2d])

default_graph_attr: Dict[str, str] = {
  # LR = left to right
  'rankdir': 'LR',
}

def draw_dot(
  root: BadTensor,
  graph_attr: Dict[str, str] = default_graph_attr,
) -> Digraph:
  dot = Digraph(format='svg', graph_attr=graph_attr)
  
  nodes, edges = trace(root)
  for n in nodes:
    assert isinstance(n, BadTensor)
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    # https://graphviz.org/doc/info/shapes.html#record
    label_tensors = (
      'data | %s | grad | %s' % (make_label(n.data), make_label(n.grad))
    ) if graph_attr['rankdir'] == 'LR' else (
      'data | { %s } | grad | { %s }' % (make_label(n.data), make_label(n.grad))
    )
    dot.node(name = uid, label = "%s | { %s }" % (n.label, label_tensors), shape='record')
    if n.op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n.op, label = n.op_format())
      # and connect this node to it
      dot.edge(uid + n.op, uid)

  for n1, n2 in edges:
    assert isinstance(n1, BadTensor)
    assert isinstance(n2, BadTensor)
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2.op)

  return dot
