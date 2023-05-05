from minigrad.tensor import BadTensor
from graphviz import Digraph
from typing import NamedTuple, Set, Generic, TypeVar

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

def draw_dot(root: BadTensor) -> Digraph:
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
  nodes, edges = trace(root)
  for n in nodes:
    assert isinstance(n, BadTensor)
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data[0], 0. if n.grad is None else n.grad[0]), shape='record')
    if n.op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n.op, label = n.op)
      # and connect this node to it
      dot.edge(uid + n.op, uid)

  for n1, n2 in edges:
    assert isinstance(n1, BadTensor)
    assert isinstance(n2, BadTensor)
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2.op)

  return dot
