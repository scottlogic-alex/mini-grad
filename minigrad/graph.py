from minigrad.tensor import BadTensor
from graphviz import Digraph
from typing import NamedTuple, Set, Generic, TypeVar, Generator, Iterable, Dict
from numpy import concatenate
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

def nums_to_num_strs(arr: Iterable[NDArray]) -> Generator[str, None, None]:
  for elem in arr:
    yield '%.2f' % (elem,)

def num_strs_to_tds(arr: Iterable[str]) -> Generator[str, None, None]:
  for elem in arr:
    yield f'<td>{elem}</td>'

def row_strs_to_trs(arr: Iterable[str]) -> Generator[str, None, None]:
  for elem in arr:
    yield f'<tr>{elem}</tr>'

def ndarray_to_table_rows(arr: NDArray) -> Generator[str, None, None]:
  as_2d = arr.reshape(-1, arr.shape[-1])
  return row_strs_to_trs(''.join(num_strs_to_tds(nums_to_num_strs(row))) for row in as_2d)

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

    catted = concatenate([n.data, n.grad], axis=-1)
    tdata = ''.join((
      f'<tr><td colspan="{catted.data.shape[-1]}">{n.label}</td></tr>'
      f'<tr><td colspan="{n.data.shape[-1]}">data</td><td colspan="{n.grad.shape[-1]}">grad</td></tr>',
      *ndarray_to_table_rows(catted),
    ))
    table = f'<table border="0" cellborder="1" cellspacing="0">{tdata}</table>'
    
    dot.node(name=uid, shape='plain', label=f'<{table}>')
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
