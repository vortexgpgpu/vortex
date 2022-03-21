
import logging
import networkx as nx

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
  name= 'cora'
  path_prefix= f'/esat/puck1/users/nshah/datasets/graphs/{name}/'
  graph_nx= process_input_graphs(name, path_prefix)

  output_path= f'../tests/opencl/bfs/{name}.txt'
  write_files_for_kernel(graph_nx, output_path)


def relabel_nodes_with_contiguous_numbers(graph_nx, start= 0):
  """
    Creates a shallow copy
  """
  mapping= {n : (idx + start) for idx, n in enumerate(list(graph_nx.nodes()))}

  return nx.relabel.relabel_nodes(graph_nx, mapping, copy= True), mapping

def process_input_graphs(name, path_prefix):
  assert name in ['cora', 'arxiv', 'citeseer']

  if name == 'cora':
    f= path_prefix + 'cora.cites'
    f= open(f, 'r')
    edge_ls_str= list(f.readlines())
    
    edge_ls = []
    for e in edge_ls_str:
      dst, src = e.split()
      dst = int(dst)
      src = int(src)
      edge_ls.append((src, dst))

  else:
    assert 0

  graph_nx= nx.DiGraph()
  graph_nx.add_edges_from(edge_ls)

  graph_nx, _ = relabel_nodes_with_contiguous_numbers(graph_nx, start= 0)
  
  return graph_nx

def write_files_for_kernel(graph_nx, path):
  
  # try:
  #   graph_nx= graph_nx.to_undirected()
  # except AttributeError:
  #   logging.info('graph is already undirected')

  data = ''
  data += f'{graph_nx.number_of_nodes()}\n'
  tot_edges= 0
  for n in sorted(list(graph_nx.nodes())):
    # n_edges= len(list(graph_nx.neighbors(n)))
    n_edges= len(list(graph_nx.out_edges(n)))
    data += f'{tot_edges} {n_edges}\n'
    tot_edges += n_edges
  
  data += '\n'
  data += '0\n' # src nope for bfs
  data += '\n'

  data += f'{graph_nx.number_of_edges()}\n'
  for n in sorted(list(graph_nx.nodes())):
    # for neighbor in graph_nx.neighbors(n):
    for neighbor in graph_nx.successors(n):
      data += f'{neighbor} 1\n' # defaulting edge weight to 1
  
  with open(path, 'w+') as fp:
    fp.write(data)

if __name__=='__main__':
  main()
