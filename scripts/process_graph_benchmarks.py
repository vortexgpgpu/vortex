
import logging
import networkx as nx

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
  # name= 'cora'
  name_ls = [\
    # 'cora', 
    # 'pubmed_diabetes',
    # 'p2p-Gnutella30',
    # 'CA-HepPh',
    # 'Cit-HepPh',
    # 'Slashdot0811',
    'web-Google',
  ]

  for name in name_ls:
    path_prefix= f'/esat/puck1/users/nshah/datasets/graphs/'
    graph_nx= process_input_graphs(name, path_prefix)

    output_path= f'../tests/opencl/bfs/{name}.txt'
    write_files_for_kernel(graph_nx, output_path)


def relabel_nodes_with_contiguous_numbers(graph_nx, start= 0):
  """
    Creates a shallow copy
  """
  mapping= {n : (idx + start) for idx, n in enumerate(list(graph_nx.nodes()))}

  return nx.relabel.relabel_nodes(graph_nx, mapping, copy= True), mapping

def simple_edge_ls(edge_ls_str, str_format= 'dst_src'):
  assert str_format in ['dst_src', 'src_dst']
  edge_ls = []
  for e in edge_ls_str:
    if str_format == 'dst_src':
      dst, src = e.split()
    elif str_format == 'src_dst':
      src, dst = e.split()
    else:
      assert 0

    dst = int(dst)
    src = int(src)
    edge_ls.append((src, dst))
  
  return edge_ls

def process_input_graphs(name, path_prefix):

  if name == 'cora':
    f= path_prefix + 'core/cora.cites'
    f= open(f, 'r')
    edge_ls_str= list(f.readlines())
    edge_ls= simple_edge_ls(edge_ls_str, str_format= 'dst_src')
    
  elif name == 'pubmed_diabetes':
    f= path_prefix + 'Pubmed-Diabetes/data/Pubmed-Diabetes.DIRECTED.cites.tab'
    f= open(f, 'r')
    edge_ls_str= list(f.readlines())

    #remove initial comments
    edge_ls_str = edge_ls_str[2:]
    
    edge_ls = []
    for e in edge_ls_str:
      _, dst, _ , src = e.split()
      dst = dst.replace('paper:', '')
      src = src.replace('paper:', '')
      dst = int(dst)
      src = int(src)
      edge_ls.append((src, dst))

  elif name == 'p2p-Gnutella30':
    f= path_prefix + 'snap/p2p-Gnutella30.txt'
    f= open(f, 'r')
    edge_ls_str= list(f.readlines())
    edge_ls_str = edge_ls_str[4:]
    edge_ls= simple_edge_ls(edge_ls_str, str_format= 'src_dst') 

  elif name == 'CA-HepPh':
    f= path_prefix + 'snap/CA-HepPh.txt'
    f= open(f, 'r')
    edge_ls_str= list(f.readlines())
    edge_ls_str = edge_ls_str[4:]
    edge_ls= simple_edge_ls(edge_ls_str, str_format= 'src_dst') 

  elif name == 'Cit-HepPh':
    f= path_prefix + 'snap/Cit-HepPh.txt'
    f= open(f, 'r')
    edge_ls_str= list(f.readlines())
    edge_ls_str = edge_ls_str[4:]
    edge_ls= simple_edge_ls(edge_ls_str, str_format= 'src_dst') 

  elif name == 'Slashdot0811':
    f= path_prefix + 'snap/Slashdot0811.txt'
    f= open(f, 'r')
    edge_ls_str= list(f.readlines())
    edge_ls_str = edge_ls_str[4:]
    edge_ls= simple_edge_ls(edge_ls_str, str_format= 'src_dst') 

  elif name == 'web-Google':
    f= path_prefix + 'snap/web-Google.txt'
    f= open(f, 'r')
    edge_ls_str= list(f.readlines())
    edge_ls_str = edge_ls_str[4:]
    edge_ls= simple_edge_ls(edge_ls_str, str_format= 'src_dst') 

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
