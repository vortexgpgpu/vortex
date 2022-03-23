import pandas as pd
from statistics import mean
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
  plot_bfs_perf()

class PerfLog():
  def __init__(self):
    self.args     = None
    self.app      = None
    self.clusters = None
    self.cores    = None
    self.warps    = None
    self.threads  = None
    self.l2cache  = None
    self.l3cache  = None
    self.driver   = None
    self.perf     = None

    # instrs                      = None
    # cycles                      = None
    # IPC                         = None
    # ibuffer stalls              = None
    # scoreboard stalls           = None
    # alu unit stalls             = None
    # lsu unit stalls             = None
    # csr unit stalls             = None
    # fpu unit stalls             = None
    # gpu unit stalls             = None
    # loads                       = None
    # stores                      = None
    # branches                    = None
    # icache reads                = None
    # icache read misses          = None
    # icache hit ratio perc       = None
    # dcache reads                = None
    # dcache writes               = None
    # dcache read misses          = None
    # dcache read hit ratio perc  = None
    # dcache write misses         = None
    # dcache write hit ratio perc = None
    # dcache bank stalls          = None
    # dcache utilization          = None
    # dcache mshr stalls          = None
    # smem reads                  = None
    # smem writes                 = None
    # smem bank stalls            = None
    # smem utilization            = None
    # memory requests             = None
    # memory reads                = None
    # memory writes               = None
    # memory average latency      = None



  def print_details(self):
    msg = ''
    msg += f'args    : {self.args}\n' 
    msg += f'app     : {self.app}\n' 
    msg += f'clusters: {self.clusters}\n' 
    msg += f'cores   : {self.cores}\n' 
    msg += f'warps   : {self.warps}\n' 
    msg += f'threads : {self.threads}\n' 
    msg += f'l2cache : {self.l2cache}\n' 
    msg += f'l3cache : {self.l3cache}\n' 
    msg += f'driver  : {self.driver}\n' 
    msg += f'perf    : {self.perf}\n'
    logger.info(msg)

def read_perf_log(path):
  with open(path, 'r') as fp:
    data= list(fp.readlines())

  df = pd.DataFrame( columns= [\
    'args',
    'app',    
    'clusters',
    'cores',  
    'warps',  
    'threads',
    'l2cache',
    'l3cache',
    'driver', 
    'perf',   
    'instrs',                    
    'cycles',                    
    'IPC',                        
    'ibuffer_stalls',            
    'scoreboard_stalls',         
    'alu_unit_stalls',           
    'lsu_unit_stalls',           
    'csr_unit_stalls',           
    'fpu_unit_stalls',           
    'gpu_unit_stalls',           
    'loads',                     
    'stores',                    
    'branches',                  
    'icache_reads',              
    'icache_read_misses',        
    'icache_hit_ratio_perc',     
    'dcache_reads',              
    'dcache_writes',             
    'dcache_read_misses',        
    'dcache_read_hit_ratio_perc',
    'dcache_write_misses',       
    'dcache_write_hit_ratio_perc',
    'dcache_bank_stalls',        
    'dcache_utilization',        
    'dcache_mshr_stalls',        
    'smem_reads',                
    'smem_writes',               
    'smem_bank_stalls',          
    'smem_utilization',          
    'memory_requests',           
    'memory_reads',              
    'memory_writes',             
    'memory_average_latency',    
  ])

  curr_d= {}
  for l_n, l in enumerate(data):
    if l.startswith('./blackbox'):
      if len(curr_d) > 10: # enough data, meaning that the './blackbox' line was not spurious
        if curr_d['instrs'] >= 0:
          df_curr= pd.DataFrame(curr_d, index= [0])
          df_curr= df_curr.reindex(df.columns,axis=1)
          df= pd.concat([df, df_curr], ignore_index= True)
          # print(curr_d)
          # print(df_curr)
          # print(df)
          # exit(1)

      curr_d= {}
      if '--perf' in l:
        curr_d['perf'] = True
      else:
        curr_d['perf'] = False

      if '--l2cache' in l:
        curr_d['l2cache'] = True
      else:
        curr_d['l2cache'] = False

      if '--l3cache' in l:
        curr_d['l3cache'] = True
      else:
        curr_d['l3cache'] = False

      l= l.split()
      curr_d['clusters'] = int(l[1].split('=')[-1])
      curr_d['cores'] = int(l[2].split('=')[-1])
      curr_d['warps'] = int(l[3].split('=')[-1])
      curr_d['threads'] = int(l[4].split('=')[-1])
      curr_d['driver'] = l[5].split('=')[-1]
      curr_d['app'] = l[6].split('=')[-1]
      if 'args' in l[7]:
        curr_d['args'] = l[7].split('=')[-1]

    elif l.startswith('PERF:'):
      if ' core' in l:
        continue # not logging core-level statistics yet
      if 'IPC' in l:
        l = l.split()
        curr_d['instrs'] = int(l[1].split('=')[-1].replace(',' , ''))
        curr_d['cycles'] = int(l[2].split('=')[-1].replace(',' , ''))
        curr_d['IPC'] = float(l[3].split('=')[-1])
      elif 'ibuffer stalls' in l:
        curr_d['ibuffer_stalls'] = int(l.split('=')[-1])
      elif 'scoreboard stalls' in l:
        curr_d['scoreboard_stalls'] = int(l.split('=')[-1])
      elif 'alu unit stalls' in l:
        curr_d['alu_unit_stalls'] = int(l.split('=')[-1])
      elif 'lsu unit stalls' in l:
        curr_d['lsu_unit_stalls'] = int(l.split('=')[-1])
      elif 'csr unit stalls' in l:
        curr_d['csr_unit_stalls'] = int(l.split('=')[-1])
      elif 'fpu unit stalls' in l:
        curr_d['fpu_unit_stalls'] = int(l.split('=')[-1])
      elif 'gpu unit stalls' in l:
        curr_d['gpu_unit_stalls'] = int(l.split('=')[-1])
      elif 'loads' in l:
        curr_d['loads'] = int(l.split('=')[-1])
      elif 'stores' in l:
        curr_d['stores'] = int(l.split('=')[-1])
      elif 'branches' in l:
        curr_d['branches'] = int(l.split('=')[-1])
      elif 'icache reads' in l:
        curr_d['icache_reads='] = int(l.split('=')[-1])
      elif 'icache read misses' in l:
        curr_d['icache_read_misses'] = int(l.split('=')[1].split()[0])
        curr_d['icache_hit_ratio_perc'] = int(l.split('=')[-1].split('%')[0])
      elif 'dcache reads' in l:
        curr_d['dcache_reads'] = int(l.split('=')[-1])
      elif 'dcache writes' in l:
        curr_d['dcache_writes'] = int(l.split('=')[-1])
      elif 'dcache read misses' in l:
        curr_d['dcache_read_misses'] = int(l.split('=')[1].split()[0])
        curr_d['dcache_read_hit_ratio_perc'] = int(l.split('=')[-1].split('%')[0])
      elif 'dcache write misses' in l:
        curr_d['dcache_write_misses'] = int(l.split('=')[1].split()[0])
        curr_d['dcache_write_hit_ratio_perc'] = int(l.split('=')[-1].split('%')[0])
      elif 'dcache bank stalls' in l:
        curr_d['dcache_bank_stalls'] = int(l.split('=')[1].split()[0])
        curr_d['dcache_utilization'] = int(l.split('=')[-1].split('%')[0])
      elif 'dcache mshr stalls' in l:
        curr_d['dcache_mshr_stalls'] = int(l.split('=')[-1])
      elif 'smem reads' in l:
        curr_d['smem_reads'] = int(l.split('=')[-1])
      elif 'smem writes' in l:
        curr_d['smem_writes'] = int(l.split('=')[-1])
      elif 'smem bank stalls' in l:
        curr_d['smem_bank_stalls'] = int(l.split('=')[1].split()[0])
        curr_d['smem_utilization'] = int(l.split('=')[-1].split('%')[0])
      elif 'memory requests' in l:
        curr_d['memory_requests'] = int(l.split('=')[1].split()[0])
        curr_d['memory_reads'] = int(l.split('=')[2].split(',')[0])
        curr_d['memory_writes'] = int(l.split('=')[3].split(')')[0])
      elif 'memory average latency' in l:
        curr_d['memory_average_latency'] = int(l.split('=')[-1].split()[0])
      else:
        assert 0, f"offending line: {l}"
    else:
      assert 0, f"offending line: {l}"
      
  return df

def ipc_w_threads(df):
  col_to_val_ls_d= {\
    'clusters': [1],
    'cores' : [1],
    'warps' : [32],
    'threads': [2,4,8,16,32],
    'app': ['bfs', 'vecadd', 'sgemm'],
    'args': ['cora.txt', \
      'pubmed_diabetes.txt',
      'p2p-Gnutella30.txt',
      'CA-HepPh.txt',
      'Cit-HepPh.txt',
      'Slashdot0811.txt',
      '-n2048',
      '-n4096',
      '-n16',
      '-n32',
      '-n64',
      ]
  }
  df_f= filter_df( df, col_to_val_ls_d)
  

  sns.set(style= 'white')
  sns.barplot(x= 'args', y= 'IPC', hue= 'threads', data=df_f)

  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.show()

  sns.set(style= 'white')
  sns.barplot(x= 'args', y= 'memory_average_latency', hue= 'threads', data=df_f)

  plt.show()

def plot_bfs_perf():
  path= '../ci/perf_regression.log'
  df= read_perf_log(path)
  
  # csv_path= '../ci/perf_regression.csv'
  # with open(csv_path, 'w+') as fp:
  #   logger.info(f'Writing dataframe to csv file: {csv_path}')
  #   df.to_csv(fp)

  ipc_w_threads(df)
  exit(1)

  col_to_val_ls_d= {\
    'clusters': [1],
    'cores' : [1],
    'warps' : [1,3,4,8,16,32],
    'threads': [2,4,8,16,32],
    'app': ['bfs'],
    'args': ['cora.txt', \
      'pubmed_diabetes.txt',
      'p2p-Gnutella30.txt',
      'CA-HepPh.txt',
      'Cit-HepPh.txt',
      'Slashdot0811.txt',
      ]
  }
  df_f= filter_df( df, col_to_val_ls_d)
  # print(df_f['IPC'].values[0])
  
  name_ls = [\
    'cora', 
    # 'pubmed_diabetes',
    # 'p2p-Gnutella30',
    # 'CA-HepPh',
    # 'Cit-HepPh',
    'Slashdot0811',
    # 'web-Google',
  ]

  clusters_ls = [1, 2]
  cores_ls    = [1, 2]
  warps_ls    = [1,2,4,8,16,32]
  threads_ls  = [2,4,8,16,32] # valid range: 2 to 32
  l2cache     = False
  l3cache     = False
  driver      = 'simx'
  perf        = True
  app_ls      = ['bfs']
  log_file    = "perf_regression.log"

def filter_df(df, col_to_val_ls_d):
  df_filtered= df
  for col, val_ls in col_to_val_ls_d.items():
    df_filtered= df_filtered[df_filtered[col].isin(val_ls)]

  return df_filtered


if __name__=='__main__':
  main()
