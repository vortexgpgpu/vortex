
import os
import sys
import subprocess
import logging

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
  perf_regression()

def perf_regression():
  
  args_ls_d = {\
    'bfs' : [\
      # 'cora.txt', 
      # 'pubmed_diabetes.txt',
      # 'p2p-Gnutella30.txt',
      # 'CA-HepPh.txt',
      # 'Cit-HepPh.txt',
      'Slashdot0811.txt',
      # 'web-Google.txt',
    ],
    # 'vecadd' : ['-n128','-n2048', '-n4096', '-n8192'],
    'vecadd' : ['-n65536'],
    'sgemm' : ['-n8','-n16', '-n32', '-n64'],
    'saxpy' : ['-n128','-n1024', '-n2048', '-n4096', '-n8192'],
  }


  clusters_ls= [1]
  cores_ls= [2]
  # warps_ls= [1,2,4,8,16,32]
  # threads_ls= [2,4,8,16,32] # valid range: 2 to 32
  warps_ls= [32]
  threads_ls= [32] # valid range: 2 to 32
  l2cache= True
  l3cache= False
  driver='simx'
  perf=True
  app_ls=['bfs'] 
  # app_ls=['vecadd'] 
  # app_ls=['sgemm', 'vecadd', 'bfs'] 
  log_file= "perf_regression.log"

  for clusters in clusters_ls:
    for cores in cores_ls:
      for warps in warps_ls:
        for threads in threads_ls:
          for app in app_ls:
            args_ls = args_ls_d[app]
            for args in args_ls:
              cmd= f"./blackbox.sh --clusters={clusters} --cores={cores} --warps={warps} --threads={threads} --driver={driver} --app={app} --args={args}"
              if l2cache:
                cmd += " --l2cache"

              if l3cache:
                cmd += " --l3cache"

              if perf:
                cmd += " --perf"

              post_process = f' | grep "PERF:" >> {log_file}'

              cmd += post_process

              logger.info(cmd)
              os.system(f'echo "{cmd}" >> {log_file}')
              os.system(cmd)


if __name__=='__main__':
  main()
