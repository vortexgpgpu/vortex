
# Grid
# Host-side Grid

  

이건 host 코드에서 kernel launch할 때 넘기는 launch shape입니다.

예를 들면 지금 dtcu_basic/main.cpp:168-172에 있는

- karg.grid_dim[0] = num_cores
    
- karg.grid_dim[1] = 1
    
- karg.block_dim[0] = NUM_THREADS
    
- karg.block_dim[1] = 1
    

이 값들이 바로 host 쪽에서 정한 launch configuration입니다.

즉 host가 “이번 kernel을 block 몇 개로 띄울지”, “각 block 안에 thread 몇 개 넣을지”를 정해서 kernel_arg_t 안에 담아 kernel로 넘기는 것입니다.

host-side grid = main.cpp에서 설정하는 karg.grid_dim[]

  

# Kernel-side Grid

= block(group)의 개수입니다.

vx_spawn_threads() 구현을 보면:

- num_groups = grid_dim[0] * grid_dim[1] * grid_dim[2]
    
- group_size = block_dim[0] * block_dim[1] * block_dim[2]  
      
    

Vortex에서:

- grid_dim = 몇 개의 work-group / block을 만들 것인가
    
- block_dim = 각 group 안에 몇 개 thread를 둘 것인가  
      
    

예를 들어 grid_dim = {8, 1} / block_dim = {4, 1} 이면:

- block 8개
    
- 각 block에 thread 4개
    
- 총 logical thread 수는 32개