
# SoftMax Offloading

## SoftMax가 문제인 이유
Transformer attention
![[스크린샷 2026-04-24 오전 3.50.42.png]]

여기서 `QK^T`와 `PV`는 GEMM이라서 TCU가 처리할 수 있지만 `softmax(S)`는 안됨

### SoftMax
SoftMax는 각 row마다 다음을 연산

![[스크린샷 2026-04-24 오전 3.51.28.png]]
즉 각 row마다:
1. row maximum 찾기
2. 각 element에서 max 빼기
3. exponential 계산
4. row sum reduction
5. division / reciprocal multiply로 normalization
6. FlashAttention-style이면 이전 tile 결과를 rescale

`max`, `sum`은 row-wise reduction이고, `exp`는 transcendental function이라 TCU가 잘 못함

> NVIDIA도 softmax에서 `exp`가 SFU에서 실행되고, 이 때문에 attention block 안에서 matrix engine이 idle해지는 bottleneck이 생긴다고 설명합니다.

---
# Offloading 방법
> SoftMax Offloading은 attention에서 GEMM은 TCU에 맡기고, SoftMax같은 비-GEMM 작업은 별도 hardware 또는 별도 pipeline으로 빼서 처리하는 것

## A. Data movement offloading
> TMA 구현 

TMA를 통해 multi-dimensional tensor tile을 global memory에서 shared memory로 옮기는 address generation + copy를 hardware에 맡김. 

* TMA는 multi-dimensional array tile copy의 address calculation을 offload해서 global memory ↔ shared memory transfer를 효율적으로 진행함.
* NVIDIA Hopper도 TMA는 large block / multidimensional tensor를 global memory와 shared memory 사이에서 옮기며, single thread가 TMA operation을 issue하고 이후 address generation과 data movement는 hardware가 처리함

Vortex에서 TMA를 구현한다면 이건 우선적으로:
- `Q`, `K`, `V` tile load
- score tile / output tile store
- 2D/3D tensor stride address generation
- shared memory layout/swizzle
- async completion / barrier
를 hardware가 처리하도록 방향을 잡을 것.

## B. SoftMax compute offloading
> SIMT core의 ALU/FPU/SFU에서 softmax를 돌리는 대신, 별도 SoftMax unit 또는 activation unit을 만듬

전용 hardware가 다음 연산을 다 처리함
- rowmax reduction
- exp approximation
- rowsum reduction
- reciprocal / division approximation
- output normalization
- optional rescale for online softmax

이 경우 Matrix Unit 옆에 SoftMax Engine을 붙이거나, TMA와 shared memory 사이에 streaming SoftMax pipeline을 붙일 수 있음. 

ITA 같은 accelerator는 quantized transformer inference를 위해 integer-only softmax를 streaming mode로 계산해서 data movement와 energy를 줄이는 구조를 제안


## C. Pipeline offloading / overlap
> Virgo와 FlashAttention-3 구현 방식 

SoftMax를 dedicated hardware로 만들지 않고, **GEMM은 matrix unit이 돌리고 SoftMax는 SIMT core가 동시에 돌게 해서 softmax latency를 숨김**

* FlashAttention-3는 Hopper의 Tensor Core와 TMA asynchrony를 이용해서 data movement와 GEMM을 overlap하고, block-wise matmul과 softmax를 interleave함
* Virgo에서도 FlashAttention-3 mapping을 통해 두 GEMM은 matrix unit에 순차적으로 보내고, SoftMax는 SIMT core에 배치해서 concurrent execution 진행
	* SoftMax를 “matrix unit으로 offload”하지 않고, **matrix unit과 SIMT softmax를 software-pipelining으로 겹치는 구조**






---
# Virgo's TMA
> Dedicated DMA Engine

1. Ampere에서는 asynchronous data copy를 통해 global memory -> shared memory movement를 offload
2. Hopper에서는 TMA engine이 flexible address generation을 지원
3. Virgo에서는 cluster-level DMA를 두고 global memory, shared memory, accumulator memory 사이의 data movement를 맡긴다
	* API로는 `virgo_dma_load`, `virgo_dma_store`, `virgo_compute`, `virgo_fence`
4. FlashAttention mapping에서는 DMA가 다음 tile을 shared memory에 미리 load하고, matrix unit은 현재 tile을 compute하고, SIMT core는 softmax를 overlap






---
# Virgo TMA Implementation
### Virgo 클러스터 구성
> Virgo cluster = Vortex core + Gemmini matrix unit + SHM

```
README.md:11-13  
Gemmini matrix unit integration, shared memory, baseline Tensor Core models,  
memory coalescer, Vortex SIMT core integration 포함  
  
README.md:36-42  
RadianceCluster, RadianceSharedMem, GemminiTile, VortexCore, Barrier가 주요 module
```

**SIMT core + DTCU + shared memory  + memory movement path + barrier synchronizer**

### SHM이 Gemmini에 연결돼 있음

```
Configs.scala:207  
use_tl_ext_mem = true  
  
Configs.scala:210  
use_shared_ext_mem = true  
  
Configs.scala:218-219  
dma_maxbytes = site(CacheBlockBytes)  
dma_buswidth = dmaBytes  
  
Configs.scala:220-223  
tl_ext_mem_base = smKey.address  
sp_banks = smKey.numBanks  
sp_capacity = shared memory size
```

즉, Virgo의 Gemmini matrix unit이 private scratchpad만 쓰는 구조가 아니라, **cluster shared memory를 external memory처럼 보고 접근하도록 구성

1. Vortex core writes/reads shared memory  
2. Gemmini matrix unit reads/writes shared memory  
3. Gemmini DMA도 shared memory path에 연결

### GemminiTile 제어방식
> MMIO command register로 제어
> GemminiTile.scala에서 `TLRegisterNode`가 command register 역할

```
GemminiTile.scala:133-139  
TLRegisterNode 생성  
address = gemminiParams.slaveAddress
```

register map:
```
GemminiTile.scala:321-330  
  
0x00 -> command register  
0x10 -> rs1 low/high  
0x18 -> rs2 low/high  
0x20 -> busy register  
0x28 -> runningLoops register
```

즉 core는 일반 memory-mapped store/load로 accelerator command를 쓰고, busy/status를 읽는다.

`DTENSOR_START/POLL` custom instruction 방식은 다르지만, 개념적으로는 같음.

1. core가 command를 issue  
2. accelerator가 비동기로 수행  
3. core는 busy/status/fence로 확인

### 명령 실행 방식
> Virgo는 CISC-style command를 내부적으로 Gemmini instruction sequence로 변환
> 즉, GemminiTile은 단순히 raw Gemmini instruction만 받는 게 아니라, 상위 command를 받아 내부적으로 stride/bounds/accumulation 관련 instruction을 생성함

command id에 따라 여러 microcode sequence를 내부적으로 만듬
```
GemminiTile.scala:241-245  
compute on given hexadeciles  
  
GemminiTile.scala:260-263  
move out to scratchpad  
  
GemminiTile.scala:264-268  
load to scratchpad hexadeciles  
  
GemminiTile.scala:276-279  
store to gmem
```


**중요!!!!!!!!!!!!!!**
DTCU에서도 TMA를 만들 때 core가 모든 address를 계산하게 하지 말고, command/descriptor 하나만 주고 hardware FSM이 내부적으로 address sequence를 생성하게 하도록 구성해야함

### Virgo SHM (Bank/Sub-bank)
> Virgo의 SHM은 bank/subbank 구조로 구성
> `RadianceSharedMem.scala` 참조

```
RadianceSharedMem.scala:30-38  
smemBase, smemBanks, smemWidth, smemDepth, smemSubbanks 계산  
  
RadianceSharedMem.scala:48-86  
bank/subbank별 TLManagerNode 생성  
  
RadianceSharedMem.scala:122-180  
uniform read/write xbar 구성  
  
RadianceSharedMem.scala:312-321  
address format:  
[ smem_base | bank_id | line_id | word_id | byte_offset ]
```

Virgo는 shared memory address bit를 bank/subbank/line으로 쪼개서, matrix unit과 core의 access pattern을 같이 처리 => 이미 있는 coalescre랑 비슷함

### VirgoSharedMemComponents가 Gemmini와 core를 shared memory에 연결
> `VirgoSharedMemComponents.scala`참조

```
VirgoSharedMemComponents.scala:53-60  
radianceTiles의 smemNodes를 fanout  
  
VirgoSharedMemComponents.scala:84-105  
Gemmini wide access를 word-size subbank access로 distribute  
  
VirgoSharedMemComponents.scala:107-111  
Gemmini spad read/write nodes를 shared memory bank/subbank에 연결  
  
VirgoSharedMemComponents.scala:245-249  
gemmini.spad_read_nodes  
gemmini.spad_write_nodes  
gemmini.spad.spad_writer.node // this is the dma write node
```

Virgo source에서는 “TMA”라고 부르진 않지만, **Gemmini 내부 DMA write node가 shared memory interconnect에 연결되는 방식**




---
# DTCU 구현 방향

## Option A: Minimal DTCU-local TMA

GMEM/L2 -> TMA -> DTCU operand buffer  
DTCU execute  
DTCU accumulator/output buffer -> TMA -> GMEM/L2

**장점:**
- cluster shared memory를 새로 만들 필요가 적음
- 현재 DTCU가 이미 `operand buffer`, `accum_buf_` 같은 내부 buffer를 쓴다면 연결이 쉽움
- 현재 dtcu_basic/dtcu_compare를 확장하기 쉽움

**단점:**
- Hopper-style TMA의 핵심인 **global memory ↔ shared memory** 구조와 다름
- SIMT core가 shared memory에 있는 tile을 후처리하거나 softmax와 overlap하기 어려움
- Virgo/FlashAttention-style mapping과 거리가 있음

---
## Option B: Hopper/Virgo-style cluster shared memory + TMA

GMEM/L2 <-> TMA <-> Cluster Shared Memory  
                       ^  
                       |  
                    DTCU reads/writes  
                       ^  
                       |  
                    SIMT core reads/writes

이 구조에서는:
1. single thread가 TMA descriptor를 setup합니다.
2. TMA가 global memory에서 tile을 읽습니다.
3. TMA가 shared memory에 tile을 씁니다.
4. DTCU가 shared memory에서 A/B tile을 읽어 compute합니다.
5. DTCU output은 accumulator memory 또는 shared memory에 씁니다.
6. TMA가 output tile을 shared memory/accumulator에서 global memory로 store

선생님 연구 방향으로는 Option B가 더 좋습니다. 왜냐하면 나중에 SoftMax까지 하려면 SIMT core가 shared memory에 있는 score/probability/output tile을 만져야 하기 때문입니다.

```
TMA total latency =  
descriptor fetch/setup latency  
+ address generation latency  
+ memory request issue latency  
+ L2/cache/memory response latency  
+ shared memory write latency  
+ completion/barrier latency
```

---
## Option C: 내가 생각한 방식

### 1. 현재 DTCU 방식과 선생님 아이디어 비교

현재 DTCU 흐름이 대략 이렇다고 하면:
```
for each output tile (tile_m, tile_n):  
    clear accumulator  
  
    for each K tile (tile_k):  
        load A tile  
        load B tile  
        compute A*B and accumulate  
  
    store D output tile
```
현재 구조는 보통 serial합니다.
```
Load K0 -> Compute K0 -> Load K1 -> Compute K1 -> Load K2 -> Compute K2 -> Store D
```

이러면 `Load K1` 동안 DTCU compute datapath가 놀고, `Compute K0` 동안 memory path가 놀 수 있습니다.

선생님이 말한 TMA 방식은 이걸 이렇게 바꾸자는 겁니다.
```
Load K0  
Compute K0 + Load K1  
Compute K1 + Load K2  
Compute K2 + Load K3  
...  
Compute last K  
Store D
```

즉 pipeline으로 보면:
```
time ---->  
  
TMA:   Load K0 | Load K1 | Load K2 | Load K3 | ...  
DTCU:          | Comp K0 | Comp K1 | Comp K2 | Comp K3
```

이렇게 됩니다.

이건 매우 타당합니다. 오히려 DTCU에 TMA를 넣는다면 **제일 먼저 구현해야 하는 optimization**에 가깝습니다.

### 2. 정확히 어떤 data를 미리 읽는 건가?

한 output tile `(tile_m, tile_n)`을 계산한다고 하겠습니다.

GEMM은:

D[M, N] = A[M, K] * B[K, N]

DTCU tile이 예를 들어:

tileM x tileN x tileK

라면 한 output tile은 `tileM x tileN` 크기입니다.

그 output tile 하나를 만들기 위해 K dimension을 따라 여러 번 partial product를 누적합니다.
```
D_tile(m, n) += A_tile(m, k0) * B_tile(k0, n)  
D_tile(m, n) += A_tile(m, k1) * B_tile(k1, n)  
D_tile(m, n) += A_tile(m, k2) * B_tile(k2, n)  
...
```

그래서 각 K tile마다 필요한 operand는:
A tile: A[tile_m, tile_k]  
B tile: B[tile_k, tile_n]


따라서 TMA가 해야 할 일은 매 K iteration마다:
```
A_next = A base + tile_m offset + next tile_k offset  
B_next = B base + next tile_k offset + tile_n offset
```
을 계산해서 다음 A/B tile을 미리 DTCU buffer 또는 shared memory에 가져오는 것입니다.

### 3. 이 구조가 왜 좋은가?

현재 serial 구조에서는 한 K tile마다 latency가:
```
T_total_per_K = T_load_A + T_load_B + T_compute
```

TMA prefetch + double buffering을 하면 steady state에서는:
```
T_total_per_K ≈ max(T_load_A+B, T_compute)
```

즉 memory load latency와 compute latency가 겹칩니다.

### 4. 이걸 구현하려면 double buffer가 필요

현재 K tile을 compute하면서 다음 K tile을 load하려면 A/B operand buffer가 하나씩만 있으면 안 됩니다.

최소한 ping-pong buffer가 필요합니다.
```
A_buf[0], A_buf[1]  
B_buf[0], B_buf[1]
```

사용 방식은:
```
K0 load -> A_buf[0], B_buf[0]  
K0 compute uses A_buf[0], B_buf[0]  
K1 load -> A_buf[1], B_buf[1]  
  
K1 compute uses A_buf[1], B_buf[1]  
K2 load -> A_buf[0], B_buf[0]  
  
K2 compute uses A_buf[0], B_buf[0]  
K3 load -> A_buf[1], B_buf[1]
```

즉 buffer index는 보통:
```
cur_buf  = tile_k_idx % 2  
next_buf = (tile_k_idx + 1) % 2
```

### 5. DTCU FSM은 어떻게 바뀌어야 하나?

현재가 대략 이런 흐름이라면:

OP_REQ  
OP_WAIT  
EXECUTE  
OUT_REQ  
OUT_WAIT

TMA를 넣은 뒤에는 더 정확히 이렇게 나뉘는 게 좋습니다.

IDLE  
DESC_REQ  
DESC_WAIT  
  
PREFETCH_FIRST  
PREFETCH_FIRST_WAIT  
  
COMPUTE_AND_PREFETCH  
COMPUTE_WAIT / TMA_WAIT 관리  
  
NEXT_K_OR_OUTPUT  
  
OUT_REQ  
OUT_WAIT  
DONE

좀 더 구체적으로는:
1. descriptor 읽기  
2. output tile index 초기화  
3. accumulator clear  
4. K0 A/B tile TMA load 시작  
5. K0 load 완료 대기  
6. K0 compute 시작  
7. 동시에 K1 A/B tile TMA load 시작  
8. K0 compute 완료 + K1 load 완료 확인  
9. K1 compute 시작  
10. 동시에 K2 load 시작  
11. 반복  
12. 마지막 K tile compute 완료  
13. output store  
14. 다음 M/N output tile로 이동

중요한 점은 `compute_done`과 `tma_done`을 따로 관리해야 한다는 것입니다.

compute_done[cur_buf]  
tma_done[next_buf]

그리고 다음 compute로 넘어가려면:
* current compute가 끝났고  
* next buffer의 TMA load가 끝났어야 함


### 6. TMA가 담당해야 하는 address generation

현재 DTCU가 K tile마다 A/B address를 계산할 때, 이걸 사실상 공짜처럼 처리하고 있다면 정확한 timing model이 아닙니다.

TMA를 넣으면 address calculation은 TMA가 명시적으로 담당해야 합니다.

**A tile address**
A는 보통 `M x K`입니다.

현재 output tile이 `tile_m_idx`, 현재 K tile이 `tile_k_idx`이면:
```
A_tile_base =  
    A_base  
  + tile_m_idx * tileM * ldmA * elem_size  
  + tile_k_idx * tileK * elem_size
```

A tile 내부 element는:
```
A_addr(row, col) =  
    A_tile_base  
  + row * ldmA * elem_size  
  + col * elem_size
```

**B tile address**
B는 보통 `K x N`입니다.

현재 output tile이 `tile_n_idx`, 현재 K tile이 `tile_k_idx`이면:
```
B_tile_base =  
    B_base  
  + tile_k_idx * tileK * ldmB * elem_size  
  + tile_n_idx * tileN * elem_size
```

B tile 내부 element는:
```
B_addr(row, col) =  
    B_tile_base  
  + row * ldmB * elem_size  
  + col * elem_size
```

TMA는 이 주소들을 element 단위로 하나씩 내면 안 되고, cache line 단위로 묶어서 request를 만들어야 합니다.

for each row in tile:  
    compute row_start  
    compute row_end  
    compute touched cache lines  
    issue one MemReq per touched cache line

즉 TMA의 핵심은 단순히 `memcpy`가 아니라:

2D/3D tensor descriptor  
+ stride address generation  
+ cache line decomposition  
+ outstanding request tracking  
+ destination buffer write

### 7. DTCU가 직접 load하지 않고 TMA가 load하게 하면 뭐가 바뀌나?

TMA 적용 후에는 이렇게 바뀝니다.

DTCU:  
  tell TMA which A/B tile to load  
  compute previous tile  
  wait only if next tile is not ready  
  
TMA:  
  calculate A/B global address  
  issue L2 requests  
  fill next operand buffer  
  signal done

역할 분리가 명확해집니다.

DTCU = compute engine  
TMA  = tensor memory movement engine

이게 좋은 구조입니다.


### 8. “한 output tile의 K tile prefetch”는 Hopper TMA와 같은가?

개념적으로는 맞습니다. 다만 정확히 말하면 선생님이 말한 구조는 **Hopper TMA 자체**라기보다는, Hopper TMA를 이용한 **software/hardware pipelined GEMM data movement pattern**입니다.

Hopper에서 TMA는:
```
global memory -> shared memory  
shared memory -> global memory
```
의 multidimensional tensor copy를 hardware가 해주는 기능입니다.

그걸 GEMM kernel에서 쓰면 보통:

현재 K tile은 Tensor Core/WGMMA가 compute  
다음 K tile은 TMA가 shared memory로 load


선생님 DTCU에서는 shared memory가 아직 없다면:
```
global memory -> DTCU operand buffer
```
로 먼저 구현해도 됩니다. 이건 Hopper TMA보다 단순한 형태이지만, 연구/구현 시작점으로는 좋습니다.

나중에 shared memory를 추가하면:

global memory -> shared memory -> DTCU

로 확장하면 됩니다.

### 9. output store도 overlap할 수 있습니다

선생님이 말한 것은 K tile load overlap입니다. 이게 1순위로 맞습니다.

그 다음 optimization은 output tile store overlap입니다.

예를 들어 output tile `(m, n)`의 모든 K accumulation이 끝났으면, D tile을 store해야 합니다. 이 store도 TMA가 맡을 수 있습니다.

그러면 다음 output tile의 첫 K tile load와 이전 output tile store를 겹칠 수 있습니다.
```
Output tile 0:  
  compute all K  
  TMA store D0  
  
Output tile 1:  
  TMA load A/B K0  
  compute...
```

가능한 overlap:
```
TMA store D(m,n) + TMA load A/B for D(m,n+1)
```
다만 TMA가 single engine이면 load와 store가 경쟁합니다. 그래서 정책이 필요합니다.

예를 들면:
1. Load prefetch 우선  
2. Store는 compute가 길 때 background로 수행  
3. outstanding queue가 꽉 차면 store를 늦춤

### 10. 구현 난이도 기준 추천 순서

**Step 1: blocking TMA load**
먼저 TMA가 K tile 하나를 load하게만 만듭니다.
```
TMA_LOAD A tile  
wait  
TMA_LOAD B tile  
wait  
DTCU compute
```
이 단계는 overlap은 없지만 address generation과 MemReq 모델을 TMA로 분리할 수 있습니다.

**Step 2: ping-pong buffer 추가**
```
A_buf[2]  
B_buf[2]
```
를 추가합니다.

**Step 3: first tile preload**
```
load K0 into buffer 0  
wait K0
```

**Step 4: compute + next load overlap**
```
compute K0 using buffer 0  
load K1 into buffer 1
```

**Step 5: done condition 정교화**
다음 K로 넘어갈 때:
```
compute_done == true  
next_tma_done == true
```
를 둘 다 확인합니다.

**Step 6: performance counter 추가**
반드시 다음 counter를 넣는 게 좋습니다.
```
tma_addr_gen_cycles  
tma_mem_wait_cycles  
tma_smem_or_buffer_write_cycles  
dtcu_compute_cycles  
dtcu_wait_for_tma_cycles  
tma_wait_for_buffer_cycles
```
이걸 넣어야 “latency hiding이 실제로 됐는지” 보여줄 수 있습니다.

특히 제일 중요한 metric은:

dtcu_wait_for_tma_cycles

입니다.

이 값이 작아질수록 TMA prefetch가 잘 된 것입니다.


### Summary of Option C

현재 DTCU: 
```
K tile load와 compute가 serial함  
address generation/memory request timing이 단순화되어 있을 가능성이 큼  
```

제안 TMA:  
```
K tile별 A/B operand movement를 DTCU compute에서 분리  
descriptor 기반으로 다음 K tile address를 생성  
cache line 단위 MemReq를 issue  
ping-pong buffer에 next tile을 채움  
DTCU는 current tile compute와 next tile load를 overlap
```


스크립트 
We add a TMA-like tensor memory engine to the DTCU path.
Instead of letting the DTCU synchronously load each K tile before compute,
the TMA prefetches the next K tile using descriptor-based 2D address generation
while the DTCU computes the current K tile.
This converts the per-K-tile execution from load-then-compute serialization
into a producer-consumer pipeline with double-buffered operands.

# Option C와 Virgo TMA의 차이점
> Option C는 **DTCU-internal K-tile prefetch engine**에 가깝고, Virgo의 것은 **cluster-level DMA/TMA-style data movement engine**에 가까움


## 1. 공통점
> 둘 다 핵심 아이디어는 같음
> current K tile compute 중에 next K tile load를 미리 시작한다

즉 둘 다 목표는:
```
Load K0 -> Compute K0 -> Load K1 -> Compute K1
```

이 serial 구조를:
```
TMA:   Load K0 | Load K1 | Load K2 | Load K3  
DTCU:          | Comp K0 | Comp K1 | Comp K2
```
처럼 바꾸는 것입니다.

그래서 큰 방향에서는 선생님이 생각하신 방식이 Virgo/FlashAttention-3 스타일의 핵심과 맞습니다.

---

## 2. 가장 큰 차이: Virgo는 “software-visible DMA/TMA”, C는 “DTCU-internal prefetch”

### Virgo-style
Virgo에서는 kernel/SIMT core가 명시적으로 이런 식으로 제어합니다.
```
virgo_dma_load(...);   // next tile load  
virgo_compute(...);    // matrix unit compute  
virgo_fence(...);      // completion wait
```

즉 software가:
```
이번에는 어떤 tile을 shared memory에 load할지  
언제 matrix unit을 시작할지  
언제 fence/barrier를 걸지
```
를 직접 orchestration합니다.

구조적으로는:
```
SIMT core issues DMA command  
        ↓  
DMA/TMA moves GMEM -> shared memory  
        ↓  
Matrix Unit reads shared memory  
        ↓  
SIMT core can also read/write shared memory
```

---
### Option C

선생님 아이디어는 DTCU가 내부적으로 이렇게 하는 것입니다.
```
DTCU receives one GEMM descriptor  
DTCU internally iterates M/N/K tiles  
DTCU tells TMA to prefetch next K tile  
DTCU computes current K tile  
DTCU waits only if next K tile is not ready
```
즉 kernel 입장에서는 여전히:
```
dtensor_start(desc);  
dtensor_poll();
```
만 하면 됩니다.

내부에서는:
```
DTCU FSM + TMA FSM이 협력해서  
K-tile streaming을 자동으로 수행
```
합니다.

이건 software-visible TMA라기보다 **implicit TMA / autonomous prefetcher**입니다.

---

## 3. Virgo는 shared memory 중심, 선생님 방식은 DTCU operand buffer 중심일 수 있음

Virgo 쪽 data path는 기본적으로:
```
GMEM -> DMA/TMA -> shared memory -> Matrix Unit
```

shared memory가 중간에 있기 때문에 SIMT core도 같은 tile을 볼 수 있습니다. 그래서 FlashAttention처럼:
```
GEMM1 결과  
-> SIMT core가 softmax  
-> GEMM2가 다시 matrix unit에서 실행
```
같은 fused workload mapping이 가능합니다.

반면 선생님이 지금 말한 방식은 먼저 이렇게 시작할 가능성이 큽니다.
```
GMEM -> TMA -> DTCU A/B operand buffer -> DTCU compute
```
이 경우 장점은 구현이 단순합니다. DTCU 안의 기존 operand buffer를 ping-pong buffer로 만들면 됩니다.

하지만 단점은:
```
SIMT core가 그 tile을 shared memory처럼 재사용하기 어렵다  
SoftMax 같은 fused computation과 연결하기 어렵다  
Virgo-style cluster-wide collaboration과는 거리가 있다
```
입니다.

만약 선생님이 나중에 구조를:

GMEM -> TMA -> shared memory -> DTCU

로 만들면 Virgo와 훨씬 가까워집니다.

---

## 4. Virgo는 더 general-purpose, 선생님 방식은 GEMM-specific

Virgo DMA/TMA-style engine은 더 일반적인 tile movement 장치입니다.
```
global memory -> shared memory  
shared memory -> global memory  
accumulator memory -> global memory
```

그리고 GEMM뿐 아니라 FlashAttention 같은 fused kernel에서 여러 pipeline을 조합할 수 있습니다.

반면 선생님 아이디어는 현재로서는 GEMM의 K-loop에 특화되어 있습니다.
```
A[m_tile, k_tile] prefetch  
B[k_tile, n_tile] prefetch  
current K tile compute  
next K tile load
```

즉 더 specialized합니다.

이건 나쁜 게 아닙니다. 오히려 현재 DTCU 프로젝트에서는 specialized한 쪽이 더 자연스럽습니다. 왜냐하면 DTCU가 이미 GEMM tile iteration을 내부에서 하고 있기 때문입니다.

---

## 5. Virgo는 kernel이 scheduling하고, 선생님 방식은 hardware FSM이 scheduling합니다

이 차이가 매우 중요합니다.
### Virgo

Scheduling 주체:
```
kernel / programmer / compiler
```

예를 들어 kernel이 직접:
```
load next tile  
compute current tile  
softmax previous tile  
fence  
barrier
```
를 배치합니다.

**장점:**
* flexible함  
* GEMM, FlashAttention, fused activation 등 다양한 mapping 가능

**단점:**
* kernel이 복잡함
* software가 pipeline을 잘 짜야 함

---
### Option C

Scheduling 주체:
```
DTCU hardware FSM
```

즉 descriptor 하나를 주면 DTCU가 알아서:
```
M/N tile iterate  
K tile iterate  
next K tile prefetch  
current K tile compute  
output store
```

**장점:**
* kernel이 매우 단순함  
* instruction overhead가 작음
* DTCU 내부 timing model을 통제하기 쉬움

**단점:**
* GEMM 외 workload로 확장하기 어려움
* SoftMax, activation, custom fusion을 끼우기 어려움

---

## 6. 그래서 선생님 방식은 Virgo보다 “더 자동화된 GEMM engine”입니다

Virgo:  
  programmable cluster-level matrix/DMA system  
  
Option C:  
  autonomous GEMM engine with internal K-tile prefetch

Virgo는 accelerator를 software가 orchestrate합니다.

선생님 방식은 accelerator가 GEMM 전체를 거의 혼자 처리합니다.


---


## Hopper TMA

Hopper의 TMA는 **Tensor Memory Accelerator**입니다.

```
global memory에 있는 multidimensional tensor tile을  
shared memory로 옮기는 작업을  
hardware가 대신 해주는 engine
```

기존 방식에서는 여러 thread가 각자 address를 계산해서 load/store해야 합니다.

```
thread 0: addr 계산 + load  
thread 1: addr 계산 + load  
thread 2: addr 계산 + load  
```

Hopper TMA에서는 보통 **single thread**가 descriptor 기반 command를 issue합니다.

single thread:  
> "이 tensor의 이 tile을 shared memory로 가져와라"

그 후에는 hardware가:
```
1. tensor descriptor 읽기  
2. tile coordinate 기반으로 2D/3D address 계산  
3. global memory request 생성  
4. shared memory에 tile write  
5. completion barrier update
```

그래서 main benefit은:
```
address generation과 tile copy를 SIMT core에서 떼어낸다
```

입니다.

그리고 GEMM에서는 이렇게 씁니다.
```
TMA loads K tile k+1 into shared memory  
Tensor Core / WGMMA computes K tile k
```

즉 pipeline은:
```
TMA:       Load K0 | Load K1 | Load K2 | Load K3  
TensorCore:        | Comp K0 | Comp K1 | Comp K2
```
이런 식입니다.

---

## Virgo는 정확히 뭐냐면

Virgo에는 Hopper와 같은 이름의 `TMA`가 있는 것은 아니고, 논문에서는 주로 **DMA engine**이라고 부릅니다.

하지만 역할은 비슷합니다.

Virgo의 DMA는:
```
global memory ↔ shared memory  
shared memory ↔ accumulator memory / output
```
같은 tile movement를 담당합니다.

Virgo programming model에서는 kernel이 이런 식으로 씁니다.
```
virgo_dma_load(...)   // next tile load  
virgo_compute(...)    // matrix unit compute  
virgo_fence(...)      // wait for async operations
```

즉 Virgo에서는 SIMT core가 직접 tile data를 다 load하는 게 아니라, DMA에게 tile movement를 맡깁니다.

그리고 matrix unit은 current tile을 compute하고, DMA는 next tile을 load합니다.
```
DMA:         Load next tile  
Matrix Unit: Compute current tile  
SIMT Core:   synchronization / optional post-processing
```

Virgo의 중요한 차이는 **cluster-level matrix unit**과 같이 움직인다는 점입니다. 즉 data movement engine이 단순 copy만 하는 게 아니라, cluster-level shared memory와 matrix unit 사이의 pipeline을 구성하는 역할을 합니다.




Hopper TMA focuses on hardware-assisted tensor copy into shared memory.
Virgo uses a similar asynchronous data movement idea to feed a disaggregated cluster-level matrix unit.


The key idea from Hopper and Virgo is to decouple tensor tile movement from matrix computation. Instead of having the compute unit synchronously wait for every operand tile, a separate TMA/DMA-like engine prefetches the next tile while the matrix unit computes the current tile.

In our DTCU, this maps naturally to the K-tile loop. For a fixed output tile, the DTCU repeatedly accumulates over K tiles. Therefore, the TMA can prefetch A and B operands for the next K tile while the DTCU computes the current K tile.