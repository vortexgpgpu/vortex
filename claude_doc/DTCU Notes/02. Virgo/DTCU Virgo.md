# What is Virgo?
> Implements disaggregated tensor core on cluster, instead on in-core

**Impact of DTCU**
- Reduced register pressure (on RF) / energy issue
	- No need to store matrix before operation
- Better scalability

최신 NVIDIA (Hopper)는 wgmma 이용함
* Read matrix operands directly from SHM
- Still does not solve register pressure / energy issue
	- wgmma reduces some LD, but still accumulates partial sum matrices back to the RF (Use registers as intermediate)

TBD



Virgo는 matrix unit을 MMIO로 제어

완료 여부는 software가 busy register를 polling해서 안다

Same as dtensor_poll기반


![[스크린샷 2026-04-03 오전 2.25.37.png]]

# Synchronization
virgo_fence()

1. matrix unit launch
2. core는 non-blocking하게 다음 작업 가능
3. 필요 시 virgo_fence로 이전 async op 완료 확인
	- virgo_fence blocks the warp until all preceding asynchronous operations have completed and their results are visible to the programmer
4. Cluster-wide barrier로 participating core들 시점 맞춤

# Software Pipelining

Virgo-optimized GEMM kernel employs software pipelining which is enabled by the asynchronous programming interface of the matrix unit. 

As shown in Figure 4, while the matrix unit is computing a tile along the K dimension consisting a consumer pipeline, either the **DMA unit or a set of SIMT core warps collaboratively fetch the next input tile along the K dimension** from the global memory to the shared memory, consisting the producer pipeline. 

Another set of SIMT core warps can collaborate to form an additional consumer pipe that does post-processing activation com- pute on a previous result tile. Because both the producer and consumer pipes run in parallel, the tile data are double- buffered in the shared memory. This mechanism allows both the SIMT core warps and the matrix unit to participate in useful work at all times, maximizing utilization of all hardware components in the cluster.

#Software_Pipelining



# Thread Block 협업방식
Virgo에서는 matrix unit이 core 안의 작은 warp-local unit이 아니라, **cluster-level의 별도 unit**입니다. 그래서 GEMM 한 번을 돌릴 때 core 쪽이 해야 할 일이 여전히 남습니다.

- global memory → shared memory 데이터 이동
- matrix unit이 낸 결과에 대한 post-processing
- 다음 tile 준비
- barrier / fence를 통한 순서 맞추기

그래서 논문은 **“multiple warps collaborate to participate in a single operation of the matrix unit”**라고 말하고, Figure 4에서는 **“all warps in the thread block”**가 이전 `(M,N)` output tile의 activation을 담당하는 예를 보여줍니다.
![[스크린샷 2026-04-08 오전 1.16.03.png]]

즉 idea는 이겁니다.

- **matrix multiply 자체**는 cluster-level matrix unit이 함
- **주변 작업들**은 thread block 전체 warps가 분담해서 함

---

## 2. thread block이 실제로 어떻게 협업하나

논문을 그대로 풀면 협업은 세 층으로 되어 있습니다.

### (1) 배치 단위: thread block/workgroup 자체가 cluster에 붙음

논문은 cluster가 **thread block 또는 workgroup이 할당되는 하드웨어 단위**라고 설명합니다. 즉 Virgo는 thread block이 cluster 안 여러 core에 걸쳐 실행되는 것을 전제로 합니다.

그래서 “thread block 전체가 하나의 matrix operation에 참여한다”는 말이 성립합니다.  
warp들이 같은 core에만 있지 않고 **cluster 내 여러 core에 퍼져 있을 수 있기 때문**입니다. 이 점 때문에 별도의 **cluster-wide synchronizer**도 필요하다고 Section 3.3에서 설명합니다. warp scheduler가 barrier release request를 보내고, synchronizer가 **모든 core에서 요청을 모아** barrier를 풀어 줍니다.

---

### (2) 역할 분담: matrix unit / DMA / SIMT warps가 파이프라인으로 나뉨

Virgo의 핵심은 **한 thread block 안의 warps가 모두 똑같은 일만 하지 않는다는 점**입니다.

논문 4.4.2는 다음처럼 설명합니다.

- matrix unit은 현재 K tile을 계산하고,
- **DMA 또는 “a set of SIMT core warps”** 가 다음 K tile을 global memory에서 shared memory로 가져오고,
- **another set of SIMT core warps** 가 이전 결과 tile에 대해 activation/post-processing을 합니다.

즉 그림으로 쓰면:

- **producer pipe**: 다음 input tile 준비
- **consumer pipe 1**: matrix unit이 현재 GEMM
- **consumer pipe 2**: 이전 output tile post-processing

이 세 개가 겹칩니다. 논문은 이것을 software pipelining + double buffering이라고 부릅니다.

여기서 중요한 점은, **“몇 warp가 producer냐”가 hardware에 고정된 숫자라는 말은 없습니다.**  
그건 kernel mapping / library 설계가 정하는 쪽입니다. 논문도 “a set of SIMT core warps”라고만 씁니다. 즉 **고정 비율이 아니라 workload-dependent한 역할 분담**입니다.

---

### (3) 동기화 방식: barrier + fence

이렇게 역할이 나뉘면 반드시 순서를 맞춰야 합니다. 그래서 Virgo는 두 가지를 씁니다.

- **cluster-wide barrier**
- **virgo_fence**

논문 low-level API는 `virgo_dma_{load,store}`, `virgo_compute`, `virgo_fence`를 정의하고, 이 API들이 **cluster-wide barrier와 함께 사용되어** participating cores를 synchronize한다고 설명합니다.

즉 협업 방식은 대충 이런 흐름입니다.

1. thread block 전체가 barrier에서 맞춤
2. 어떤 warp/warps가 `virgo_compute`로 matrix unit kick
3. 다른 warp set이 `virgo_dma_load`나 직접 SMEM data movement
4. 또 다른 warp set이 이전 output tile activation
5. `virgo_fence`로 matrix 결과가 보일 때까지 동기화
6. 다음 iteration으로 진행

논문 FlashAttention pseudocode도 이 패턴을 그대로 보여줍니다:

- `virgo_fence(0)`
- `threadblock_barrier()`
- `virgo_compute(...)`
- `virgo_dma_load(...)`
- SIMT softmax compute
- `virgo_fence(2)`  
    이런 순서입니다.

---

## 3. 그럼 `128×64×128`과 thread block은 정확히 어떻게 연결되나

여기서 가장 조심해서 말해야 합니다.

논문이 **직접 말한 것**은:

- Virgo matrix unit이 **`128×64×128` tile size of a single operation**을 expose하고,
- 이것이 **thread block size를 결정**한다는 것입니다.

그런데 논문이 **말하지 않은 것**은:

- `128×64×128`이면 반드시 thread가 몇 개인지
- row 방향 몇 thread, col 방향 몇 thread로 나누는지
- 한 warp가 결과 몇 row를 맡는지

같은 **정량적 mapping 공식**입니다.

즉 논문은 “tile이 크기 때문에 계산 atom이 threadblock/workgroup-level이 된다”는 architectural principle은 명확히 주지만,  
**“128×64×128 → blockDim=(x,y,z)”** 같은 exact kernel launch recipe는 제시하지 않습니다.

---

## 4. 그래도 논문에서 유추 가능한 숫자 감각은 있다

논문 Table 2 evaluation config를 보면 shared SoC config로

- **SIMT Width: 8 warps/core, 8 lanes/warp**
- Virgo는 **1 matrix unit per cluster**
- shared memory 128KB
- Virgo matrix unit은 **16×16 FP16 systolic array / 8×8 FP32**  
    라고 적혀 있습니다.

여기서 유추할 수 있는 건:

- cluster 안에는 여러 core가 있고,
- 각 core마다 여러 warps가 있으며,
- 그 전체가 하나의 cluster-level matrix unit을 함께 먹여 살리는 구조라는 점입니다.

즉 Virgo는 **warp 하나가 자기 register만 들고 tile 하나를 해결하는 구조가 아니라**,  
**cluster 내 다수 warps가 shared memory를 매개로 tile을 공급하고 후처리하는 구조**입니다. 이 점에서 현재 Vortex WMMA-style per-warp tile model과 철학이 다릅니다.

---

## 5. 아주 쉽게 비유하면

Virgo의 `128×64×128` tile 하나를 처리할 때 thread block 협업은 이런 느낌입니다.

- **matrix unit**: “나는 128×64 output tile의 GEMM을 한다.”
- **warp group A**: “다음 K chunk의 A/B를 shared memory에 채워 넣는다.”
- **warp group B**: “방금 계산 끝난 이전 output tile에 activation / rescale 같은 element-wise 작업을 한다.”
- **전체 thread block**: barrier/fence로 타이밍을 맞춘다.

즉 **tile 크기 때문에 thread block이 커지는 것**이 아니라,  
**tile이 커서 warp 하나로는 주변 일을 감당할 수 없으니 thread block 전체를 계산 단위로 쓰는 것**에 가깝습니다.

---
# 내부 Storage 방식

Virgo cluster 안에는
- **Scratchpad**
- **Systolic Array**
- **Accum. Memory**
- **Coarse-Grain FSM**

Virgo matrix unit 자체가 **private scratchpad + private accumulator memory**를 가진 구조.

Gemmini 기반 matrix unit이
- source matrices를 **private wide scratchpad banks**에서 움직이고,
- accumulation은 **individual PEs 또는 private accumulator memory**에서 하며,
- Virgo에서는 **accumulator memory를 retained**.

그래서 Virgo를 따라가려면
- **dense tile buffer에 해당하는 것**  
    → Gemmini의 **private wide scratchpad banks**
- **accumulator buffer에 해당하는 것**  
    → Virgo의 **private accumulator memory**


특히 논문은 Virgo가 operand와 accumulator를 **core register file에서 완전히 분리** 

operand는 shared memory 쪽에서 끌어오고, accumulator는 separate accumulator memory에 둬서 register pressure를 없앱니다.

**tile shape를 thread가 정의하지 않고**, matrix unit 내부에 **operand staging storage + accumulator storage**를 둠.

![[스크린샷 2026-04-08 오전 3.32.15.png|632]]