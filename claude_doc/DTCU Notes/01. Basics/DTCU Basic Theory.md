
| **Vocabularies**      |
| --------------------- |
| [[DTCU Vocabularies]] |
목차

- [[#Tile|Tile]]
- [[#Thread 갯수에 따라 Tile값이 달라지는 이유|Thread 갯수에 따라 Tile값이 달라지는 이유]]
- [[#C Accumulation|C Accumulation]]
- [[#Row / Column Major|Row / Column Major]]
- [[#Systolic Array|Systolic Array]]
- [[#Cache on SimX|Cache on SimX]]
- [[#Barrier|Barrier]]
- [[#Software Pipelining|Software Pipelining]]
- [[#FMA / FEDP / WMMA|FMA / FEDP / WMMA]]


---
# Tile

## 타일링 이란?
Kernel이 큰 매트릭스를 타일 크기로 쪼개줌
* 큰 GEMM D=A×B(+C)에 대해서
* 입력 A/B와 출력 D를 tile size에 맞게 나누고, 각 타일에 대해 K 방향을 k씩 잘라서 반복 누적

## 타일링 후 연산
예를 들어, 4×4 matrix 두 개를 곱하고, tile size가 2×2×2,
![[스크린샷 2026-04-02 오전 1.31.44.png]] ![[스크린샷 2026-04-02 오전 1.31.53.png]]

출력 행렬 (C)는 다음과 같이 계산됨 : ![[스크린샷 2026-04-02 오전 1.32.28.png|164]]

- C11 타일을 계산하기 위해 4개의 잘게 쪼개진 입력 타일 (A10, B01, A11, B11)들을 연산함
- C11 연산에 필요한건 A의 bottom row / B의 right column들임
	- 이게 K-tile로 쪼갠거 (A는 M×K, B는 K×N)
	- 쪼개진 것들이 FEDP로 들어가서 연산이 된다
		- A10 × B01이 들어가면 각 element에 대해 dot product를 계산하고 accumulator에 더함

## K 타일에 대해서
> K는 internal size

같은 output tile 하나를 계산할 때, K dimension이 너무 크면 여러번 나눠서 계산해야됨

예를 들어,
- output tile 하나가 8x4이고, 전체 K가 16, tileK는 8
- output tile 하나를 계산할 때 K를 두 번 나눠서 계산해야 됨 (전체 K / tileK = 2)

 즉 하나의 output tile을 완성하기 위해 K 방향 partial product를 여러 번 더해야됨
 * 그렇기 때문에 **K-split partial sums에 대해 accumulation** 필요
 #k-tile
 
 #tile #tile_computation

---
# Thread 갯수에 따라 Tile값이 달라지는 이유
- 타일을 레지스터에 담아서 한 번에 처리하는데, 그 레지스터 용량이 ‘warp 안의 thread 수’에 비례해서 늘어나기 때문
	- DTCU의 경우에는 상관 X
- Total fragment registers = NUM_THREADS × registers_per_thread, so larger warps can hold and process larger tiles in one WMMA operation
#thread_on_tile 
---

# C Accumulation
> 최초 accumulation 값 지정용 (initial value”가 필요한 경우에만 쓰는 옵션)
> 내부적으로 큰 GEMM을 연산하는데 사용되는 accumulator (fragC)랑 헷갈리지 말 것!

원래 GEMM가 ![](data:image/png;base64,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)면, output tile 하나를 계산할 때도 시작값이 0이 아니라 C_tile임.

즉, 같은 output tile에 대해:
- 처음 accumulator가 0이 아니고 C라면, 
- C00 위에 첫 번째 K tile 결과 더하고, 두 번째 K tile 결과 더함
- 최종 결과를 D에 저장

### 이게 있는 이유
> 1개 이상의 DTCU를 필요로하는 엄청 큰 연산을 위해서 남겨놓음

하나의 DTCU call로 수행이 안될 때 (너무 큰 GEMM를 여러 단계로 쪼개야 할 때)
* 한 단계 결과를 다음 단계에 넘겨야 됨
* external accumulator chaining
* 여러 cluster의 dtcu 사용
* partial result를 어디선가 모아야됨
* distributed partial-sum interface 등등
#accumulation
---
# Row / Column Major
> 메모리에 1D로 저장된 element들을 저장하는 방식의 차이

| Row Major        | Column Major     |
| ---------------- | ---------------- |
| a, b, c, d, e, f | a, d, b, e, c, f |
![[스크린샷 2026-04-02 오전 1.53.51.png|94]]

우리 DTCU에서 B가 column-major, 나머진 row-major

## Leading Dimension
ldmA, ldmB는 보통 leading dimension(stride)
- row_major에서는 보통 ldm = Number of Columns
- col_major에서는 보통 ldm = Number of Rows

예를 들어 A가 M x K라면,
- row_major: ldmA = K
- col_major: ldmA = M
#ldm #leading_dimension #row_major #column_major
---
# Systolic Array
> 한 tile 크기의 PE를 격자 모양으로 깔고, 데이터가 오른쪽(A), 아래(B)로 흘러가며 연산 실행

* 타일링 후 systolic array위에서 계산

4×4 systolic array라면,
![[스크린샷 2026-04-02 오전 11.56.13.png|333]]
**PE(Processing Element)**는 다음 연산을 반복함
1. 왼쪽에서 A 값을 받음
2. 위에서 B 값을 받음
3. 둘을 곱함
4. 자기 accumulator에 더함
5. A는 오른쪽으로 넘김
6. B는 아래로 넘김

acc←acc+a×b

이게 바로 **MAC (Multiply-Accumulate)**

행렬 A의 값은 보통 **왼쪽에서 오른쪽으로** 흘러가고,  
행렬 B의 값은 보통 **위에서 아래로** 흘러갑니다.


예를 들어 2x2 systolic array라면,![[스크린샷 2026-04-02 오전 11.59.17.png|165]]

- A의 값은 **왼쪽에서 오른쪽으로**
- B의 값은 **위에서 아래로**


각 PE는 매 cycle마다 대충 이런 일을 합니다.
1. 왼쪽에서 A 하나 받음
2. 위에서 B 하나 받음
3. 둘을 곱함
4. 자기 partial sum에 더함
5. A는 오른쪽으로 넘김
6. B는 아래로 넘김

즉 각 PE는 사실상 이런 연산을 계속 합니다.

## FEDP와의 비교

### FEDP 방식
- **연산의 종류**입니다.
- 작은 `dot product + accumulate`를 합니다.
- 보통 **output 하나**를 만드는 primitive입니다.
- row 하나와 column 하나를 받아 scalar 하나를 냅니다.

**방식**
사람 한 명이 종이에 숫자를 보면서
- 하나 곱하고
- 더하고
- 또 하나 곱하고
- 더하고
### Systolic array 방식
- **하드웨어 조직 방식**입니다.
- 많은 PE를 2D mesh로 놓고 데이터를 흘려보냅니다.
- 여러 output element를 **동시에** 계산합니다.
- output tile 전체를 pipeline처럼 만듭니다.

**방식**
- A 데이터는 가로로 흐르고
- B 데이터는 세로로 흐르고
- 각 작업대가 동시에 곱해서 누적

#systolic_array

---
# Cache on SimX
>**cache_sim.cpp** - 실제 캐시 구현
>**sockets.cpp** 
>**cluster.cpp** - L2 Cache 구현체 위치 (line 46)

**L2 cache (cache_sim.cpp) does not actually transfer data to cores**
It's only for modelling transaction history / RW counts / cache hit/miss count / Number of delayed cycles

## 구현 방식
**mem_rsp_in** - signal that tells “event is complete” instead of actual data transfer

### Timing Path
* LSU creates MemReq, puts into L1/L2(cachesim)
* After a few cycle, issues MemRsp
* This mimics the delay between 계층

### Functional Data Path
* The actual memory data that LD has to bring is from memory
* Emulator::dcache_read
* mmu_.read()

> cache_sim은 transaction 히스토리(req/res 수, hit/miss, 시간)를 기록하고, 실제 코어에 데이터는 emulator.cpp가 RAM/MMU에서 읽어서 줌

---
# Barrier

## Local Barrier
> 여러 Warp가 같이 하는 작업을 할 때 쓰는 집합 대기선
    
예를들어,
* 1번이 안 끝난 warp도 있는데 누군가 먼저 2번으로 넘어가면 꼬임 => 중간에 “여기서 전부 모일 때까지 기다려”라는 지점
## Cluster Barrier
> 같은 cluster 안 여러 core가 모두 특정 지점에 도착할 때까지 기다렸다가, 전부 모이면 한꺼번에 풀어줌

DTCU가 cluster barrier를 사용해서 synchronization 구현함

## Barrier in SimX

**Inside the core**
- sim/simx/emulator.cpp - Counts if all warps inside the core have arrived
- barrier.arrival_count++
- If local barrier, compare # of arrived warps to count
- If global barrier, check if all active warps arrived in that core
- Cluster barrier라고 해도, 이 core 안의 참여 **warp들이 다 도착했는지를 먼저 체크한다**
    
**At the Cluster-level**
1. Core arrives at barrier => Mark as arrived
	- Turn on bit (barrier.mask.set(local_core_id))
2. Count # of arrived cores
	- if (barrier.mask.count() == count)
3. Release if all arrived
	- sockets_.at(s)->resume(c)
	- Execute all cores that were waiting
4. Reset for the next round
	- barrier.mask.reset()
	- barrier.phase++
    
**Phase**
> Tells the which # of round it is #phase  


**How Vortex tells if it’s global or local barrier**
- MSB of barrier id tells if it’s global or local
	- bool is_global = (bar_id >> 31)

#barrier

---
# Software Pipelining 
> Kernel을 이용해서 여러 코어가 같이 collaboration하는 것

[[DTCU Virgo#Software Pipelining]] 참고

예를 들면 kernel 안에서,
- 이번 tile compute
- 다음 tile load
- 그 다음 결과 store
를 **겹치도록 loop 구조를 짜는 것**


#Software_Pipelining 



---
# Dense Tile Buffer / Accumulator Buffer

### Dense Tile Buffer
> **A tile이나 B tile 전체를 연속적인 버퍼로 저장하는 것**

예를 들어 fp16 input에서 `64 x 128 x 16`이면,
- `A_tile[64][16]`
- `B_tile[16][128]`
같이 **tile 전체를 통째로 보관**하는 버퍼.

즉 dense tile buffer는:
- input operand staging용
- A/B를 lane별 fragment로 나누지 않고
- **행렬 tile 자체로 저장**하는 것입니다.

이 버퍼는 보통:
- load 후
- compute engine이 읽고
- 한 tile 계산이 끝날 때까지 유지됨


### Accumulator Buffer
> **partial sum / output tile 전용 버퍼**

예를 들어 `64 x 128` output tile이면,
- `accum[64][128]`
같이 전체 output tile의 누적합을 저장.

역할은:
- `tile_k = 0`일 때 0 또는 C preload로 초기화
- `tile_k = 1,2,...` 돌면서 계속 더함
- 마지막 K tile이 끝나면 D로 write-back

즉 accumulator buffer는 **K 방향 누적을 위한 저장소**

> Virgo 역시 **private accumulator memory**를 두고, loop가 돌면서 partial sum을 거기에 쌓고, 끝에 global memory로 내보냅니다.


![[스크린샷 2026-04-08 오전 3.23.29.png]]

### frag
- thread/lane에 나눠 가진 조각
- WMMA execution에 맞는 storage
- thread count와 강하게 연결됨
### dense tile buffer
- 입력 행렬 tile 전체 저장소
- thread와 무관한 tile-centric storage
- 주소 계산, variable shape 지원, footprint 계산이 쉬움
### accumulator buffer
- output tile 전체 누적합 저장소
- K loop 동안 계속 살아 있음
- 마지막에 D로 write-back

--- 
# FMA / FEDP / WMMA

## FMA (Fused Multiply-Add)
> 가장 작은 단위로, 숫자 3개를 이용하여 연산

![[스크린샷 2026-04-08 오전 3.20.50.png]]
## FEDP
> FMA보다 한단계 큰 연산으로, Output Element 하나에 대해 계산
![[스크린샷 2026-04-08 오전 3.20.41.png]]

([[#타일링 후 연산]] 참조)

- **FMA**는 곱하고 한 번 더하는 것 1회
- **FEDP**는 그런 FMA들을 여러 번 돌려서 **dot product 하나**를 만드는 것


## WMMA (Warp-level Matrix Multiply-Accumulate)
> API로 하나의 tile에 대해서 계산함

- fragment A/B/C를 레지스터에 준비하고
- custom instruction으로 TCU를 호출해서
- fragment D를 받는다.
