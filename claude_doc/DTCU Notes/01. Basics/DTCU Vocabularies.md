---

---

---
# SIMT Related

## Thread
* 가장 작은 프로그래밍 단위
* 커널이 실행될 때 생기는 작업 하나
#thread

## Work Group / Thread Block
* Thread를 묶은 것
* 서로 협력해서 연산 실행
* 같은 block 속 threads는
	* - **shared memory**를 함께 쓸 수 있고
	- **barrier synchronization**을 할 수 있고
	- 하나의 block으로 스케줄됨

#work_group #thread_block #lane 

## Warp
* 하드웨어 실행 단위
* 32 Thread =  1 Warp
	* 예를들어 프로그래머가 256 thread 짜리 block을 만들었다면,
		* warp 0: thread 0~31
		- warp 1: thread 32~63
		- warp 2: thread 64~95
		- ...
	이렇게 잘라서 실행합니다.
* 스케줄러가 warp 단위로 issue함
* warp 단위로 같은 명령어를 같이 실행
	* 같은 warp 안에서 분기(`if`)가 갈라지면  **warp divergence**가 생겨서 성능이 떨어짐


| Thread   | 코드상 최소 단위     |
| -------- | ------------- |
| **Warp** | 실제 실행되는 최소 단위 |
#warp

## Lane
* 한 warp 안에서 자리 번호
* 예를 들어 Warp가 32개의 thread로 이뤄졌다면
	* lane 0
	- lane 1
	- lane 2
	- ...
	- lane 31
#lane


## Core (SIMT Core)
Warp scheduler + Execution units(ALU/FPU/LSU) + Register File
#core

## Cluster
여러 SIMT core를 묶은 상위 단위 + L2 Cache


---
# TCU Related
## FMA (Fused Multiply-Add)
- A * B + C를 한 번에 계산하는 연산
	- 각 K에 대해 A/B를 곱하고 C (accumulator)를 더하는 형태    
#fma
## FEDP
- 하나의 element를 만들기 위한 dot-product 계산기
	- **입력**: A의 한 row 조각 포인터, B의 한 col 조각 포인터, 그리고 C의 초기값
	- **출력**: 그 자리의 최종 D 값 (보통 fp32 값)
- FEDP(a_row, b_col, c_init)는 내부에서:
	1. Accumulator를 c_init로 초기화하고
	2. K를 돌면서 C(accumulator) = FMA(a_row[k], b_col[k], acc) 반복
	3. 최종 acc는 D (출력값)
- FMA가 *하나의* 연산을 뜻한다면 FEDP는 여러 스텝(dot product)을 구현
	- 따라서 FEDP 내부에서 FMA를 반복 호출  
#fedp
## Tile
- 하나의 wmma가 다루는 fragment의 크기
	- 보통 TCU는 엄청 큰 매트릭스를 한번에 연산하지 못함
	- 따라서 큰 매트릭스를 tile-granularity로 쪼개서 연산함
- M/N/K 값으로 정의됨
	- M×N - 한 번의 매트릭스 연산이 처리하는 출력 타일의 크기
	- K - 그 타일을 만들기 위해 곱해지는 길이 / Internal Size라고도 칭함
	- C[M×N] = A[M×K] × B[K×N]
- 우리 DTCU는 8×4×8의 타일 사이즈를 갖고 있음
	- 입력: 8×8 / 8×4
	- 출력: 8×8
- 구현 방식은 [[DTCU Basic Theory#Din]]
#tile

## GEMM
* d