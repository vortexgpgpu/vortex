목차

- [[#DTCU_basic|DTCU_basic]]
- [[#DTCU_compare|DTCU_compare]]

# DTCU_basic
> Single-tile 검증용 테스트 케이스
## Abstract
* Checks for handshake
	* 커널이 DTENSOR_START로 DTCU 작업을 kick-off할 수 있는지
	* DTENSOR_POLL로 done을 확인할 수 있는지
	* 정상적으로 종료되는지
* 데이터 이동의 검증
	* Descriptor를 잘 읽는지
* 연산 검증
	* Output을 CPU가 연산한 값과 대조하여 무결성 검증


## Host (main.cpp)
>즉 Host는 행렬 준비(값 만들기, 메모리 저장, 실행, 검증)만 하고 하고, DTCU와 직접 통신하지 않음
  실제 DTCU 통신은 커널이 수행함

1. 입력 행렬 A,B를 CPU에서 생성
2. CPU에서 reference 결과(ref D)를 계산
3. Vortex runtime으로 device 열기 (vx_dev_open)
4. device memory 할당
	- vx_mem_alloc으로 버퍼 생성 (A_buf, B_buf, D_buf, desc_buf)
	- vx_mem_address으로 버퍼의 device virtual address를 얻음
5. descriptor를 host에서 구성해서 desc_buf에 복사
	- ptrA/ptrB/ptrD, ldmA/ldmB/ldmD, fmt_s/fmt_d, flags 설정
	- **DTCU가 descriptor를 읽는 방식(특히 ldm/format/flags)이 host와 맞춰야 됨
 6. 커널 인자 업로드 (kernel_arg_t) 및 args_buffer 생성 (vx_upload_bytes)
 7. 커널 바이너리(kernel.vxbin) 업로드 (vx_upload_kernel_file) → krnl_buffer
 8. 실행 (vx_start(device, krnl_buffer, args_buffer))
 9. D_buf에서 결과를 다시 읽어오고(vx_copy_from_dev), reference와 비교



## Kernel (kernel.cpp)
> 커널이 직접 A/B/D를 RW하지 않음
> DTCU가 descriptor 주소를 참조하여 실행함

1. 1개의 코어가 dtensor_start(desc_addr) 실행 
	- 이게 DTENSOR_START custom instruction을 내보내고 kick-off 시킴
2. done이 될 때까지 dtensor_poll()을 루프
	- DTENSOR_POLL custom instruction을 반복


## Microarchitecture (SimX)

### DTENSOR_START
* Kernel이 DTENSOR_START를 실행
* **DECODE** - 이 instruction을 **FUType::DTCU_Control**로 decode
* **EXECUTE** - 
	* core_id == 0 같은 조건을 걸어 “대표 core만” DTCU를 kick-off 
	* cluster->dtensor()->start(desc_addr) 호출
* **DTCU** - descriptor를 보고 cache에서 읽는 것처럼 모델링하고, 연산 후 다시 쓴다
	* 내부 state를 busy로 둠
	* descriptor를 읽기
		* 다음 tick에서 ram_->read(desc_, desc_addr, sizeof(desc_))
	* operand 읽기
	* 연산
	* output 쓰기


## Cache Lines on DTCU_basic
> `dtcu_basic`에선 총 6 cache line을 사용

**single tile이고 크기가 고정**
- descriptor: 64B → 1 line
- A: `8 x 8 x 2B = 128B` → 2 lines
- B: `8 x 4 x 2B = 64B` → 1 line
- C: 사용 안 함 (`flags=0x1`)
- D: `8 x 4 x 4B = 128B` → 2 lines
총 `1 + 2 + 1 + 2 = 6` 


---

# DTCU_compare



---
## Statistics

### 시간(host_ms)은 무엇을 재고 있나

지금 main.cpp에서 두 run 모두 timer는 정확히 이 구간만 잽니다.

- vx_start(...)
- vx_ready_wait(...)

즉 host에서 device memory를 allocate하고, A/B/C/D/descriptor를 vx_copy_to_dev로 올리는 시간은 둘 다 포함되지 않습니다.

“dtcu는 메모리에 operand 넣는 시간이 안 들어가고, in-core tcu는 memory에서 register로 복사하는 시간이 포함되는 거 아니냐”

### 맞지 않은 부분
device memory에 이미 올라간 A/B/C를 읽어서 연산하는 시간은 둘 다 포함됩니다.
- in-core TCU는 kernel 안에서 load_matrix_sync로 A/B/C를 읽습니다. 이건 당연히 vx_start ~ vx_ready_wait 안에 들어갑니다.  
- DTCU도 dtensor_start 이후 DTensorCore::tick() 안에서 descriptor fetch, operand line request, output line request를 다 수행합니다. 이것도 역시 vx_start ~ vx_ready_wait 안에 들어갑니다.  

즉 device DRAM/L2에서 operand를 가져오는 시간 자체는 두 방식 모두 측정 구간 안에 있습니다.

### 맞는 부분
host → device upload 시간은 둘 다 안 들어갑니다.

즉 지금 host_ms는
- “전체 애플리케이션 체감시간”  
- “호스트가 입력을 준비해서 올리고, 커널 돌리고, 결과를 받고, 종료하는 시간”  
이 아니라,
- “입력이 이미 device에 있다고 가정했을 때의 kernel completion latency”  
      
이건 공평합니다. 다만 무엇을 재는지 정의가 그거일 뿐입니다.

---

## 2) DTCU의 memory fetch가 실제로 시간에 들어가느냐

네. 들어갑니다.

사용자님이 올리신 d_tensor_core.cpp를 보면, DTCU는 내부에서 그냥 ram_->read()만 하는 게 아니라, 그 전에 명시적으로 memory request를 발행하고 그 응답을 기다립니다.

핵심 흐름은 이렇습니다.
- DESC_REQ에서 descriptor line request 발행  
- OP_REQ / OP_WAIT에서 operand line request들을 순차 발행하고 응답 대기  
- 그 다음에 load_operands()로 실제 unpack  
- OUT_REQ / OUT_WAIT에서 output line request 발행  
- 그 다음에 store_output()  

즉 DTCU가 operand를 device memory에서 가져오는 latency 모델은 분명히 실행 시간에 포함됩니다.  
이 점에서는 in-core TCU와 비교 자체는 성립합니다.

---

## 3) 그렇다면 지금 시간 비교는 공평한가

### 네, 공평한 부분

“device에 데이터가 이미 올라가 있는 상태에서, 연산을 끝내기까지 걸리는 시간”을 비교한다는 의미에서는 공평합니다.

즉 지금 host_ms, cycles는 둘 다 다음을 포함합니다.
- kernel launch 이후  
- device-side A/B/C fetch  
- compute  
- D writeback  
- completion  

그래서 “device resident workload latency” 비교로는 괜찮습니다.

---

## 4) 하지만 완전히 apples-to-apples는 아닌 부분

여기부터가 중요합니다.

### A. DTCU는 cache-line level 모델, in-core TCU는 instruction-level LSU path

in-core TCU는 kernel이 직접:
- load_matrix_sync
- mma_sync
- store_matrix_sync

를 수행합니다. 즉 memory access가 일반 core/LSU/TCU path로 들어갑니다.

반면 DTCU는:
- descriptor 하나를 받고
- 내부에서 A/B/C/D에 대해 cache-line request를 abstract하게 발행
- 그 뒤 ram_->read/write로 functional data를 조립/저장

즉 DTCU는 accelerator-level line traffic 모델,  
in-core TCU는 core instruction path 모델입니다.

그래서:
- cycles
- host_ms
- L2 / memory request
- final output correctness

비교는 의미가 있습니다.

하지만
- loads
- stores
- instr_lsu
- instr_tcu

같은 core instruction 카운터를 그대로 DTCU와 1:1 비교하는 건 적절하지 않습니다.

왜냐하면 DTCU는 많은 일을 core instruction이 아니라 DTCU 내부 state machine이 하기 때문입니다.

즉 지금 출력에서 loads=0, stores=0, instr_tcu=0 같은 값이 이상하게 보이는 건 자연스러운 부분이 있습니다.  
그 카운터들은 DTCU 비교의 핵심 지표로 보기 어렵습니다.

---

### B. DTCU의 custom L2 MemReq count와 perf counter는 같은 숫자가 아닙니다

지금 로그에 DTCU custom print가 나옵니다.
- desc=1, op=64, output=16, total=81
    
그런데 perf summary의 l2cache reqs는 402입니다.

이건 이상한 게 아니라, 두 숫자가 정의가 다르기 때문입니다.

- DTCU custom count
- descriptor / operand / output에 대해
- DTCU가 의도적으로 모델링한 unique cache-line level request  
- perf L2 req
- 시스템 전체 실행 중 발생한
- core side traffic + DTCU side traffic + 기타 runtime traffic이 섞인 값

즉 DTCU 내부 print의 81을 in-core TCU의 l2_reads + l2_writes와 직접 비교하면 안 됩니다.

비교용으로는 오히려 stats_tcu.l2_reads + l2_writes vs stats_dtcu.l2_reads + l2_writes처럼  
같은 counter 체계끼리만 비교해야 합니다.

---

### C. 지금 cycles에는 DTCU의 polling overhead도 조금 섞입니다

DTCU path의 kernel은 사실상:
- warp 0 / thread 0이 dtensor_start
- dtensor_poll busy wait

구조입니다.

즉 DTCU run의 cycles에는:
- DTCU 실제 연산 시간
- 그걸 기다리는 core-side polling

반면 in-core TCU는 worker blocks가 직접 compute를 수행합니다.

이건 엄밀히 말하면 구조적 차이입니다.  
하지만 사용자님 아키텍처 자체가 원래 그런 실행 모델이라면, 그 구조를 포함한 latency 비교로 보는 게 맞습니다.

즉 이건 “불공평한 버그”라기보다는 설계 차이의 일부입니다.

---

## 5) 지금 통계들 중 무엇이 믿을 만한가

### 비교에 가장 적합

- host_ms  
    cycles
- l2_reads + l2_writes
- mem_reads + mem_writes
- correctness (CPU ref, cross-check)

이건 현재 테스트 목적에 잘 맞습니다.

### 해석은 가능하지만 조심해야 함

- stall_lsu
- stall_tcu

이건 DTCU와 in-core TCU가 내부 실행 방식이 달라서, 절대치 비교는 약간 조심해야 합니다.

### 지금 비교 지표로는 부적절하거나 의미가 약함

- loads
- stores
- instr_lsu
- instr_tcu

특히 DTCU에서는 이 값들이 “accelerator work”를 제대로 대표하지 않습니다.

---

## 6) 지금 테스트가 사용자님 목적에 대해 충분히 공평한가

### 네, 공평한 부분

사용자님이 지금 하려는 건 대략 이런 질문입니다.

“같은 큰 GEMM (D = A * B + C)를 할 때,  
in-core TCU와 DTCU 중 누가 더 적은 system-level time / cycle / memory traffic으로 끝내는가?”

이 질문에 대해서는 현재 테스트가 꽤 공평합니다.

이유는:

- 같은 A/B/C    
- 같은 output D
- 같은 GEMM shape
- 같은 CPU reference
- 같은 run harness
- 같은 vx_start ~ vx_ready_wait
- 같은 memory hierarchy 위에서 실행
    

### 하지만 “instruction-level microarchitecture fairness”는 아닙니다

DTCU와 in-core TCU는 원래 execution model이 다르므로,  
instruction counters까지 똑같은 잣대로 보면 안 됩니다.

즉 지금 테스트는
- system-level workload comparison으로는 좋고,
- micro-op level apples-to-apples comparison으로는 아닙니다.

---

## 7) 제가 권하는 정리

지금 이 테스트에서 발표/보고할 때는 이렇게 말하는 게 가장 정확합니다.

### 시간
- host_ms와 cycles는 kernel launch 이후 completion까지의 device-side execution latency
- host-to-device input upload 시간은 둘 다 제외
- device memory에서 operand를 가져오는 시간은 둘 다 포함  
### traffic
- L2 / memory counters는 system-level traffic 비교용
- DTCU custom L2 MemReq count는 DTCU 내부 모델 sanity check용이지, in-core TCU counter와 직접 비교용은 아님
### counter selection
- 핵심 비교 지표는
	- cycles
	- host_ms
	- l2_total
	- mem_total
- loads/stores/instr_*는 보조적 정보로만 사용

---

## 최종 결론

지금 테스트는 “kernel 실행 구간만 비교한다”는 정의 아래에서는 공평합니다.  
특히 사용자님이 걱정한 device memory에서 operand를 가져오는 시간은 DTCU와 in-core TCU 둘 다 포함됩니다.

다만 완전히 주의해야 할 점은 세 가지입니다.

1. host upload 시간은 둘 다 빠져 있다  
    → end-to-end 시간이 아니라 execution latency다
    
2. DTCU는 line-level accelerator model, in-core TCU는 core instruction path  
    → loads/stores/instr_* 같은 core counter는 직접 비교하기 부적절하다
    

DTCU custom memreq print와 perf L2/memory counter는 다른 계측 체계다  
→ 같은 종류의 counter끼리만 비교해야 한다