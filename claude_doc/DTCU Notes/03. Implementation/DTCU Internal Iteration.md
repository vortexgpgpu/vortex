---
share_link: https://share.note.sx/t1ekxbzo#bZ0KIsLGGuU80eC5/c6loLgYiKjMsTf+s6e6ghG7Das
share_updated: 2026-04-08T13:42:31-04:00
---

> 큰 GEMM이 있으면 DTCU 내부적으로 loop를 돌아서 알아서 쪼개줌 (decomposition)
> ITCU는 kernel이 쪼개주고 뿌려줌


# Abstract
## Virgo 방식
Virgo에서도 DTCU가 자체적으로 decomposition을 진행
* 내부에 coarse-grain FSM이 있어서 **`i, j, k`를 자동으로 iterate**
* Kernel 쪽에서 DMA, barrier, fence를 이용하여 kernel-level 타일크기에 맞춤
* 즉 Virgo는 native tile 하나를 크게 잡고, **더 큰 GEMM은 native tile을 반복함**.

## 우리 방식
- **바깥쪽**: 큰 GEMM을 `tile_m_idx_ / tile_n_idx_ / tile_k_idx_`로 자름 (아래 2번 해당)
- **안쪽**: 잘린 native tile 하나를 기존 WMMA register/fragment step으로 다시 나눠서 계산

1. Descriptor를 읽어서 전체 Matrix 크기를 구함
2. 전체 M, N, K 값을 DTCU native tile 크기로 나눠서 각 dimension별 필요한 tile 수 계산
	1. `init_tile_state_()`
3. 한번에 한개의 output tile `(tile_m_idx_, tile_n_idx_)`에 대하여 연산 실행 
	1. 해당 output tile에 대해 `tile_k_idx_ = 0 ~ tiles_k_-1` 를 모두 순회하면서 accumulate
		1. 현재 연산할 tile의 A/B/C/D base address를 계산
			1. `tile_ptrA_()` `tile_ptrB_()` `tile_ptrC_()`
		2. 해당 타일의 data 읽음
			1. `load_operands()`
		3. WMMA 연산
			1. `execute_wmma()`
	2. K가 남았으면 1번 다시 해서 모두 순회![[스크린샷 2026-04-08 오전 12.55.30.png]]
4. K가 끝나면 output tile 하나를 `D`의 올바른 위치에 저장
	1. `store_output()`
5. 다음 output tile로 이동
    - 먼저 `tile_n_idx_++`로 하나씩 increment
    - N이 끝나면 `tile_n_idx_=0`으로 리셋
    - 이후 `tile_m_idx_++`
    - `advance_output_tile_()`
    - ![[스크린샷 2026-04-08 오전 12.58.28.png]]
6. M/N도 다 끝났으면 종료
	1. ![[스크린샷 2026-04-08 오전 12.56.28.png]]

순회 순서 (K->N->M형식)
```
for (tile_m = 0; tile_m < tiles_m_; ++tile_m)  
  for (tile_n = 0; tile_n < tiles_n_; ++tile_n)  
    for (tile_k = 0; tile_k < tiles_k_; ++tile_k)  
      accumulate_one_native_tile(...)
```

`advance_output_tile_()`  와 `tick()`의 `EXECUTE`, `OUT_WAIT` state 참고


---

# 1. 필요한 tile 수 계산 
> init_tile_state()에서 필요한 타일 갯수를 계산
> Partial Matrix 지원 X

- `tile_k_elems = cfg::tileK * i_ratio` 
	- K는 input element type에 따라 따라 달라지기 때문에 따로 계산해야됨
- `M_total_ = desc.M != 0 ? desc.M : cfg::tileM` 
	- 0이면 그냥 기본값 사용
- `N_total_ = desc.N != 0 ? desc.N : cfg::tileN`
	- 0이면 그냥 기본값 사용
- `K_total_ = desc.K != 0 ? desc.K : tile_k_elems`

Partial matrix를 지원하지 않기 때문에 무조건 native tile의 배수여야됨
- `M_total_ % tileM == 0`
- `N_total_ % tileN == 0`
- `K_total_ % tileK == 0`

이후 행렬을 연산하기 위해 필요한 타일들의 갯수를 M/N/K 별로 저장함.
- `tiles_m_ = M_total_ / tileM`
- `tiles_n_ = N_total_ / tileN`
- `tiles_k_ = K_total_ / tileK`

큰 GEMM 속 타일들을 iterate 할 때는 다음 index를 사용하여 몇번째인지 확인함
- `tile_m_idx_`
- `tile_n_idx_`
- `tile_k_idx_`

#multi-tile #tile_computation

---

# 2. 각 tile의 base address 계산

큰 행렬을 자른 뒤, 각 tile이 원래 전체 행렬의 어디에 해당하는지는 `tile_ptrA_/B_/C_/D_()`가 계산
### A tile
`tile_ptrA_()`은 A의 row-major tile 시작 주소를 구함
- row offset = `tile_m_idx_ * tileM`
- col offset = `tile_k_idx_ * tileK`

```
row = tile_m_idx_ * cfg::tileM;  
col = tile_k_idx_ * cfg::tileK * i_ratio;  
return ptrA + (row * ldmA + col) * in_sz;
```

A는 (M축 tile, K축 tile)로 잘림

### B tile
`tile_ptrB_()`은 B의 column-major tile 시작 주소를 구함
- row offset = `tile_k_idx_ * tileK`
- col offset = `tile_n_idx_ * tileN`

```
row = tile_k_idx_ * cfg::tileK * i_ratio;  
col = tile_n_idx_ * cfg::tileN;  
return ptrB + (row + col * ldmB) * in_sz; // col-major라서 row + col * ldmB 형식
```

B는 (K축 tile, N축 tile)로 잘림

### C/D tile
`tile_ptrC_()` / `tile_ptrD_()`
- row offset = `tile_m_idx_ * tileM`
- col offset = `tile_n_idx_ * tileN`


출력 tile은 (M축 tile, N축 tile)**로 잘림

---

# 3. 자른 tile 속에서의 계산

1. load_operand()가 잘린 tile (native tile 크기)의 index를 참고
	 * tile_m_idx_, tile_n_idx_, tile_k_idx_ 에 해당하는 **A/B/C 조각**을 RAM에서 읽고,
	 * `fragA_`, `fragB_`, `fragC_`에 넣음
 * `execute_wmma()`가 그 fragment를 가지고 다시 내부 `m/n/k` step을 순회 
	 * [[DTCU Overview#Execution (WMMA)]]참고


---

# 4. K 방향 accumulation (Output Tile 계산)
> 같은 output tile에 대해 `tile_k_idx_`를 바꿔가며 누적함. (`fragC_` 사용)

 DTCU의 경우 fragC가 타일 간 accumulation을 전달함.
 * 이러한 K tile 간 partial sum 전달은 현재 외부 memory가 아니라 internal fragC_로 해결
	 * 이건 Virgo 논문을 따라 간 것

>  **4.4.1 Thread Block Tiling**
>  As the loop iterates, the Gemmini matrix unit accumulates partial sum data onto its _private accumulator memory_, which gets moved out and stored to the global memory at the end of the loop. 

### 첫 번째 K tile일 때
> 첫 K tile에서만 accumulator 초기화 또는 초기 C값을 넣음
> ([[DTCU Basic Theory#C Accumulation]] 참고) 

`load_operands()`에서 `tile_k_idx_ == 0`
- `flags & 0x1`이면 `fragC_ = 0`으로 초기화
- 아니면 `C` tile을 읽어서 `fragC_`에 넣음


### 이후 K tile부터

`tile_k_idx_ > 0`이면 `fragC_`를 다시 zero/load 하지 않음.

이전 K tile 계산 결과가 `fragC_`에 남아 있고,  그 위에 다음 K tile의 결과가 더해짐

이후 `tick()`의 `EXECUTE` state에서:
- 아직 K tile이 남았으면 `++tile_k_idx_` 후 다시 operand load/execute로 감
- 마지막 K tile이면 그제서야 output store로 감

---

## Output Matrix 구성 방식
> 각 output tile을 원래 전체 D matrix의 정확한 위치에 바로 써 넣어 output 도출
>  `store_output()`

### 방법
각 lane과 fragment index `r`에 대해 최종 `fragC_` 값을 읽고,  
현재 output tile의 base address `tile_ptrD_()`에 다음 offset을 더해서 사용
```
base_addr = tile_ptrD_()  
          + block_row * ldmD * out_sz  
          + block_col * out_sz;  
  
addr = base_addr  
     + elem_row * ldmD * out_sz  
     + elem_col * out_sz;
```

즉 현재 `(tile_m_idx_, tile_n_idx_)` tile의 결과가 전체 D matrix 안의 대응 위치에 정확히 저장됨! 하하!

tile 하나를 다 쓴 뒤 `advance_output_tile_()`가 다음 `(m,n)` output tile로 넘어가도록 함.

쉽게 설명:

전체 output matrix `D`는:
- 먼저 `(0,0)` output tile 완성 후 write
- 그다음 `(0,1)` output tile 완성 후 write
- 그다음 `(1,0)`
- 그다음 `(1,1)`

이런 식으로 tile-by-tile로 완성함.





