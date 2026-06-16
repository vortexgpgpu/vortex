# DTCU baseline (dtcu_compare) — pre-TMA

- 작성일: 2026-06-16
- 커밋: 289767e7e (현재 HEAD) + `core.cpp` DTCU_Control 수정
- 환경: apptainer (`vortex.sif`), SimX, tutorial 설정
- 빌드/실행:
  ```
  CONFIGS="-DEXT_TCU_ENABLE -DNUM_CORES=1 -DNUM_WARPS=1 -DNUM_THREADS=4 -DL2_ENABLE -DPERF_ENABLE" \
  ./ci/blackbox.sh --driver=simx --app=dtcu_compare --cores=1 --warps=1 --threads=4 --l2cache --perf=2
  ```
- 전체 로그: `claude_doc/run_dtcu_compare.log`

---

## 1. 정확도 결론: PASSED

- GEMM: M=1024, N=512, K=256, fp16 입력 / fp32 출력, C accumulator(flags=0).
- in-core TCU, DTCU 둘 다 CPU reference와 일치하고 서로도 일치 (errors_tcu=0, errors_dtcu=0, cross_errors=0).
- 즉 git log의 "result not correct"(e848e546a)는 **이후 커밋들에서 이미 해결됨.** 현재 HEAD는 정답.

## 2. 성능 baseline (TMA 개선 측정 기준점)

| 지표 | In-core TCU | DTCU | DTCU/TCU |
|---|---|---|---|
| cycles | 130,190,615 | 37,692,802 | 0.290 |
| instrs | 20,627,962 | 3,426,396 | 0.166 |
| IPC | 0.158 | 0.091 | — |
| host_ms | 208,233 | 52,234 | 0.251 |
| l2_reads | 779,526 | 426,043 | — |
| l2_writes | 655,602 | 33,034 | — |
| mem_reads | 45,123 | 45,117 | — |
| mem_writes | 655,626 | 33,058 | 0.112 |

- DTCU가 cycle ~3.4×, instruction ~6×, write 트래픽 ~20× 적음 → disaggregation 이득.
- DTCU L2 MemReq breakdown: desc=1, op=425,984, output=32,768, total=458,753.

## 3. 관찰 / 후속 확인 사항

- in-core TCU, DTCU 둘 다 `instr_tcu=0`, `instr_lsu=0` 로 출력됨. WMMA(FUType::TCU)가 도는 in-core 경로에서도 0이라, `VX_CSR_MPM_INSTR_TCU` 쿼리/카운터 plumbing 확인 필요 (이번 수정과 무관한 기존 이슈; 정확도엔 영향 없음).
- DTCU의 `DTCU_Control` 명령은 의도적으로 per-FU 카운터에 안 잡힘 (전체 instrs에만 포함). [DTCU_Control_commit_fix_report.md] 참고.

## 4. TMA 관점 메모

- 현재 DTCU는 K-tile load → compute가 serial (`OP_REQ/OP_WAIT → EXECUTE`).
- EXECUTE는 multi-cycle 모델(`exec_cycles_left_`)이라, TMA prefetch overlap이 cycle로 측정 가능.
- TMA 도입 후 핵심 측정치: `dtcu_wait_for_tma_cycles` (작아질수록 prefetch 효과).
