# Copyright © 2019-2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
import argparse, random, struct, sys, ast, os, math
import numpy as np

# Imported if --ref=cuda or --ref=cpp
_torch = None
_ext = {} # Changed to dict to cache extensions by format
_cpp_ext = None

ULP = 1  # acceptance threshold in ULPs
MAX_PRINTED_ERRORS_PER_FEATURE = 100 # Max errors to print per feature

# ----------------- format specs -----------------
FORMAT_SPECS = {
  "tf32": {"eb": 8, "sb": 10, "desc": "TF32 (e8m10)"},
  "fp16": {"eb": 5, "sb": 10, "desc": "FP16 (e5m10)"},
  "bf16": {"eb": 8, "sb": 7,  "desc": "BF16 (e8m7)"},
  "bf8":  {"eb": 5, "sb": 2,  "desc": "BF8 (e5m2)"},
  "fp8":  {"eb": 4, "sb": 3,  "desc": "FP8 (e4m3)"},
}

_FMT_ALIASES = {
  "bf8(e5m2)": "bf8",
  "fp8(e4m3)": "fp8",
}

# Configuration ported from tcu_eval.py
FMT_CONFIG = {
    "fp16": {
        "k_tile": 16, "n_tile": 16, "backing_k": 16,
        "dtype_str": "float16",
        "use_ptx": False,
        "cuda_header": "#include <cuda_fp16.h>",
        "cpp_type": "half",
        "wmma_type": "half",
        "prec_arg": "",
    },
    "bf16": {
        "k_tile": 16, "n_tile": 16, "backing_k": 16,
        "dtype_str": "bfloat16",
        "use_ptx": False,
        "cuda_header": "#include <cuda_bf16.h>",
        "cpp_type": "__nv_bfloat16",
        "wmma_type": "__nv_bfloat16",
        "prec_arg": "",
    },
    "tf32": {
        "k_tile": 8, "n_tile": 16, "backing_k": 16,
        "dtype_str": "float32",
        "use_ptx": False,
        "cuda_header": "",
        "cpp_type": "float",
        "wmma_type": "wmma::precision::tf32",
        "prec_arg": "",
    },
    "fp8": {
        "k_tile": 32, "n_tile": 8, "backing_k": 32,
        "dtype_str": "uint8", # Passed as raw bytes
        "use_ptx": True,
        "cuda_header": "#include <cuda_fp8.h>",
        "ptx_type": "e4m3",
    },
    "bf8": {
        "k_tile": 32, "n_tile": 8, "backing_k": 32,
        "dtype_str": "uint8", # Passed as raw bytes
        "use_ptx": True,
        "cuda_header": "#include <cuda_fp8.h>",
        "ptx_type": "e5m2",
    }
}

def normalize_fmt(fmt: str):
  if fmt is None:
    return None
  f = fmt.strip().lower()
  f = _FMT_ALIASES.get(f, f)
  if f not in FORMAT_SPECS:
    raise ValueError(f"Unknown --fmt '{fmt}'. Supported: {', '.join(FORMAT_SPECS.keys())}")
  return f

def fmt_to_eb_sb(fmt: str):
  f = normalize_fmt(fmt)
  spec = FORMAT_SPECS[f]
  return spec["eb"], spec["sb"]

def fmt_from_eb_sb(eb: int, sb: int):
  for k, v in FORMAT_SPECS.items():
    if v["eb"] == eb and v["sb"] == sb:
      return k
  return None

def fmt_hex_digits(eb: int, sb: int) -> int:
  total_bits = 1 + int(eb) + int(sb)
  return (total_bits + 3) // 4

# ----------------- tiny helpers -----------------
def be32_from_float(x: float) -> int:
  return struct.unpack(">I", struct.pack(">f", float(np.float32(x))))[0]

def bias(eb: int) -> int:
  return (1 << (eb - 1)) - 1

def split(x: int, eb: int, sb: int):
  s = (x >> (eb + sb)) & 1
  E = (x >> sb) & ((1 << eb) - 1)
  F = x & ((1 << sb) - 1)
  ALL1 = (1 << eb) - 1
  b = bias(eb)

  if E == ALL1 and F != 0:
    return s, 4, 0, 0  # qNaN
  if E == ALL1 and F == 0:
    return s, 3, 0, 0  # Inf
  if E == 0 and F == 0:
    return s, 0, 0, 0  # Zero
  if E == 0:
    return s, 1, 1 - b - sb, F  # subnormals
  return s, 2, E - b - sb, (1 << sb) | F  # normals

def lzc(x: int, width: int) -> int:
  return width if x == 0 else width - x.bit_length()

def csa(a: int, b: int, c: int, mask: int):
  s = (a ^ b ^ c) & mask
  k = ((a & b) | (a & c) | (b & c)) << 1
  return s, (k & mask)

def cpa(s: int, c: int, mask: int) -> int:
  return (s + c) & mask

def twos_to_int(x: int, width: int) -> int:
  return x - (1 << width) if ((x >> (width - 1)) & 1) else x

def ulp_diff(a: int, b: int) -> int:
  def to_ordered(u: int) -> int:
    return u ^ 0x80000000
  if (a & 0x7f800000) == 0x7f800000 and (a & 0x007fffff) != 0:
    return 0  # a is NaN
  if (b & 0x7f800000) == 0x7f800000 and (b & 0x007fffff) != 0:
    return 0  # b is NaN
  return abs(to_ordered(a) - to_ordered(b))

# ----------------- FEDP core -----------------
class FEDP:
  def __init__(self, lanes=4, eb=5, sb=10, frm="RNE", renorm=False, trace=False, W=25, HR=3, no_window=False):
    self.eb = eb
    self.sb = sb
    self.renorm = renorm
    self.frm = frm
    self.lanes = lanes
    self.ebC, self.sbC = 8, 23
    self.W = W
    self.HR = HR
    self.no_window = bool(no_window)
    self.trace = bool(trace)

  def s1_decode_mul(self, A, B):
    sb = self.sb
    inv = False
    nan = False
    pos = False
    neg = False
    terms = []

    for a, b in zip(A, B):
      sa, ca, ea, Ma = split(a, self.eb, sb)
      sb_, cb, eb, Mb = split(b, self.eb, sb)

      if ca == 4 or cb == 4:
        nan = True
        continue
      if (ca == 3 and cb == 0) or (cb == 3 and ca == 0):
        inv = True
        continue
      if ca == 3 or cb == 3:
        sgn = sa ^ sb_
        pos |= (sgn == 0)
        neg |= (sgn == 1)
        continue

      lzc_prod = 0
      if self.renorm:
        lz_a = lzc(Ma, sb) if ca == 1 else 0
        lz_b = lzc(Mb, sb) if cb == 1 else 0
        lzc_prod = lz_a + lz_b

      P = Ma * Mb
      e = ea + eb
      sgn = sa ^ sb_
      terms.append((sgn, P, e, lzc_prod))

    if self.trace:
      print(f"  [S1:decode_mul] terms={len(terms)} inv={inv} nan={nan} pos={pos} neg={neg}")
    return terms, inv, nan, pos, neg

  def s1_mapC(self, Cbits: int):
    sc, cc, eC, Mc = split(Cbits, self.ebC, self.sbC)
    if self.trace:
      print(f"  [S1:mapC] sc={sc} Mc=0x{Mc:x} eC={eC} cc={cc}")
    if cc == 3:
      cc = -3 if sc else 3
    return sc, Mc, eC, cc

  def s2_align(self, terms, Cp):
    normalized_terms = []
    for s, P_raw, e_raw, lead_prod in terms:
        if P_raw:
            P_norm = P_raw << lead_prod
            e_norm = e_raw - lead_prod
            normalized_terms.append((s, P_norm, e_norm))

    sc, Mc, eC, cc = Cp
    W = self.W
    sb = self.sb
    tops = []
    base_msb_offset = 2 * sb

    for s, P, e in normalized_terms:
      if P:
        top_val = e + base_msb_offset
        tops.append(top_val)

    if cc in (0, 1, 2):
      if Mc:
        tops.append(eC + self.sbC)

    if not tops:
      if self.trace:
        print("  [S2:align] no terms; nothing to align")
      return [], 0, 0, cc

    L = max(tops) - (W-1)
    aligned_terms = []
    sticky = 0

    def align_and_add(val: int, e: int):
      nonlocal sticky
      k = e - L
      if k >= 0:
        aligned_terms.append(val << k)
      else:
        sh = -k
        mag = -val if val < 0 else val
        part = mag >> sh
        T = -part if val < 0 else part
        aligned_terms.append(T)
        if (mag & ((1 << sh) - 1)) != 0:
          sticky |= 1

    for s, P, e in normalized_terms:
      if P:
        align_and_add(-P if s else P, e)

    if Mc and cc in (0, 1, 2):
      align_and_add(-Mc if sc else Mc, eC)

    if self.trace:
      print(f"  [S2:align] aligned_terms={len(aligned_terms)} sticky={sticky} L={L}")
    return aligned_terms, sticky, L, cc

  def s3_accumulate(self, aligned_terms):
    if not aligned_terms:
      if self.trace:
        print("  [S3:accumulate] Vw=0 (no terms)")
      return 0
    if self.no_window:
      V = 0
      for T in aligned_terms:
        V += T
      return V

    WW = self.W + self.HR
    mask = (1 << WW) - 1
    s_acc = 0
    c_acc = 0
    for T in aligned_terms:
      s_acc, c_acc = csa(s_acc, c_acc, T & mask, mask)
    Vw_unsigned = cpa(s_acc, c_acc, mask)
    Vw = twos_to_int(Vw_unsigned, WW)
    if self.trace:
      print(f"  [S3:accumulate] Vw={Vw} (0x{Vw_unsigned:x})")
    return Vw

  def s4_normalize(self, V: int, st: int, L: int):
    WW = self.W + self.HR
    if V == 0:
      return 0, 0, 0, st, -10**9

    s = 1 if V < 0 else 0
    X = (-V if V < 0 else V)
    if not self.no_window:
      X &= (1 << WW) - 1

    i = X.bit_length() - 1
    e = L + i
    sh = (i + 1) - 24

    if sh >= 0:
      kept = (X >> sh) & ((1 << 24) - 1)
      rem = X & ((1 << sh) - 1)
      g = (rem >> (sh - 1)) & 1 if sh > 0 else 0
      low = rem & ((1 << (sh - 1)) - 1) if sh > 1 else 0
      st_out = 1 if (low != 0 or st) else 0
    else:
      kept = (X << (-sh)) & ((1 << 24) - 1)
      g = 0
      st_out = st

    if self.trace:
      print(f"  [S4:normalize] s={s} kept=0x{kept:06x} g={g} st={st_out} e={e}")
    return s, kept, g, st_out, e

  def s5_rounding(self, s, kept, g, st, e, cc, inv, nan, pos, neg):
    if nan or inv or (pos and neg):
      result = 0x7fc00000
    elif cc == 4:
      result = 0x7fc00000
    elif cc in (3, -3):
      cneg = 1 if cc < 0 else 0
      if pos or neg:
        if (pos and neg) or (pos and cneg) or (neg and not cneg):
          result = 0x7fc00000
        else:
          result = 0xff800000 if cneg else 0x7f800000
      else:
        result = 0xff800000 if cneg else 0x7f800000
    elif pos and not neg:
      result = 0x7f800000
    elif neg and not pos:
      result = 0xff800000
    elif kept == 0 and st == 0:
      result = 0x00000000
    else:
      result = self._f32_round_pack(s, kept, g, st, e, self.frm)
    if self.trace:
      print(f"  [S5:rounding] final=0x{result:08x}")
    return result

  @staticmethod
  def _f32_round_pack(s, kept, g, st, e_unb, frm):
    discarded = (g == 1 or st == 1)
    if frm == "RNE":
      round_up = (g == 1 and (st == 1 or (kept & 1) == 1))
    elif frm == "RTZ":
      round_up = False
    elif frm == "RDN":
      round_up = (s == 1 and discarded)
    elif frm == "RUP":
      round_up = (s == 0 and discarded)
    elif frm == "RMM":
      round_up = (g == 1)
    else:
      raise ValueError(f"Unknown rounding mode: {frm}")

    if round_up:
      kept_rounded = kept + 1
    else:
      kept_rounded = kept

    if kept_rounded & (1 << 24):
      kept_rounded >>= 1
      e_unb += 1

    be = e_unb + 127
    if be >= 255:
      if frm == "RTZ":
        return (s << 31) | 0x7f7fffff
      if frm == "RDN":
        return (s << 31) | 0x7f7fffff if s == 0 else 0xff800000
      if frm == "RUP":
        return (s << 31) | 0x7f800000 if s == 0 else 0xff7fffff
      return (s << 31) | 0x7f800000

    if be <= 0:
      k = 1 - be
      if frm == "RTZ":
        m = (kept_rounded >> k) if k < 25 else 0
        return (s << 31) | (m & 0x7FFFFF)
      if k >= 25:
        if (frm == "RUP" and s == 0 and discarded):
            return (s << 31) | 1
        if (frm == "RDN" and s == 1 and discarded):
            return (s << 31) | 0x80000001
        return (s << 31)

      lsb = (kept_rounded >> k) & 1
      new_g = (kept_rounded >> (k - 1)) & 1
      st_mask = (1 << (k - 1)) - 1
      new_st = 1 if ((kept_rounded & st_mask) != 0 or g != 0 or st != 0) else 0
      m_sub = kept_rounded >> k

      if frm == "RNE":
          round_up_sub = (new_g == 1 and (new_st == 1 or lsb == 1))
      elif frm == "RDN":
          round_up_sub = (s == 1 and (new_g == 1 or new_st == 1))
      elif frm == "RUP":
          round_up_sub = (s == 0 and (new_g == 1 or new_st == 1))
      elif frm == "RMM":
          round_up_sub = (new_g == 1)
      else: round_up_sub = False

      m_final = m_sub + 1 if round_up_sub else m_sub
      if m_final == (1 << 23):
          return (s << 31) | (1 << 23)
      return (s << 31) | (m_final & 0x7FFFFF)

    m = kept_rounded & 0x7FFFFF
    return (s << 31) | ((be & 0xff) << 23) | m

  def dotp(self, A, B, Cbits):
    terms, inv, nan, pos, neg = self.s1_decode_mul(A, B)
    Cp = self.s1_mapC(Cbits)
    aligned, sticky1, L, cc = self.s2_align(terms, Cp)
    V = self.s3_accumulate(aligned)
    s, kept, g, sticky2, e = self.s4_normalize(V, sticky1, L)
    return self.s5_rounding(s, kept, g, sticky2, e, cc, inv, nan, pos, neg)

# ----------------- references -----------------
def _to_float_np(x: int, eb: int, sb: int) -> np.longdouble:
  s = (x >> (eb + sb)) & 1
  E = (x >> sb) & ((1 << eb) - 1)
  F = x & ((1 << sb) - 1)
  b = bias(eb)

  if E == 0 and F == 0:
    return np.longdouble(-0.0) if s else np.longdouble(0.0)
  if E == (1 << eb) - 1:
    if F != 0:
      return np.longdouble(np.nan)
    return np.longdouble(-np.inf) if s else np.longdouble(np.inf)
  if E == 0:
    m = np.longdouble(F) / np.longdouble(1 << sb)
    e = 1 - b
  else:
    m = np.longdouble(1.0) + np.longdouble(F) / np.longdouble(1 << sb)
    e = E - b
  v = np.ldexp(m, int(e))
  return -v if s else v

def round_longdouble_to_f32_bits(x: np.longdouble, frm: str) -> int:
  if np.isnan(x):
    return 0x7fc00000
  if np.isposinf(x):
    return 0x7f800000
  if np.isneginf(x):
    return 0xff800000
  if x == 0:
    s = 1 if np.signbit(x) else 0
    return s << 31
  s = 1 if x < 0 else 0
  ax = -x if s else x
  ax = np.longdouble(ax)
  f, e2 = np.frexp(ax)
  e_unb = int(e2) - 1
  t = ax / (np.longdouble(2.0) ** np.longdouble(e_unb - 23))
  mant_floor = int(t)
  frac = t - np.longdouble(mant_floor)
  if frac == 0:
    g, st = 0, 0
  elif frac < 0.5:
    g, st = 0, 1
  elif frac == 0.5:
    g, st = 1, 0
  else:
    g, st = 1, 1
  return FEDP._f32_round_pack(s, mant_floor, g, st, e_unb, frm)

def ref_numpy(A, B, Cbits, eb, sb, frm) -> int:
  a_ld = np.array([_to_float_np(x, eb, sb) for x in A], dtype=np.longdouble)
  b_ld = np.array([_to_float_np(x, eb, sb) for x in B], dtype=np.longdouble)
  c_ld = np.longdouble(_to_float_np(Cbits, 8, 23))
  with np.errstate(invalid='ignore', over='ignore'):
    s_ld = (a_ld * b_ld).sum(dtype=np.longdouble) + c_ld
  return round_longdouble_to_f32_bits(s_ld, frm)

def _ensure_cuda_ext(fmt: str, arch_flag: str):
  """
  Build/load a CUDA extension dynamically based on the requested format.
  Uses the robust template logic from tcu_eval.py.
  """
  global _torch, _ext

  # Create a unique key for the extension
  ext_key = f"{fmt}_{arch_flag}"
  if ext_key in _ext:
    return _torch, _ext[ext_key]

  try:
    import torch as _t
    from torch.utils.cpp_extension import load_inline
  except Exception as e:
    raise RuntimeError(f"--ref=cuda requires PyTorch with CUDA: {e}")

  if not _t.cuda.is_available():
    raise RuntimeError("--ref=cuda requires CUDA-capable PyTorch")

  if fmt not in FMT_CONFIG:
      raise ValueError(f"Unsupported cuda format: {fmt}")

  cfg = FMT_CONFIG[fmt]

  # C++ Source (Binding)
  cpp_src = r"""
    #include <torch/extension.h>
    void wmma_tile_gemm_batched_launcher(torch::Tensor A,
                                         torch::Tensor B_colmajor,
                                         torch::Tensor C,
                                         torch::Tensor D,
                                         int Bcount);
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
      m.def("wmma_tile_gemm_batched_launcher",
            &wmma_tile_gemm_batched_launcher,
            "Batched WMMA/PTX Tile GEMM");
    }
    """

  # CUDA Source
  if not cfg["use_ptx"]:
    # Standard WMMA C++ Path
    cuda_src_template = r"""
    #include <torch/extension.h>
    #include <ATen/cuda/CUDAContext.h>
    #include <mma.h>
    {cuda_header}

    using namespace nvcuda;
    using elem_t = {cpp_type};

    __global__ void wmma_tile_kernel(const elem_t* __restrict__ A,
                                     const elem_t* __restrict__ B,
                                     const float* __restrict__ C,
                                     float* __restrict__ D,
                                     int Bcount) {{
      int b = blockIdx.x;
      if (b >= Bcount) return;

      const int K_BACKING = {backing_k};
      const int N_TILE    = {n_tile};

      // Stride setup matching tcu_eval logic:
      // A: row-major 16 x K_BACKING
      // B: col-major K_BACKING x 16 (but we pass pre-transposed memory with stride K)
      const int STRIDE_AB = 16 * K_BACKING;
      const int LD_AB     = K_BACKING;
      const int STRIDE_CD = 16 * 16;
      const int LD_CD     = 16;

      const elem_t* Ab = A + b * STRIDE_AB;
      const elem_t* Bb = B + b * STRIDE_AB;
      const float* Cb = C + b * STRIDE_CD;
      float* Db = D + b * STRIDE_CD;

      wmma::fragment<wmma::matrix_a, 16, N_TILE, {k_tile}, {wmma_type}, wmma::row_major {prec_arg}> a;
      wmma::fragment<wmma::matrix_b, 16, N_TILE, {k_tile}, {wmma_type}, wmma::col_major {prec_arg}> bfrag;
      wmma::fragment<wmma::accumulator, 16, N_TILE, {k_tile}, float> c, d;

      wmma::load_matrix_sync(a, Ab, LD_AB);
      wmma::load_matrix_sync(bfrag, Bb, LD_AB);
      wmma::load_matrix_sync(c, Cb, LD_CD, wmma::mem_row_major);

      wmma::mma_sync(d, a, bfrag, c);

      wmma::store_matrix_sync(Db, d, LD_CD, wmma::mem_row_major);
    }}

    void wmma_tile_gemm_batched_launcher(torch::Tensor A,
                                         torch::Tensor B_colmajor,
                                         torch::Tensor C,
                                         torch::Tensor D,
                                         int Bcount) {{
      dim3 grid(Bcount);
      dim3 block(32);
      wmma_tile_kernel<<<grid, block>>>(
          reinterpret_cast<const elem_t*>(A.data_ptr()),
          reinterpret_cast<const elem_t*>(B_colmajor.data_ptr()),
          reinterpret_cast<const float*>(C.data_ptr()),
          reinterpret_cast<float*>(D.data_ptr()),
          Bcount);
    }}
    """
    cuda_src = cuda_src_template.format(
        cuda_header=cfg["cuda_header"],
        cpp_type=cfg["cpp_type"],
        k_tile=cfg["k_tile"],
        n_tile=cfg["n_tile"],
        backing_k=cfg["backing_k"],
        wmma_type=cfg["wmma_type"],
        prec_arg=cfg["prec_arg"]
    )

  else:
    # Manual PTX FP8 Path
    cuda_src_template = r"""
    #include <torch/extension.h>
    #include <cuda_fp8.h>

    __global__ void ptx_mma_kernel(const void* __restrict__ A,
                                   const void* __restrict__ B,
                                   const float* __restrict__ C,
                                   float* __restrict__ D,
                                   int Bcount) {{
      int b = blockIdx.x;
      if (b >= Bcount) return;

      int tid = threadIdx.x;
      const int STRIDE_A = 16 * 32;
      const int STRIDE_B = 16 * 32;

      const char* Ab = (const char*)A + b * STRIDE_A;
      const char* Bb = (const char*)B + b * STRIDE_B;

      __shared__ __align__(16) char Ash[16 * 32];
      __shared__ __align__(16) char Bsh[32 * 16];

      int4* Ash_ptr = (int4*)Ash;
      const int4* Ab_ptr = (const int4*)Ab;
      Ash_ptr[tid] = Ab_ptr[tid];

      const int2* Bb_ptr = (const int2*)Bb;
      int2* Bsh_ptr = (int2*)Bsh;
      int2 val = Bb_ptr[tid * 2];
      Bsh_ptr[tid * 2] = val;

      __syncthreads();

      uint32_t ra[4];
      uint32_t rb[2];
      uint32_t smem_a_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&Ash[0]));
      uint32_t addr_a = smem_a_ptr + (tid % 16) * 32;

      asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%0, %1, %2, %3}}, [%4];"
        : "=r"(ra[0]), "=r"(ra[1]), "=r"(ra[2]), "=r"(ra[3])
        : "r"(addr_a)
      );

      uint32_t smem_b_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&Bsh[0]));
      uint32_t addr_b = smem_b_ptr + (tid % 32) * 16;

      asm volatile (
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{%0, %1}}, [%2];"
        : "=r"(rb[0]), "=r"(rb[1])
        : "r"(addr_b)
      );

      float d[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};

      asm volatile (
        "mma.sync.aligned.m16n8k32.row.col.f32.{ptx_type}.{ptx_type}.f32 "
        "{{%0, %1, %2, %3}}, {{%4, %5, %6, %7}}, {{%8, %9}}, {{%0, %1, %2, %3}};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]),
          "r"(rb[0]), "r"(rb[1])
      );

      float* Db = D + b * (16*16);
      int groupID = tid >> 2;
      int threadID = tid & 3;
      int row0 = groupID;
      int col0 = threadID * 2;
      int row2 = groupID + 8;
      const float* Cb = C + b * (16*16);

      auto store = [&](int r, int c, float val) {{
          int idx = r*16 + c;
          Db[idx] = val + Cb[idx];
      }};
      store(row0, col0,     d[0]);
      store(row0, col0 + 1, d[1]);
      store(row2, col0,     d[2]);
      store(row2, col0 + 1, d[3]);
    }}

    void wmma_tile_gemm_batched_launcher(torch::Tensor A,
                                         torch::Tensor B_colmajor,
                                         torch::Tensor C,
                                         torch::Tensor D,
                                         int Bcount) {{
      dim3 grid(Bcount);
      dim3 block(32);
      ptx_mma_kernel<<<grid, block>>>(
          reinterpret_cast<const void*>(A.data_ptr()),
          reinterpret_cast<const void*>(B_colmajor.data_ptr()),
          reinterpret_cast<const float*>(C.data_ptr()),
          reinterpret_cast<float*>(D.data_ptr()),
          Bcount);
    }}
    """
    cuda_src = cuda_src_template.format(ptx_type=cfg["ptx_type"])

  # Ensure compilation for correct arch
  os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch_flag.replace("sm_", "").replace(".", ""))

  ext = load_inline(
    name=f"fedp_wmma_{ext_key}",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=["-O3", "-std=c++17", f"-arch={arch_flag}", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__"],
    verbose=False,
  )

  _torch = _t
  _ext[ext_key] = ext
  return _torch, ext

def _u32_to_f32(val_u32: int) -> np.float32:
  return np.array([val_u32], dtype=np.uint32).view(np.float32)[0]

def ref_cuda(A, B, Cbits, eb, sb, arch_flag="sm_89") -> int:
  """
  Unified CUDA reference using tcu_eval backend logic.
  Map dot-product A*B+C to GEMM row 0, col 0.
  """
  fmt = fmt_from_eb_sb(eb, sb)
  torch, ext = _ensure_cuda_ext(fmt, arch_flag)
  device = torch.device("cuda")
  cfg = FMT_CONFIG[fmt]

  K = len(A)
  if K > cfg["k_tile"]:
    raise ValueError(f"--ref=cuda({fmt}) expects n <= {cfg['k_tile']}, got {K}")

  # 1. Prepare Data
  # A and B inputs are Python integers (raw bits). We must pack them into the
  # correct numpy types expected by the tensor core backend.

  if fmt == "fp16":
    a_np = np.array(A, dtype=np.uint16).view(np.float16)
    b_np = np.array(B, dtype=np.uint16).view(np.float16)
    torch_dtype = torch.float16
  elif fmt == "bf16":
    # No native np.bfloat16. Pass as int16, view as bfloat16 in Torch
    a_np = np.array(A, dtype=np.int16)
    b_np = np.array(B, dtype=np.int16)
    torch_dtype = torch.bfloat16
  elif fmt == "tf32":
    # TF32 in memory is float32. We expand the packed bits to float32.
    a_np = np.array([_to_float_np(x, 8, 10) for x in A], dtype=np.float32)
    b_np = np.array([_to_float_np(x, 8, 10) for x in B], dtype=np.float32)
    torch_dtype = torch.float32
  elif fmt in ["fp8", "bf8"]:
    # Pass as raw bytes (uint8)
    a_np = np.array(A, dtype=np.uint8)
    b_np = np.array(B, dtype=np.uint8)
    # torch.float8 types exist in newer torch, but our backend casts void* -> int4/int2
    # so we just pass uint8 tensors and let the kernel re-interpret.
    torch_dtype = torch.uint8
  else:
    raise ValueError(f"Unknown format for ref_cuda: {fmt}")

  # 2. Layout for GEMM
  # GEMM: C_mxn = A_mxk * B_kxn + C_mxn
  # We want a simple dot product. We use 1xK * Kx1 = 1x1.
  # However, the kernels operate on tiles (16x16, etc).
  # We place Vector A into Row 0 of Matrix A Tile.
  # We place Vector B into Column 0 of Matrix B Tile.
  # Result is at (0,0).

  backing_k = cfg["backing_k"]
  n_tile = cfg["n_tile"] # typically 16 or 8

  # A_h: 1 (batch) x 16 (rows) x backing_k (cols)
  # Row-major.
  A_h = np.zeros((1, 16, backing_k), dtype=a_np.dtype)
  A_h[0, 0, :K] = a_np

  # B_h: 1 (batch) x 16 (rows) x backing_k (cols)
  # We need to construct B such that when treated as COL-MAJOR by the kernel,
  # the first column (which is contiguous in row-major memory?)
  # Wait, tcu_eval logic:
  #   B_row[:, :K, 0] = b_terms  (B_row is 16xK, b_terms is Kx1)
  #   B_h = transpose(B_row) -> 1 x K x 16
  #   The kernel receives B_colmajor pointer.
  #   wmma::load_matrix_sync(..., layout=col_major) expects stride to be leading dimension (rows).
  #
  # Let's stick to the tcu_eval construction exactly:
  # B input to launcher is "B_colmajor".
  # In tcu_eval: B_row = zeros(16, backing_k); B_row[:K, 0] = b_vec
  #              B_h   = transpose(B_row) -> (backing_k, 16)

  B_row = np.zeros((1, 16, backing_k), dtype=b_np.dtype)
  B_row[0, :K, 0] = b_np
  # Transpose last two dims: (1, 16, K) -> (1, K, 16)
  B_h = np.transpose(B_row, (0, 2, 1)).copy()

  # C: 1 x 16 x 16 (float32)
  c_val = _u32_to_f32(Cbits)
  C_h = np.zeros((1, 16, 16), dtype=np.float32)
  C_h[0, 0, 0] = c_val

  # 3. Transfer to GPU
  if fmt == "bf16":
    # Special view handling for BF16 since numpy lacks it
    A_d = torch.from_numpy(A_h).to(device=device).view(torch.bfloat16)
    B_d = torch.from_numpy(B_h).to(device=device).view(torch.bfloat16)
  else:
    A_d = torch.from_numpy(A_h).to(device=device, dtype=torch_dtype)
    B_d = torch.from_numpy(B_h).to(device=device, dtype=torch_dtype)

  C_d = torch.from_numpy(C_h).to(device=device, dtype=torch.float32)
  D_d = torch.empty_like(C_d)

  # 4. Launch
  ext.wmma_tile_gemm_batched_launcher(A_d, B_d, C_d, D_d, 1)
  torch.cuda.synchronize()

  # 5. Readback
  D00 = float(D_d[0, 0, 0].item())
  return be32_from_float(D00)


# ----------------- C++ reference -----------------
def _ensure_cpp_ext(source_path):
  global _torch, _cpp_ext
  if _cpp_ext is not None:
    return _torch, _cpp_ext

  try:
    import torch as _t
    from torch.utils.cpp_extension import load_inline
  except Exception as e:
    raise RuntimeError(f"--ref=cpp requires PyTorch: {e}")

  abs_path = os.path.abspath(source_path)
  if not os.path.exists(abs_path):
    raise FileNotFoundError(f"C++ source not found at: {abs_path}")

  cpp_includes = f'#include "{abs_path}"'
  cpp_src = """
  #include <torch/extension.h>
  #include <iostream>
  #include <cstring>

  #undef LOG
  // HEADER_PATH_PLACEHOLDER

  uint32_t fedp_run_wrapper(
      torch::Tensor A, torch::Tensor B, uint32_t Cbits,
      int n, int eb, int sb, int lanes, std::string frm_str,
      int W, bool renorm, bool no_window) {

      int frm = 0;
      if (frm_str == "RNE") frm = 0;
      else if (frm_str == "RTZ") frm = 1;
      else if (frm_str == "RDN") frm = 2;
      else if (frm_str == "RUP") frm = 3;
      else if (frm_str == "RMM") frm = 4;
      else frm = 0;

      FEDP fedp(eb, sb, lanes, frm, W, renorm, no_window);

      float c_float;
      uint32_t c_u32 = Cbits;
      std::memcpy(&c_float, &c_u32, sizeof(c_float));

      auto a_ptr = A.data_ptr<int32_t>();
      auto b_ptr = B.data_ptr<int32_t>();

      float res_float = fedp(
        reinterpret_cast<const uint32_t*>(a_ptr),
        reinterpret_cast<const uint32_t*>(b_ptr),
        c_float,
        (uint32_t)n
      );

      uint32_t res_u32;
      std::memcpy(&res_u32, &res_float, sizeof(res_u32));
      return res_u32;
  }

  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fedp_run", &fedp_run_wrapper, "FEDP Runner Wrapper");
  }
  """

  cpp_wrapper_src = cpp_src.replace("// HEADER_PATH_PLACEHOLDER", cpp_includes)

  _cpp_ext = load_inline(
    name="fedp_cpp_class_wrapper",
    cpp_sources=cpp_wrapper_src,
    extra_cflags=["-O3", "-std=c++17"],
    verbose=False,
  )
  _torch = _t
  return _torch, _cpp_ext

def ref_cpp(A, B, Cbits, eb, sb, W, HR, renorm, frm, no_window, source_path):
  torch, ext = _ensure_cpp_ext(source_path)
  tA = torch.tensor(A, dtype=torch.int32)
  tB = torch.tensor(B, dtype=torch.int32)
  return ext.fedp_run(tA, tB, Cbits, len(A), eb, sb, len(A), frm, W, renorm, no_window)


# ----------------- test-case generators -----------------
def _pack_fp(s, e, f, eb, sb):
  return (s << (eb + sb)) | (e << sb) | f

def _stable_feature_tag(name: str) -> int:
  v = 0
  for c in name.encode("ascii"):
    v = (v * 131 + c) & 0xffffffff
  return v or 1

def generate_fp_value(feature, eb, sb, test_id):
  all_exp = (1 << eb) - 1
  max_frac = (1 << sb) - 1
  tag = _stable_feature_tag(feature)
  s = (test_id ^ tag) & 1

  if feature == "zeros":
    return _pack_fp(test_id & 1, 0, 0, eb, sb)
  elif feature == "infinities":
    return _pack_fp(test_id & 1, all_exp, 0, eb, sb)
  elif feature == "nans":
    if sb == 0:
      return _pack_fp(s, all_exp, 0, eb, sb)
    qbit = 1 << (sb - 1)
    if sb == 1:
      return _pack_fp(s, all_exp, qbit, eb, sb)
    sub_payload_mask = qbit - 1
    if test_id % 2 == 0:
      sub_payload = (test_id // 2) % qbit
      f = qbit | sub_payload
    else:
      sub_payload = ((test_id // 2) % sub_payload_mask) + 1
      f = sub_payload
    return _pack_fp(s, all_exp, f, eb, sb)
  elif feature == "subnormals":
    if sb == 0:
      return _pack_fp(s, 0, 0, eb, sb)
    case = test_id % 3
    if case == 0: f = 1
    elif case == 1: f = max_frac
    else: f = random.randint(1, max_frac)
    return _pack_fp(s, 0, f, eb, sb)
  elif feature == "normals":
    if eb < 2:
      return _pack_fp(s, 0, 0, eb, sb)
    case = test_id % 5
    if case == 0: e, f = 1, random.randint(0, max_frac)
    elif case == 1: e, f = all_exp - 1, max_frac
    elif case == 2: e, f = bias(eb), random.randint(0, 15) & max_frac
    else: e, f = random.randint(1, all_exp - 1), random.randint(0, max_frac)
    return _pack_fp(s, e, f, eb, sb)
  return 0

def _mk_cancel_cases(n, eb, sb, count):
  cases = []
  for _ in range(count):
    E = random.randint(1, (1 << eb) - 2)
    F = random.getrandbits(sb)
    a = (random.getrandbits(1) << (eb + sb)) | (E << sb) | F
    b = (random.getrandbits(1) << (eb + sb)) | \
        (random.randint(1, (1 << eb) - 2) << sb) | random.getrandbits(sb)
    a2 = a ^ (1 << (eb + sb))
    A = []
    B = []
    for _ in range(n // 2):
      A += [a, a2]
      B += [b, b]
    if n % 2 == 1:
      A.append(generate_fp_value("normals", eb, sb, 0))
      B.append(generate_fp_value("normals", eb, sb, 1))
    C = be32_from_float(np.random.choice([0.0, 2.0**-149, -2.0**-149]))
    cases.append((A, B, C))
  return cases

CUSTOM_CASES = [
  [["0xfbff","0xfbff"],["0x83ff","0x7bff"],"0x4f7f8000"],
  [["0xe150","0xf4d7","0x4bcc","0xf3c1"],["0x83ff","0xda97","0x83ff","0x7ac6"],"0x4e51ad09"],
]

def _mk_custom(n, eb, sb):
  if not (eb == 5 and sb == 10):
    return []
  def _pi(x):
    return int(x, 16) if isinstance(x, str) else int(x)
  out = []
  for A, B, C in CUSTOM_CASES:
    if len(A) == n and len(B) == n:
      out.append(([ _pi(v) for v in A ],
                  [ _pi(v) for v in B ],
                  _pi(C)))
  return out

def _print_case(tag, idx, A, B, C, hex_digits=4):
  a_str = ",".join(f"0x{x:0{hex_digits}x}" for x in A)
  b_str = ",".join(f"0x{x:0{hex_digits}x}" for x in B)
  c_str = f"0x{C:08x}"
  print(f'[{tag} #{idx}] inputs="[{a_str}];[{b_str}];{c_str}"')

# ----------------- harness -----------------
def test(n, eb, sb, frm, renorm, iters, seed, debug, trace,
         test_id_filter, ref_mode, arch_flag, max_errors,
         W, HR, no_window, cpp_source):

  fedp = FEDP(eb=eb, sb=sb, frm=frm, renorm=renorm, lanes=n, trace=trace, W=W, HR=HR, no_window=no_window)
  hex_digits = fmt_hex_digits(eb, sb)

  if ref_mode == "cuda":
    # Ensure ext is loaded early to fail fast
    fmt = fmt_from_eb_sb(eb, sb)
    if fmt is None:
         print(f"Error: CUDA ref does not support custom eb={eb}/sb={sb}. Use --fmt.", file=sys.stderr)
         sys.exit(1)
    _ensure_cuda_ext(fmt, arch_flag=arch_flag)
    ref_fn = lambda A, B, C, EB, SB: ref_cuda(A, B, C, EB, SB, arch_flag=arch_flag)
  elif ref_mode == "cpp":
    _ensure_cpp_ext(cpp_source)
    ref_fn = lambda A, B, C, EB, SB: ref_cpp(A, B, C, EB, SB, W, HR, renorm, frm, no_window, cpp_source)
  else:
    ref_fn = lambda A, B, C, EB, SB: ref_numpy(A, B, C, EB, SB, frm=frm)

  features = ["normals", "subnormals", "zeros", "infinities", "nans", "cancellation", "custom"]
  tests_per_feature = max(1, (iters + len(features) - 1) // len(features))

  cancel_cases = _mk_cancel_cases(n, eb, sb, tests_per_feature)
  custom_cases = _mk_custom(n, eb, sb)

  ok_by_feature = {f: [0, 0] for f in features}
  errors_by_feature = {f: 0 for f in features}
  max_ulp_by_feature = {f: 0 for f in features}
  errors = 0

  for test_id in range(iters):
    current_seed = seed + test_id
    random.seed(current_seed)
    np.random.seed(current_seed)

    if test_id_filter is not None and test_id != test_id_filter:
      continue

    feature_idx = test_id // tests_per_feature
    if feature_idx >= len(features):
      continue
    feature = features[feature_idx]
    sub_id = test_id % tests_per_feature

    if feature == "cancellation":
      if sub_id >= len(cancel_cases): continue
      A, B, C = cancel_cases[sub_id]
    elif feature == "custom":
      if sub_id >= len(custom_cases): continue
      A, B, C = custom_cases[sub_id]
    else:
      A = [0] * n
      B = [0] * n
      op_focus = test_id % 3
      for i in range(n):
        a_feat = feature if (op_focus == 0 and i % 2 == 0) else "normals"
        b_feat = feature if (op_focus == 1 and i % 2 == 0) else "normals"
        A[i] = generate_fp_value(a_feat, eb, sb, sub_id + i)
        B[i] = generate_fp_value(b_feat, eb, sb, sub_id + n + i)
      c_feat = feature if op_focus == 2 else "normals"
      C = generate_fp_value(c_feat, 8, 23, sub_id)

    if debug or trace:
      _print_case(f"{feature} TID {test_id}", sub_id, A, B, C, hex_digits=hex_digits)

    got = fedp.dotp(A, B, C)
    try:
      ref = ref_fn(A, B, C, eb, sb)
    except (RuntimeError, ValueError) as e:
      print(f"SKIPPING {feature} TID {test_id}: ref_fn failed: {e}")
      continue

    current_ulp = ulp_diff(got, ref)
    prev_max_ulp = max_ulp_by_feature[feature]
    is_new_max = current_ulp > prev_max_ulp
    if is_new_max:
      max_ulp_by_feature[feature] = current_ulp

    passed = (current_ulp <= ULP)
    ok, tot = ok_by_feature[feature]
    ok_by_feature[feature][1] = tot + 1
    if passed:
      ok_by_feature[feature][0] = ok + 1
    else:
      errors += 1
      errors_by_feature[feature] += 1
      print_this_error = (
        errors_by_feature[feature] <= MAX_PRINTED_ERRORS_PER_FEATURE
        or is_new_max
      )
      if print_this_error:
        print(f"✘ FAIL {feature} TID {test_id}: got=0x{got:08x} ref=0x{ref:08x} ulp={current_ulp}")
        _print_case(feature, sub_id, A, B, C, hex_digits=hex_digits)
      elif errors_by_feature[feature] == MAX_PRINTED_ERRORS_PER_FEATURE + 1:
        print(f"ℹ Stopping error print for {feature} (limit={MAX_PRINTED_ERRORS_PER_FEATURE} reached).")

      if max_errors > 0 and errors >= max_errors:
        print(f"\nStopping early after {errors} errors (limit={max_errors}).", file=sys.stderr)
        break
    if max_errors > 0 and errors >= max_errors:
      break

  print("-" * 40)
  total_ok = 0
  total_tot = 0
  for f, (ok, tot) in ok_by_feature.items():
    if tot == 0: continue
    mark = "✔" if ok == tot else "✘"
    pc = math.floor(1000.0 * ok / tot) / 10
    summary_str = f"{mark} {f}: {ok}/{tot} ({pc:.1f}%)"
    if ok < tot:
      summary_str += f" MAX_ULP={max_ulp_by_feature[f]}"
    print(summary_str)
    total_ok += ok
    total_tot += tot
  print("-" * 40)
  if total_tot > 0:
    pc = math.floor(1000.0 * total_ok / total_tot) / 10
    print(f"OVERALL: {total_ok}/{total_tot} ({pc:.1f}%)")
  if errors > 0:
    sys.exit(1)

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--n", type=int, default=4,
                  help="Number of dot-product lanes")
  ap.add_argument("--eb", type=int, default=5)
  ap.add_argument("--sb", type=int, default=10)
  ap.add_argument("--fmt", type=str, default=None,
                  choices=["tf32","fp16","bf16","bf8","fp8"],
                  help="Select a native input format (overrides --eb/--sb)")
  ap.add_argument("--frm", type=str, choices=["RNE", "RTZ", "RDN", "RUP", "RMM"], default="RNE",
                  help="Floating-point rounding mode")
  ap.add_argument("--renorm", action="store_true", help="renormalize product")
  ap.add_argument("--iters", type=int, default=100000)
  ap.add_argument("--seed", type=int, default=1)
  ap.add_argument("--debug", action="store_true", help="print every test scenario")
  ap.add_argument("--trace", action="store_true", help="print per-stage outputs")
  ap.add_argument("--test", type=int, default=None, help="Run a single test by its global TID")
  ap.add_argument("--run", type=str, default=None,
                  help="Directly run a single case. Format: '[A];[B];C'")
  ap.add_argument("--ref", type=str, choices=["numpy", "cuda", "cpp"], default="numpy",
                  help="Reference: 'numpy' (longdouble), 'cuda' (WMMA/PTX), 'cpp' (external)")
  ap.add_argument("--arch", type=str, default="sm_89",
                  help="CUDA arch for --ref=cuda (e.g., sm_80, sm_89, sm_90)")
  ap.add_argument("--cpp-source", type=str, default="fedp.h",
                  help="Path to external C++ source containing FEDP class")
  ap.add_argument("--max-errors", type=int, default=0,
                  help="Stop after N errors (0 = do not stop)")
  ap.add_argument("--W", type=int, default=25,
                  help="Accumulator window width")
  ap.add_argument("--HR", type=int, default=4,
                  help="Accumulator Headroom bits")
  ap.add_argument("--no-window", action="store_true",
                  help="Disable window clipping in FEDP accumulator")
  args = ap.parse_args()

  if args.fmt is not None:
    args.fmt = normalize_fmt(args.fmt)
    args.eb, args.sb = fmt_to_eb_sb(args.fmt)

  if args.run:
    try:
      A_str, B_str, C_str = args.run.split(';')
      A = ast.literal_eval(A_str)
      B = ast.literal_eval(B_str)
      C = ast.literal_eval(C_str)
      if len(A) != len(B):
        print(f"ERROR: --run A and B must have same length", file=sys.stderr)
        sys.exit(1)
      args.n = len(A)
      fedp = FEDP(lanes=args.n, eb=args.eb, sb=args.sb, frm=args.frm, renorm=args.renorm,
                  trace=args.trace, W=args.W, HR=args.HR, no_window=args.no_window)
      hex_digits = fmt_hex_digits(args.eb, args.sb)
      if not args.trace:
        _print_case("direct", 0, A, B, C, hex_digits=hex_digits)

      if args.ref == "cuda":
        fmt = fmt_from_eb_sb(args.eb, args.sb)
        if not fmt:
            print("Error: CUDA ref requires standard --fmt", file=sys.stderr)
            sys.exit(1)
        if args.n > FMT_CONFIG[fmt]["k_tile"]:
          print(f"ERROR: --ref=cuda requires n <= {FMT_CONFIG[fmt]['k_tile']}.", file=sys.stderr)
          sys.exit(1)
        _ensure_cuda_ext(fmt, arch_flag=args.arch)
        ref_fn = lambda A_, B_, C_, EB, SB: ref_cuda(A_, B_, C_, EB, SB, arch_flag=args.arch)
      elif args.ref == "cpp":
        _ensure_cpp_ext(args.cpp_source)
        ref_fn = lambda A_, B_, C_, EB, SB: ref_cpp(A_, B_, C_, EB, SB, args.W, args.HR, args.renorm, args.frm, args.no_window, args.cpp_source)
      else:
        ref_fn = lambda A_, B_, C_, EB, SB: ref_numpy(A_, B_, C_, EB, SB, frm=args.frm)

      result = fedp.dotp(A, B, C)
      ref = ref_fn(A, B, C, args.eb, args.sb)
      print(f"Result:     0x{result:08x}")
      print(f"Ref({args.ref}): 0x{ref:08x}")
      print(f"ULP diff:   {ulp_diff(result, ref)}")
    except Exception as e:
      print(f"Error parsing --run input: {e}", file=sys.stderr)
      sys.exit(0)

  test(args.n, args.eb, args.sb, args.frm, args.renorm,
       args.iters, args.seed, args.debug, args.trace, args.test,
       args.ref, args.arch, args.max_errors, args.W, args.HR, args.no_window, args.cpp_source)