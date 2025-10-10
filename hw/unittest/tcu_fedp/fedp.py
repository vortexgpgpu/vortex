#!/usr/bin/env python3
import argparse, random, struct, sys, ast
import numpy as np

ULP = 1

# ----------------- tiny helpers -----------------
def be32_from_float(x):
  return struct.unpack(">I", struct.pack(">f", float(np.float32(x))))[0]
def bias(eb): return (1<<(eb-1)) - 1

def split(x, eb, sb):
  s = (x>>(eb+sb)) & 1
  E = (x>>sb) & ((1<<eb)-1)
  F = x & ((1<<sb)-1)
  ALL1 = (1<<eb)-1
  if E==ALL1 and F!=0: return s,4,0,0        # qNaN
  if E==ALL1 and F==0: return s,3,0,0        # Inf
  if E==0 and F==0:    return s,0,0,0        # Zero
  if E==0:             return s,1,1-bias(eb),F    # Sub
  return s,2,E-bias(eb),(1<<sb)|F                 # Norm (implicit 1)

def lzc(x, width): return width if x==0 else width - x.bit_length()
def csa(a,b,c,mask):
  s = (a ^ b ^ c) & mask
  k = ((a & b) | (a & c) | (b & c)) << 1
  return s, (k & mask)
def cpa(s,c,mask): return (s + c) & mask
def twos_to_int(x,width): return x - (1<<width) if (x>>(width-1))&1 else x

# ----------------- FEDP core -----------------
class FEDP:
  def __init__(self, eb=5, sb=10, lanes=4, trace=False):
    self.eb,self.sb,self.lanes = eb,sb,lanes
    self.ebC,self.sbC = 8,23
    self.W,self.HR = 53,4   # W=53, headroom=4
    self.trace = bool(trace)

  # ---- S1: decode + multiply (no product renorm; LZC only for subnormals) ----
  def s1_decode_mul(self, A, B):
    sb=self.sb; sbmask=(1<<(sb+1))-1
    inv=nan=False; pos=neg=False; acc={}
    for a,b in zip(A,B):
      sa,ca,ea,Ma = split(a,self.eb,sb)
      sb_,cb,eb,Mb = split(b,self.eb,sb)
      if ca==4 or cb==4: nan=True; continue
      if (ca==3 and cb==0) or (cb==3 and ca==0): inv=True; continue
      if ca==3 or cb==3:
        pos |= ((sa^sb_)==0); neg |= ((sa^sb_)==1); continue
      if ca in (1,2) and cb in (1,2):
        if ca==1 and Ma: t=lzc(Ma,sb)+1; Ma=(Ma<<t)&sbmask; ea-=t
        if cb==1 and Mb: t=lzc(Mb,sb)+1; Mb=(Mb<<t)&sbmask; eb-=t
        P = Ma*Mb
        e = ea+eb-2*sb
        sgn = sa^sb_
        acc[e] = acc.get(e,0) + (-P if sgn else P)
    terms=[(1 if v<0 else 0, -v if v<0 else v, e) for e,v in acc.items() if v!=0]
    if self.trace:
        print(f"  [S1:decode_mul] terms={len(terms)} inv={inv} nan={nan} pos={pos} neg={neg}")
    return terms, inv, nan, pos, neg

  def s1_mapC(self, Cbits):
    sc,cc,eu,Mc = split(Cbits,self.ebC,self.sbC)
    eC = eu - self.sbC
    if self.trace:
        print(f"  [S1:mapC] sc={sc} Mc=0x{Mc:x} eC={eC} cc={cc}")
    cc = (-3 if sc else 3) if cc==3 else cc
    return sc,Mc,eC,cc

  # ---- S2: align to accumulator window ----
  def s2_align(self, terms, Cp):
    sc,Mc,eC,cc = Cp
    sb=self.sb
    W,WW = self.W, self.W + self.HR
    tops=[]
    for s,P,e in terms:
      if P:
        hi2 = (P >> (2*sb+1)) & 1
        tops.append(e + 2*sb + hi2)
    if Mc and cc in (1,2):
      tops.append(eC + self.sbC)
    if not tops:
        if self.trace: print(f"  [S2:align] no terms, skipping")
        return [], 0, 0, cc
    L = max(tops) - (W-1)

    aligned_terms = []; sticky = 0
    def align_and_add(val, e):
        nonlocal sticky
        k=e-L
        if k>=0:
          aligned_terms.append(val<<k)
        else:
          sh=-k
          mag = -val if val<0 else val
          part = mag>>sh
          T = ((-part) if val<0 else part)
          aligned_terms.append(T)
          sticky |= (mag & ((1<<sh)-1))!=0

    for s,P,e in terms: align_and_add(-P if s else P, e)
    if Mc and cc in (1,2): align_and_add(-Mc if sc else Mc, eC)

    if self.trace:
        print(f"  [S2:align] aligned_terms={len(aligned_terms)} sticky={sticky} L={L}")
    return aligned_terms, sticky, L, cc

  # ---- S3: accumulate aligned terms ----
  def s3_accumulate(self, aligned_terms):
    if not aligned_terms:
        if self.trace: print(f"  [S3:accumulate] Vw=0")
        return 0
    WW = self.W + self.HR
    mask = (1 << WW) - 1
    s_acc = c_acc = 0
    for T in aligned_terms:
        s_acc, c_acc = csa(s_acc, c_acc, T & mask, mask)
    Vw_unsigned = cpa(s_acc, c_acc, mask)
    Vw = twos_to_int(Vw_unsigned, WW)
    if self.trace:
        print(f"  [S3:accumulate] Vw={Vw} (0x{Vw_unsigned:x})")
    return Vw

  # ---- S4: normalize the accumulated sum ----
  def s4_normalize(self, V, st, L):
    WW=self.W + self.HR
    if V==0:
        if self.trace: print(f"  [S4:normalize] zero in, zero out")
        return 0,0,0,st, -10**9 # Pass incoming sticky through
    s = 1 if V<0 else 0
    X = (-V if V<0 else V) & ((1<<WW)-1)
    i = WW-1 - lzc(X, WW)
    e = L + i
    sh = (i+1) - 24
    if sh>=0:
      kept = (X>>sh) & ((1<<24)-1)
      rem  = X & ((1<<sh)-1)
      g    = (rem>>(sh-1))&1 if sh>0 else 0
      low  = rem & ((1<<(sh-1))-1) if sh>1 else 0
      st_out = 1 if (low!=0 or st) else 0
      if g==1 and low==0:
        st_out = 0
    else:
      kept = (X<<(-sh)) & ((1<<24)-1)
      g=0
      st_out = st # if we shift left, original sticky is all we have
    if self.trace:
        print(f"  [S4:normalize] s={s} kept=0x{kept:06x} g={g} st={st_out} e={e}")
    return s,kept,g,st_out,e

  # ---- S5: IEEE-754 single-precision rounding ----
  def s5_rounding(self,s,kept,g,st,e,cc,inv,nan,pos,neg):
    if nan or inv or (pos and neg):
      result = 0x7fc00000
    elif cc == 4:
      result = 0x7fc00000
    elif cc in (3, -3):
      cneg = 1 if cc < 0 else 0
      if pos or neg:
        if (pos and not neg and cneg==1) or (neg and not pos and cneg==0):
          result = 0x7fc00000
        else:
          result = 0xff800000 if cneg else 0x7f800000
      else:
        result = 0xff800000 if cneg else 0x7f800000
    elif pos and not neg:
      result = 0x7f800000
    elif neg and not pos:
      result = 0xff800000
    elif kept==0 and st==0:
      result = (s<<31)
    else:
      result = self._f32_round_pack(s,kept,g,st,e)
    if self.trace:
      print(f"  [S5:rounding] final=0x{result:08x}")
    return result

  @staticmethod
  def _f32_round_pack(s, kept24, g, st, e_unb):
    be = e_unb + 127
    if be >= 255: return (0xff800000 if s else 0x7f800000)
    if be <= 0:
      k = 1 - be
      wide = (kept24<<2) | (g<<1) | (1 if st else 0)
      sh = k + 2
      m = wide >> sh
      rb = (wide>>(sh-1))&1 if sh>0 else 0
      tail = wide & ((1<<(sh-1))-1) if sh>1 else 0
      inc = rb & ((tail!=0) | ((m&1)==1))
      m += inc
      if m >= (1<<23): return (s<<31)|(1<<23)
      return (s<<31) | m
    inc = g & (st | ((kept24&1)==1))
    frac = kept24 & 0x7fffff
    m = frac + inc
    be += 1 if m >= (1<<23) else 0
    m &= (1<<23)-1
    if be >= 255: return (0xff800000 if s else 0x7f800000)
    return (s<<31) | ((be&0xff)<<23) | m

  # ---- branchless top ----
  def dotp(self,A,B,Cbits):
    t,inv,nan,pos,neg = self.s1_decode_mul(A,B)
    Cp = self.s1_mapC(Cbits)
    aligned, sticky1, L, cc = self.s2_align(t, Cp)
    V = self.s3_accumulate(aligned)
    s,kept,g,sticky2,e = self.s4_normalize(V,sticky1,L)
    return self.s5_rounding(s,kept,g,sticky2,e,cc,inv,nan,pos,neg)

# ----------------- reference + harness (unchanged) -----------------
def _to_float_np(x, eb, sb):
  s = (x>>(eb+sb)) & 1
  E = (x>>sb) & ((1<<eb)-1)
  F = x & ((1<<sb)-1)
  b = bias(eb)
  if E == 0 and F == 0:
    return np.longdouble(-0.0) if s else np.longdouble(0.0)
  if E == (1<<eb)-1:
    return np.longdouble(np.nan) if F else (np.longdouble(-np.inf) if s else np.longdouble(np.inf))
  if E == 0:
    m = np.longdouble(F) / np.longdouble(1<<sb); e = 1 - b
  else:
    m = np.longdouble(1.0) + np.longdouble(F) / np.longdouble(1<<sb); e = E - b
  val = np.ldexp(m, int(e))
  return -val if s else val

def ref_numpy(A, B, Cbits, eb, sb):
  a_ld = np.array([_to_float_np(x, eb, sb) for x in A], dtype=np.longdouble)
  b_ld = np.array([_to_float_np(x, eb, sb) for x in B], dtype=np.longdouble)
  c_ld = np.longdouble(_to_float_np(Cbits, 8, 23))
  with np.errstate(invalid='ignore', over='ignore'):
    s_ld = (a_ld * b_ld).sum(dtype=np.longdouble) + c_ld
  return be32_from_float(s_ld)

def ulp_diff(a,b):
  if (a & 0x7f800000)==0x7f800000 and (a & 0x007fffff)!=0: return 0
  if (b & 0x7f800000)==0x7f800000 and (b & 0x007fffff)!=0: return 0
  def to_ordered(u): return u ^ 0x80000000
  return abs(to_ordered(a) - to_ordered(b))

def _pack_fp(s, e, f, eb, sb): return (s<<(eb+sb)) | (e<<sb) | f

def generate_fp_value(feature, eb, sb, test_id):
    """Directed value generator, inspired by the C++ harness."""
    all_exp = (1<<eb)-1; max_frac = (1<<sb)-1
    s = (test_id ^ (hash(feature)*0x9E3779B9)) & 1

    if feature == "zeros":
        return _pack_fp(test_id & 1, 0, 0, eb, sb)
    elif feature == "infinities":
        return _pack_fp(test_id & 1, all_exp, 0, eb, sb)
    elif feature == "nans":
        if sb == 0: return _pack_fp(s, all_exp, 0, eb, sb) # Cannot form NaN
        qbit = 1<<(sb-1); payload = (test_id % (qbit-1 if qbit>1 else 1)) + 1
        return _pack_fp(s, all_exp, qbit | payload, eb, sb)
    elif feature == "subnormals":
        if sb == 0: return _pack_fp(s, 0, 0, eb, sb)
        case = test_id % 3
        if case == 0: f = 1          # Smallest
        elif case == 1: f = max_frac # Largest
        else: f = random.randint(1, max_frac)
        return _pack_fp(s, 0, f, eb, sb)
    elif feature == "normals":
        if eb < 2: return _pack_fp(s, 0, 0, eb, sb)
        case = test_id % 5
        if case == 0: # Smallest normal
            e, f = 1, random.randint(0, max_frac)
        elif case == 1: # Largest finite
            e, f = all_exp - 1, max_frac
        elif case == 2: # Near 1.0
            e, f = bias(eb), random.randint(0, 15) & max_frac
        else: # Random
            e, f = random.randint(1, all_exp-1), random.randint(0, max_frac)
        return _pack_fp(s, e, f, eb, sb)
    return 0 # Default case

def _mk_cancel_cases(n, eb, sb, count):
    cases = []
    for _ in range(count):
        E = random.randint(1,(1<<eb)-2); F = random.getrandbits(sb)
        a = (random.getrandbits(1)<<(eb+sb))|(E<<sb)|F
        b = (random.getrandbits(1)<<(eb+sb))| (random.randint(1,(1<<eb)-2)<<sb) | random.getrandbits(sb)
        a2 = a ^ (1<<(eb+sb)); A=[]; B=[]
        for _ in range(n//2): A+=[a,a2]; B+=[b,b]
        if n%2==1: A.append(generate_fp_value("normals",eb,sb,0)); B.append(generate_fp_value("normals",eb,sb,1))
        C = be32_from_float(np.random.choice([0.0, 2.0**-149, -2.0**-149]))
        cases.append((A,B,C))
    return cases

CUSTOM_CASES = [[["0xfbff","0xfbff"],["0x83ff","0x7bff"],"0x4f7f8000"],[["0xe150","0xf4d7","0x4bcc","0xf3c1"],["0x83ff","0xda97","0x83ff","0x7ac6"],"0x4e51ad09"]]
def _mk_custom(n):
    def _pi(x): return int(x,16) if isinstance(x,str) else int(x)
    return [([_pi(v) for v in A], [_pi(v) for v in B], _pi(C)) for A,B,C in CUSTOM_CASES if len(A)==n]

def _print_case(tag, idx, A, B, C):
  a_str = ",".join(f"0x{x:04x}" for x in A)
  b_str = ",".join(f"0x{x:04x}" for x in B)
  c_str = f"0x{C:08x}"
  print(f'[{tag} #{idx}] inputs="[{a_str}];[{b_str}];{c_str}"')

def _on_fail(tag, idx, A,B,C, got, ref):
    print(f"✘ FAIL {tag} #{idx}: got=0x{got:08x} ref=0x{ref:08x} ulp={ulp_diff(got,ref)}")
    _print_case(tag, idx, A,B,C)

def test(n, eb, sb, iters, seed, debug, trace, test_id_filter):
    random.seed(seed); np.random.seed(seed)
    fedp=FEDP(eb,sb,n,trace=trace)

    features = ["normals","subnormals","zeros","infinities","nans","cancellation","custom"]
    tests_per_feature = max(1, (iters + len(features) - 1) // len(features))

    # Pre-generate cases for special generators
    cancel_cases = _mk_cancel_cases(n, eb, sb, tests_per_feature)
    custom_cases = _mk_custom(n)

    ok_by_feature = {f: [0,0] for f in features}

    for test_id in range(iters):
        if test_id_filter is not None and test_id != test_id_filter: continue

        feature_idx = test_id // tests_per_feature
        if feature_idx >= len(features): continue
        feature = features[feature_idx]
        sub_id = test_id % tests_per_feature

        A, B, C = [], [], 0
        if feature == "cancellation":
            if sub_id < len(cancel_cases): A, B, C = cancel_cases[sub_id]
            else: continue
        elif feature == "custom":
            if sub_id < len(custom_cases): A, B, C = custom_cases[sub_id]
            else: continue
        else: # Use the new systematic generator
            A, B = [0]*n, [0]*n
            op_focus = test_id % 3
            for i in range(n):
                a_feat = feature if (op_focus==0 and i%2==0) else "normals"
                A[i] = generate_fp_value(a_feat, eb, sb, sub_id+i)
                b_feat = feature if (op_focus==1 and i%2==0) else "normals"
                B[i] = generate_fp_value(b_feat, eb, sb, sub_id+n+i)
            c_feat = feature if op_focus==2 else "normals"
            C = generate_fp_value(c_feat, 8, 23, sub_id)

        # Run the case
        if debug or trace: _print_case(f"{feature} TID {test_id}", sub_id, A,B,C)
        got = fedp.dotp(A,B,C)
        ref = ref_numpy(A,B,C,eb,sb)

        passed = ulp_diff(got,ref) <= ULP
        if not passed: _on_fail(f"{feature} TID {test_id}", sub_id, A,B,C, got, ref)

        ok_by_feature[feature][0] += 1 if passed else 0
        ok_by_feature[feature][1] += 1

    print("-"*40)
    total_ok, total_tot = 0, 0
    for f, (ok, tot) in ok_by_feature.items():
        if tot == 0: continue
        mark = "✔" if ok==tot else "✘"
        print(f"{mark} {f}: {ok}/{tot} ({100.0*ok/tot:.1f}%)")
        total_ok += ok; total_tot += tot

    print("-"*40)
    if total_tot > 0:
        print(f"OVERALL: {total_ok}/{total_tot} ({100.0*total_ok/total_tot:.1f}%)")
    elif test_id_filter is not None:
        print(f"Could not find test with TID #{test_id_filter}")

if __name__=="__main__":
  ap=argparse.ArgumentParser()
  ap.add_argument("--n",type=int,default=4)
  ap.add_argument("--eb",type=int,default=5)
  ap.add_argument("--sb",type=int,default=10)
  ap.add_argument("--iters",type=int,default=100000)
  ap.add_argument("--seed",type=int,default=1)
  ap.add_argument("--debug", action="store_true", help="print every test scenario")
  ap.add_argument("--trace", action="store_true", help="print per-stage outputs")
  ap.add_argument("--test",type=int,default=None, help="Run a single test by its global TID")
  ap.add_argument("--run",type=str,default=None, help="Directly run a single case. Format: '[A];[B];C'")
  args=ap.parse_args()

  if args.run:
      try:
          A_str, B_str, C_str = args.run.split(';')
          A,B,C = ast.literal_eval(A_str), ast.literal_eval(B_str), ast.literal_eval(C_str)
          fedp=FEDP(args.eb,args.sb,len(A),trace=args.trace)
          if not args.trace: _print_case("direct", 0, A,B,C)
          result = fedp.dotp(A,B,C); ref = ref_numpy(A,B,C,args.eb,args.sb)
          print(f"Result: 0x{result:08x}\nNumpy:  0x{ref:08x}\nULP diff: {ulp_diff(result, ref)}")
      except Exception as e:
          print(f"Error parsing --run input: {e}", file=sys.stderr)
      sys.exit(0)

  test(args.n,args.eb,args.sb,args.iters,args.seed,args.debug,args.trace,args.test)