import subprocess
import pandas as pd

def clog2(x: int) -> int:
    return 0 if x < 2 else 1 + clog2(x // 2)

def sizeof(dtype):
    if dtype == 'float64_t':
        return 8
    elif dtype == 'float32_t':
        return 4
    elif dtype == 'float16_t':
        return 2
    elif dtype == 'int64_t':
        return 8
    elif dtype == 'int32_t':
        return 4
    elif dtype == 'int16_t':
        return 2
    elif dtype == 'int8_t':
        return 1
    else:
        raise ValueError(f"Unsupported data type: {dtype}")

class WMMAConfig:
    def __init__(self, NT, NR, XB, Ot, It, DP):
        self.NT = NT
        self.NR = NR
        self.XB = XB
        self.Ot = Ot
        self.It = It
        self.DP = DP

        self.i_ratio = XB / sizeof(It)
        self.o_ratio = XB / sizeof(Ot)

        # compute xtileM / xtileN / xtileK
        tile_cap = NT * NR
        lg_tile_cap = clog2(tile_cap)
        tile_eN = lg_tile_cap // 2
        tile_eM = lg_tile_cap - tile_eN
        self.xtileM = 1 << tile_eM
        self.xtileN = 1 << tile_eN
        self.xtileK = tile_cap // max(self.xtileM, self.xtileN)

        # compute tcM / tcN / tcK
        block_cap = NT
        lg_block_cap = clog2(block_cap)
        block_en = lg_block_cap // 2
        block_em = lg_block_cap - block_en
        self.tcM = (1 << block_em)
        self.tcN = (1 << block_en)
        if DP == 0:
            self.tcK = block_cap // max(self.tcM, self.tcN)
        else:
            self.tcK = DP

        assert (self.xtileM * self.xtileK <= tile_cap), "xtileM * xtileK <= tile_cap"
        assert (self.xtileN * self.xtileK <= tile_cap), "xtileN * xtileK <= tile_cap"
        assert (self.xtileM * self.xtileN <= tile_cap), "xtileM * xtileN <= tile_cap"

        assert (self.tcM * self.tcK <= block_cap), "tcM * tcK <= block_cap"
        assert (self.tcK * self.tcN <= block_cap), "tcK * tcN <= block_cap"
        assert (self.tcM * self.tcN <= block_cap), "tcM * tcN <= block_cap"

        # registers per fragment
        self.nRA = (self.xtileM * self.xtileK) // NT
        self.nRB = (self.xtileN * self.xtileK) // NT
        self.nRC = (self.xtileM * self.xtileN) // NT

        # tile dimensions adjusted for fragment types
        self.tileM = self.xtileM
        self.tileN = self.xtileN
        self.tileK = self.xtileK * self.i_ratio

    def validate_dims(self, M, N, K):
        assert M % self.tileM == 0, f"M({M}) % tileM({self.tileM}) != 0"
        assert N % self.tileN == 0, f"N({N}) % tileN({self.tileN}) != 0"
        assert K % self.tileK == 0, f"K({K}) % tileK({self.tileK}) != 0"

    def dump(self):
        print(f"WMMAConfig(NT={self.NT}, NR={self.NR}, XB={self.XB}, It={self.It}, Ot={self.Ot})")
        print(f"Tile: M={self.tileM}, N={self.tileN}, K={self.tileK}")
        print(f"TC: M={self.tcM}, N={self.tcN}, K={self.tcK}")
        print(f"Registers: nRA={self.nRA}, nRB={self.nRB}, nRC={self.nRC}")

def verify(NT, XB, Ot, It, DPLEN):
    compile_cmd = [
        "gcc", "-std=c++17", "-O2", "-DNDEBUG", "../tests/regression/sgemm_tcu/tensor_generic.cpp", "-lstdc++",
        f"-DNUM_THREADS={NT}", f"-DXLENB={XB}", f"-DOTYPE={Ot}", f"-DITYPE={It}", f"-DDPLEN={DPLEN}"
    ]
    print("Running:", " ".join(compile_cmd))
    comp = subprocess.run(compile_cmd, capture_output=True)
    if comp.returncode != 0:
        return "Compile Error!"
    run = subprocess.run(["./a.out"])
    if run.returncode != 0:
        return "Failed!"
    return "Passed!"

# Parameters
NR = 8
XB_list = [4]

# [It, Ot] pairs
ItxOt_list = [
    ['float16_t', 'float16_t'],
    ['float16_t', 'float32_t'],
    ['float32_t', 'float32_t'],
    ['int8_t', 'int32_t'],
    ['int16_t', 'int32_t'],
    ['int32_t', 'int32_t']
]

NT_list = [1, 2, 4, 8, 16, 32, 64]

rows = []
for XB in XB_list:
    for NT in NT_list:
        for It, Ot in ItxOt_list:
            # Calculate WMMAConfig with DP=0
            cfg0 = WMMAConfig(NT, NR, XB, Ot, It, 0)
            dplen = cfg0.tcK
            while dplen != 0:
                print(f"Calculating WMMAConfig for NT={NT}, NR={NR}, XB={XB}. Ot={Ot}, It={It}, DPLEN={dplen}")
                cfg = WMMAConfig(NT, NR, XB, Ot, It, dplen)

                m_steps = cfg.xtileM // cfg.tcM
                n_steps = cfg.xtileN // cfg.tcN
                k_steps = cfg.xtileK // cfg.tcK

                a_block_size = cfg.tcM * cfg.tcK  # size of A micro-tile
                a_sub_blocks = NT // a_block_size  # number of A micro-tiles per register
                a_sub_steps  = m_steps // a_sub_blocks  # number of A sub-steps per register

                b_block_size = cfg.tcK * cfg.tcN  # size of B micro-tile
                b_sub_blocks = NT // b_block_size  # number of B micro-tiles per register
                b_sub_steps  = n_steps // b_sub_blocks  # number of B sub-steps per register

                if a_sub_steps == 0:
                    print(f"Skipping NT={NT}, DPLEN={dplen} due to a_sub_steps=0")
                    dplen //= 2
                    continue

                if b_sub_steps == 0:
                    print(f"Skipping NT={NT}, DPLEN={dplen} due to b_sub_steps=0")
                    dplen //= 2
                    continue

                status = verify(NT, XB, Ot, It, dplen)
                rows.append({
                    'NT': NT,
                    'XB': XB,
                    'Ot': Ot,
                    'It': It,
                    'tileM': cfg.tileM,
                    'tileN': cfg.tileN,
                    'tileK': cfg.tileK,
                    'tcM': cfg.tcM,
                    'tcN': cfg.tcN,
                    'tcK': cfg.tcK,
                    'nRA': cfg.nRA,
                    'nRB': cfg.nRB,
                    'nRC': cfg.nRC,
                    'm_steps': m_steps,
                    'n_steps': n_steps,
                    'k_steps': k_steps,
                    'a_block_size': a_block_size,
                    'b_block_size': b_block_size,
                    'status': status
                })
                dplen //= 2

df = pd.DataFrame(rows)
print(df.to_string(index=False))