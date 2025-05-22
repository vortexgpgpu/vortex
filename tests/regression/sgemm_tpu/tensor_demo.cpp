#include <array>
#include <cassert>
#include <cmath>
#include <iostream>

// 32-lane vector register storing raw 32-bit floats
struct VReg {
  std::array<float, 32> data;
};

// 16×16 matrix stored as flat array
struct Mat16 {
  std::array<float, 16 * 16> data;
  // element access
  float &operator()(int row, int col) { return data[row * 16 + col]; }
  const float &operator()(int row, int col) const { return data[row * 16 + col]; }
};

// Per-lane pack into vA using flat mdata (size 256)
void load_direct(std::array<VReg, 8> &vR, int lane, const std::array<float, 16 * 16> &mdata) {
  int i = lane / 4;
  int j = lane % 4;
  for (int r = 0; r < 8; ++r) {
    int rt = r / 4;
    int k = r % 4;
    int row = rt * 8 + i;
    int col = k * 4 + j;
    vR[r].data[lane] = mdata[row * 16 + col];
  }
}

// Per-lane store from vD into flat mdata
void store_direct(std::array<float, 16 * 16> &mdata, int lane, const std::array<VReg, 8> &vR) {
  int i = lane / 4;
  int j = lane % 4;
  for (int r = 0; r < 8; ++r) {
    int rt = r / 4;
    int cb = r % 4;
    int row = rt * 8 + i;
    int col = cb * 4 + j;
    mdata[row * 16 + col] = vR[r].data[lane];
  }
}

// Per-lane pack into vB using flat mdata
void load_transposed(std::array<VReg, 8> &vR, int lane, const std::array<float, 16 * 16> &mdata) {
  int block = lane / 16;
  int off = lane % 16;
  int bi = off / 4;
  int bj = off % 4;
  for (int r = 0; r < 8; ++r) {
    int k = r / 2;
    int tile = r % 2;
    int cb = tile * 2 + block;
    int row = k * 4 + bi;
    int col = cb * 4 + bj;
    vR[r].data[lane] = mdata[row * 16 + col];
  }
}

// 8×4×4 micro‑op: vd = (va @ subB) + vc
VReg hmma_844(int step_idx, const VReg &va, const VReg &vb, const VReg &vc) {
  float subA[8][4], acc[8][4];
  for (int x = 0; x < 8; ++x)
    for (int y = 0; y < 4; ++y) {
      subA[x][y] = va.data[x * 4 + y];
      acc[x][y] = vc.data[x * 4 + y];
    }
  int cb = step_idx & 3;
  int half = cb & 1;
  float subB[4][4];
  int off = half * 16;
  for (int x = 0; x < 4; ++x)
    for (int y = 0; y < 4; ++y)
      subB[x][y] = vb.data[off + x * 4 + y];
  VReg vd{};
  for (int x = 0; x < 8; ++x) {
    for (int y = 0; y < 4; ++y) {
      float sum = 0;
      for (int z = 0; z < 4; ++z)
        sum += subA[x][z] * subB[z][y];
      vd.data[x * 4 + y] = acc[x][y] + sum;
    }
  }
  return vd;
}

// Full 16×16×16 HMMA using per-lane loads/stores
Mat16 HMMA_161616_844_fp32(const Mat16 &A, const Mat16 &B, const Mat16 &C) {
  Mat16 D{};
  std::array<VReg, 8> vA{}, vB{}, vC{}, vD{};

  // per-lane load
  for (int lane = 0; lane < 32; ++lane) {
    load_direct(vA, lane, A.data);
    load_transposed(vB, lane, B.data);
    load_direct(vC, lane, C.data);
  }

  // micro-ops
  for (int k = 0; k < 4; ++k) {
    for (int step = 0; step < 8; ++step) {
      int rt = step / 4;
      int cb = step % 4;
      const VReg &vc = (k == 0 ? vC[step] : vD[step]);
      vD[step] = hmma_844(step, vA[rt * 4 + k], vB[2 * k + (cb / 2)], vc);
    }
  }

  // per-lane store
  for (int lane = 0; lane < 32; ++lane) {
    store_direct(D.data, lane, vD);
  }
  return D;
}

int main() {
  Mat16 A, B, C, Dref;
  // init
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j) {
      A(i, j) = i * 16 + j;
      B(i, j) = i * 16 + j;
      C(i, j) = i * 16 + j;
    }
  // ref
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j) {
      float s = 0;
      for (int k = 0; k < 16; ++k)
        s += A(i, k) * B(k, j);
      Dref(i, j) = s + C(i, j);
    }
  auto D = HMMA_161616_844_fp32(A, B, C);
  float err = 0;
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j)
      err = std::max(err, std::fabs(D(i, j) - Dref(i, j)));
  std::cout << "Max abs error: " << err << "\n"
            << (err < 1e-4f ? "PASSED!" : "FAILED!") << '\n';
  return 0;
}
