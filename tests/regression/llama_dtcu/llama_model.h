#ifndef _LLAMA_DTCU_MODEL_H_
#define _LLAMA_DTCU_MODEL_H_

// Checkpoint and tokenizer loading, ported from the llama2.c code in
// tests/regression/llama/llama.cpp (same file format, same BPE encode/decode).

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

struct LlamaConfig {
  int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
};

struct LlamaWeights {
  const float* token_embedding_table;  // [vocab, dim]
  const float* rms_att_weight;         // [layer, dim]
  const float* rms_ffn_weight;         // [layer, dim]
  const float* wq;                     // [layer, dim, n_heads*head]
  const float* wk;                     // [layer, dim, n_kv_heads*head]
  const float* wv;
  const float* wo;                     // [layer, n_heads*head, dim]
  const float* w1;                     // [layer, hidden, dim]
  const float* w2;                     // [layer, dim, hidden]
  const float* w3;                     // [layer, hidden, dim]
  const float* rms_final_weight;       // [dim]
  const float* wcls;                   // [vocab, dim]
};

struct LlamaModel {
  LlamaConfig  cfg{};
  LlamaWeights w{};
  int     fd = -1;
  float*  data = nullptr;
  ssize_t file_size = 0;

  ~LlamaModel() {
    if (data && data != MAP_FAILED) munmap(data, file_size);
    if (fd != -1) close(fd);
  }

  bool load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "llama_dtcu: cannot open %s\n", path); return false; }
    if (fread(&cfg, sizeof(cfg), 1, f) != 1) { fclose(f); return false; }
    const int shared_weights = cfg.vocab_size > 0 ? 1 : 0;
    cfg.vocab_size = abs(cfg.vocab_size);
    fseek(f, 0, SEEK_END); file_size = ftell(f); fclose(f);
    fd = open(path, O_RDONLY);
    if (fd == -1) return false;
    data = (float*)mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) { data = nullptr; return false; }
    const float* ptr = data + sizeof(LlamaConfig) / sizeof(float);
    const int head_size = cfg.dim / cfg.n_heads;
    const unsigned long long L = cfg.n_layers;
    w.token_embedding_table = ptr; ptr += (size_t)cfg.vocab_size * cfg.dim;
    w.rms_att_weight = ptr;        ptr += L * cfg.dim;
    w.wq = ptr;                    ptr += L * cfg.dim * (cfg.n_heads * head_size);
    w.wk = ptr;                    ptr += L * cfg.dim * (cfg.n_kv_heads * head_size);
    w.wv = ptr;                    ptr += L * cfg.dim * (cfg.n_kv_heads * head_size);
    w.wo = ptr;                    ptr += L * (cfg.n_heads * head_size) * cfg.dim;
    w.rms_ffn_weight = ptr;        ptr += L * cfg.dim;
    w.w1 = ptr;                    ptr += L * cfg.dim * cfg.hidden_dim;
    w.w2 = ptr;                    ptr += L * cfg.hidden_dim * cfg.dim;
    w.w3 = ptr;                    ptr += L * cfg.dim * cfg.hidden_dim;
    w.rms_final_weight = ptr;      ptr += cfg.dim;
    ptr += (size_t)cfg.seq_len * head_size / 2;   // legacy freq_cis_real
    ptr += (size_t)cfg.seq_len * head_size / 2;   // legacy freq_cis_imag
    w.wcls = shared_weights ? w.token_embedding_table : ptr;
    return true;
  }
};

struct LlamaTokenizer {
  std::vector<std::string> vocab;
  std::vector<float> scores;
  std::vector<int> sorted;       // vocab ids sorted by string
  unsigned int max_token_length = 0;
  char byte_pieces[512];

  bool load(const char* path, int vocab_size) {
    vocab.resize(vocab_size); scores.resize(vocab_size);
    for (int i = 0; i < 256; ++i) { byte_pieces[i * 2] = (char)(unsigned char)i; byte_pieces[i * 2 + 1] = '\0'; }
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "llama_dtcu: cannot open %s\n", path); return false; }
    if (fread(&max_token_length, sizeof(int), 1, f) != 1) { fclose(f); return false; }
    for (int i = 0; i < vocab_size; ++i) {
      int len = 0;
      if (fread(&scores[i], sizeof(float), 1, f) != 1) { fclose(f); return false; }
      if (fread(&len, sizeof(int), 1, f) != 1) { fclose(f); return false; }
      vocab[i].resize(len);
      if (len && fread(&vocab[i][0], len, 1, f) != 1) { fclose(f); return false; }
    }
    fclose(f);
    sorted.resize(vocab_size);
    for (int i = 0; i < vocab_size; ++i) sorted[i] = i;
    std::sort(sorted.begin(), sorted.end(),
              [&](int a, int b) { return strcmp(vocab[a].c_str(), vocab[b].c_str()) < 0; });
    return true;
  }

  int lookup(const char* s) const {
    auto it = std::lower_bound(sorted.begin(), sorted.end(), s,
                               [&](int a, const char* key) { return strcmp(vocab[a].c_str(), key) < 0; });
    if (it != sorted.end() && strcmp(vocab[*it].c_str(), s) == 0) return *it;
    return -1;
  }

  // llama2.c encode(): optional BOS, dummy-prefix space token, UTF-8 codepoints with
  // byte fallback, then greedy best-score pair merging.
  void encode(const char* text, bool bos, bool eos, std::vector<int>& tokens) const {
    tokens.clear();
    if (bos) tokens.push_back(1);
    if (text[0] != '\0') tokens.push_back(lookup(" "));
    std::string buf;
    for (const char* c = text; *c != '\0'; ++c) {
      if ((*c & 0xC0) != 0x80) buf.clear();
      buf.push_back(*c);
      if ((*(c + 1) & 0xC0) == 0x80 && buf.size() < 4) continue;
      const int id = lookup(buf.c_str());
      if (id != -1) tokens.push_back(id);
      else for (unsigned char ch : buf) tokens.push_back((int)ch + 3);
      buf.clear();
    }
    while (true) {
      float best_score = -1e10f; int best_id = -1, best_idx = -1;
      for (size_t i = 0; i + 1 < tokens.size(); ++i) {
        const std::string s = vocab[tokens[i]] + vocab[tokens[i + 1]];
        const int id = lookup(s.c_str());
        if (id != -1 && scores[id] > best_score) { best_score = scores[id]; best_id = id; best_idx = (int)i; }
      }
      if (best_idx == -1) break;
      tokens[best_idx] = best_id;
      tokens.erase(tokens.begin() + best_idx + 1);
    }
    if (eos) tokens.push_back(2);
  }

  const char* decode(int prev_token, int token) const {
    const char* piece = vocab[token].c_str();
    if (prev_token == 1 && piece[0] == ' ') ++piece;
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) piece = byte_pieces + byte_val * 2;
    return piece;
  }
};

#endif // _LLAMA_DTCU_MODEL_H_
