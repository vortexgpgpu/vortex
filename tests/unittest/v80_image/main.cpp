#include <v80_image.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

using namespace vortex;

static bool check(bool cond, const char* what) {
  if (!cond) {
    printf("Error: %s\n", what);
  }
  return cond;
}

static void write_file(const std::string& path, const std::string& data) {
  FILE* f = fopen(path.c_str(), "wb");
  if (f == nullptr) {
    perror(path.c_str());
    exit(1);
  }
  fwrite(data.data(), 1, data.size(), f);
  fclose(f);
}

int main() {
  char tmpl[] = "v80_image.XXXXXX";
  const char* made = mkdtemp(tmpl);
  if (made == nullptr) {
    perror("mkdtemp");
    return 1;
  }
  const std::string dir = made;
  const std::string sysfs = dir + "/sysfs";
  mkdir(sysfs.c_str(), 0755);
  bool ok = true;

  ok &= check(v80_normalize_bdf("01:00") == "0000:01:00", "short bdf");
  ok &= check(v80_normalize_bdf("01:00.0") == "0000:01:00", "short bdf with function");
  ok &= check(v80_normalize_bdf("0000:01:00") == "0000:01:00", "full bdf");
  ok &= check(v80_normalize_bdf("0001:a3:00.2") == "0001:a3:00", "full bdf with function");

  // The digest follows the content, not the name.
  const std::string a = dir + "/a.vbin", a2 = dir + "/a_copy.vbin";
  const std::string b = dir + "/b.vbin", grown = dir + "/grown.vbin";
  const std::string empty = dir + "/empty.vbin";
  write_file(a, std::string(200000, 'x'));
  write_file(a2, std::string(200000, 'x'));
  write_file(b, std::string(199999, 'x') + "y");
  write_file(grown, std::string(200001, 'x'));
  write_file(empty, "");
  const uint64_t da = v80_image_digest(a);
  ok &= check(da != 0, "digest of a readable file");
  ok &= check(da == v80_image_digest(a2), "same content, same digest");
  ok &= check(da != v80_image_digest(b), "one byte differs");
  ok &= check(da != v80_image_digest(grown), "one byte longer");
  ok &= check(v80_image_digest(empty) != 0, "an empty file is still a readable one");
  ok &= check(v80_image_digest(dir + "/missing.vbin") == 0, "unreadable file");

  // A card that is not on the bus holds nothing.
  v80_image_record rec("01:00", dir, sysfs);
  v80_image_t image, got;
  image.digest = da;
  for (int i = 0; i < v80_image_t::kIdentWords; ++i) {
    image.ident[i] = 0x1000u + i;
  }
  ok &= check(rec.pci_instance() == 0, "no card, no instance");
  ok &= check(rec.clear(), "clearing an absent record");
  ok &= check(!rec.holds(da, &got), "no record");

  // The instance needs both functions, and follows each of them.
  const std::string pf0 = sysfs + "/0000:01:00.0", pf2 = sysfs + "/0000:01:00.2";
  mkdir(pf0.c_str(), 0755);
  ok &= check(rec.pci_instance() == 0, "PF0 alone is not a card with a shell");
  v80_image_t orphan = image;
  ok &= check(rec.store(orphan) && !rec.holds(da, &got), "a record without an instance holds nothing");
  mkdir(pf2.c_str(), 0755);
  image.pci_instance = rec.pci_instance();
  ok &= check(image.pci_instance != 0, "instance of a present card");
  // A function that is removed and added back is a new node; a rename onto a
  // fresh directory is how that looks from here.
  const std::string fresh = sysfs + "/fresh";
  mkdir(fresh.c_str(), 0755);
  rename(fresh.c_str(), pf2.c_str());
  ok &= check(rec.pci_instance() != 0 && rec.pci_instance() != image.pci_instance,
              "PF2 added back: a new instance");
  image.pci_instance = rec.pci_instance();
  mkdir(fresh.c_str(), 0755);
  rename(fresh.c_str(), pf0.c_str());
  ok &= check(rec.pci_instance() != 0 && rec.pci_instance() != image.pci_instance,
              "PF0 added back: a new instance");
  image.pci_instance = rec.pci_instance();
  ok &= check(v80_image_record("0000:01:00.1", dir, sysfs).path() == rec.path(),
              "every spelling of the card shares one record");

  ok &= check(rec.store(image), "store");
  // Whoever writes the card next has to be able to void this record.
  struct stat st;
  ok &= check(stat(rec.path().c_str(), &st) == 0 && (st.st_mode & 0777) == 0666,
              "the record is writable by every user");
  ok &= check(rec.holds(da, &got), "the stored image is held");
  bool same = got.digest == image.digest && got.pci_instance == image.pci_instance;
  for (int i = 0; i < v80_image_t::kIdentWords; ++i) {
    same &= got.ident[i] == image.ident[i];
  }
  ok &= check(same, "the record round-trips");
  ok &= check(!rec.holds(v80_image_digest(b), &got), "another image is not held");
  ok &= check(!rec.holds(0, &got), "an unreadable image is never held");

  // A shorter record over a longer one leaves nothing of the longer one.
  v80_image_t small = image;
  for (int i = 0; i < v80_image_t::kIdentWords; ++i) {
    small.ident[i] = 1;
  }
  ok &= check(rec.store(small) && rec.load(&got) && got.ident[5] == 1, "shorter record over a longer one");
  ok &= check(rec.store(image), "store back");

  // The card was enumerated again since the record was written.
  v80_image_t stale = image;
  stale.pci_instance = image.pci_instance + 1;
  ok &= check(rec.store(stale), "store for an earlier enumeration");
  ok &= check(!rec.holds(da, &got), "a record from an earlier enumeration is void");

  // Another card has its own record.
  mkdir((sysfs + "/0000:02:00.0").c_str(), 0755);
  mkdir((sysfs + "/0000:02:00.2").c_str(), 0755);
  ok &= check(rec.store(image), "store again");
  ok &= check(!v80_image_record("02:00", dir, sysfs).holds(da, &got), "per-card records");

  ok &= check(rec.clear(), "clear");
  ok &= check(!rec.holds(da, &got), "a cleared record holds nothing");
  ok &= check(rec.store(image) && rec.holds(da, &got), "store after clear");

  // What the device answers decides what is done with a held image.
  uint32_t words[v80_image_t::kIdentWords];
  for (int i = 0; i < v80_image_t::kIdentWords; ++i) {
    words[i] = image.ident[i];
  }
  ok &= check(v80_image_check(image, words) == v80_image_check_t::same, "the recorded words");
  for (int i = 0; i < v80_image_t::kIdentWords; ++i) {
    words[i] ^= 0x100;
    ok &= check(v80_image_check(image, words) == v80_image_check_t::different, "one word differs");
    words[i] = 0xFFFFFFFFu;
    ok &= check(v80_image_check(image, words) == v80_image_check_t::different, "one word reads all-ones");
    words[i] = image.ident[i];
  }
  for (int i = 0; i < v80_image_t::kIdentWords; ++i) {
    words[i] = 0xFFFFFFFFu;
  }
  ok &= check(v80_image_check(image, words) == v80_image_check_t::silent, "every word reads all-ones");
  ok &= check(!v80_image_answers(words), "all-ones is no answer");
  ok &= check(v80_image_check(image, nullptr) == v80_image_check_t::silent, "a read failed");
  // An all-ones identity is never the recorded image, even if one was stored.
  v80_image_t ones = image;
  for (int i = 0; i < v80_image_t::kIdentWords; ++i) {
    ones.ident[i] = 0xFFFFFFFFu;
  }
  ok &= check(v80_image_check(ones, words) == v80_image_check_t::silent, "all-ones against all-ones");

  // Only a regular file is a record: nothing else is followed or waited on.
  const std::string target = dir + "/target";
  ok &= check(rec.store(image), "store before the link");
  rename(rec.path().c_str(), target.c_str());
  ok &= check(symlink(target.c_str(), rec.path().c_str()) == 0, "plant a symlink");
  ok &= check(!rec.load(&got) && !rec.holds(da, &got), "a symlink is not read");
  ok &= check(!rec.store(image) && !rec.clear(), "a symlink is not written or voided");
  v80_image_record direct("09:00", dir, sysfs);
  write_file(direct.path(), "");
  rename(target.c_str(), direct.path().c_str());
  ok &= check(direct.load(&got) && got.digest == da, "the link's target was left alone");
  unlink(rec.path().c_str());
  ok &= check(mkfifo(rec.path().c_str(), 0666) == 0, "plant a fifo");
  ok &= check(!rec.load(&got) && !rec.store(image) && !rec.clear(), "a fifo is not a record");
  unlink(rec.path().c_str());

  // Anything that does not parse as one record is no record.
  write_file(rec.path(), "vortex-v80-image 1 1234 5678 1 2 3\n");
  ok &= check(!rec.load(&got), "truncated record");
  write_file(rec.path(), "vortex-v80-image 2 1234 5678 1 2 3 4 5 6\n");
  ok &= check(!rec.load(&got), "unknown record version");
  write_file(rec.path(), "vortex-v80-image 1 1234 5678 1 2 3 4 5 6 7\n");
  ok &= check(!rec.load(&got), "trailing data");
  write_file(rec.path(), "vortex-v80-image 1 0 5678 1 2 3 4 5 6\n");
  ok &= check(!rec.load(&got), "a zero digest names no image");

  const std::string cleanup = "rm -rf '" + dir + "'";
  if (system(cleanup.c_str()) != 0) {
    ok = false;
  }

  printf("%s\n", ok ? "PASSED!" : "FAILED!");
  return ok ? 0 : 1;
}
