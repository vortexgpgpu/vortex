#include <v80_image.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

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
  write_file(a, std::string(200000, 'x'));
  write_file(a2, std::string(200000, 'x'));
  write_file(b, std::string(199999, 'x') + "y");
  write_file(grown, std::string(200001, 'x'));
  const uint64_t da = v80_image_digest(a);
  ok &= check(da != 0, "digest of a readable file");
  ok &= check(da == v80_image_digest(a2), "same content, same digest");
  ok &= check(da != v80_image_digest(b), "one byte differs");
  ok &= check(da != v80_image_digest(grown), "one byte longer");
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

  mkdir((sysfs + "/0000:01:00.0").c_str(), 0755);
  image.pci_instance = rec.pci_instance();
  ok &= check(image.pci_instance != 0, "instance of a present card");
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

  // The card was enumerated again since the record was written.
  v80_image_t stale = image;
  stale.pci_instance = image.pci_instance + 1;
  ok &= check(rec.store(stale), "store for an earlier enumeration");
  ok &= check(!rec.holds(da, &got), "a record from an earlier enumeration is void");

  // Another card has its own record.
  mkdir((sysfs + "/0000:02:00.0").c_str(), 0755);
  ok &= check(rec.store(image), "store again");
  ok &= check(!v80_image_record("02:00", dir, sysfs).holds(da, &got), "per-card records");

  ok &= check(rec.clear(), "clear");
  ok &= check(!rec.holds(da, &got), "a cleared record holds nothing");
  ok &= check(rec.store(image) && rec.holds(da, &got), "store after clear");

  // Anything that is not exactly one record is no record.
  write_file(rec.path(), "vortex-v80-image 1 1234 5678 1 2 3\n");
  ok &= check(!rec.load(&got), "truncated record");
  write_file(rec.path(), "vortex-v80-image 2 1234 5678 1 2 3 4 5 6\n");
  ok &= check(!rec.load(&got), "unknown record version");
  write_file(rec.path(), "vortex-v80-image 1 1234 5678 1 2 3 4 5 6 7\n");
  ok &= check(!rec.load(&got), "trailing data");

  const std::string cleanup = "rm -rf '" + dir + "'";
  if (system(cleanup.c_str()) != 0) {
    ok = false;
  }

  printf("%s\n", ok ? "PASSED!" : "FAILED!");
  return ok ? 0 : 1;
}
