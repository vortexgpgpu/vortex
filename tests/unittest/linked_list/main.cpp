#include <linked_list.h>
#include <stdio.h>
#include <vector>

using namespace vortex;

struct Node {
  int value;
  LinkedListNode<Node> hook;
};

using List = LinkedList<Node, &Node::hook>;

static bool expect(const List& list, const std::vector<int>& want, const char* what) {
  std::vector<int> got;
  for (auto& n : list) {
    got.push_back(n.value);
  }
  if (got == want)
    return true;
  printf("Error: %s: got {", what);
  for (size_t i = 0; i < got.size(); ++i) printf("%s%d", i ? ", " : "", got[i]);
  printf("}, expected {");
  for (size_t i = 0; i < want.size(); ++i) printf("%s%d", i ? ", " : "", want[i]);
  printf("}\n");
  return false;
}

int main() {
  Node n[6] = {{1, {}}, {2, {}}, {3, {}}, {4, {}}, {5, {}}, {6, {}}};
  List list;
  bool ok = true;

  // The simulator's timing wheel delivers same-cycle events in push_back order.
  list.push_back(&n[0]);
  list.push_back(&n[1]);
  list.push_back(&n[2]);
  ok &= expect(list, {1, 2, 3}, "push_back");

  list.push_front(&n[3]);
  ok &= expect(list, {4, 1, 2, 3}, "push_front");

  list.insert(++list.begin(), &n[4]);
  ok &= expect(list, {4, 5, 1, 2, 3}, "insert before position");

  list.erase(list.begin());
  list.pop_back();
  ok &= expect(list, {5, 1, 2}, "erase/pop_back");

  list.push_back(&n[5]);
  ok &= expect(list, {5, 1, 2, 6}, "push_back after erase");

  printf("%s\n", ok ? "PASSED!" : "FAILED!");
  return ok ? 0 : 1;
}
