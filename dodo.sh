# Show the commit to be sure
git show --name-only cc372a94cd9cd61155c3560acdfe1deb0cfe1cd1

# Restore the chosen files from the *parent* of X (i.e., before X)
git restore --source=cc372a94cd9cd61155c3560acdfe1deb0cfe1cd1^ -- \
  sim/simx/core.cpp sim/simx/decode.cpp sim/simx/emulator.cpp sim/simx/emulator.h \
  sim/simx/execute.cpp sim/simx/func_unit.cpp sim/simx/types.h
# (on older Git: use `git checkout <hash>^ -- <paths>`)

# Commit and push
git add -A
git commit -m "Revert selected files to their state before cc372a94"
git push
