# Contributing to Vortex on Github

## Github Details
- There are two main repos, `vortex` (public, this one) and `vortex-dev` (private)
- todo: Most current development is on `vortex`
- If you have a legacy version of `vortex`, you can use the releases branch or tags to access the repo at that point in time

## Contribution Process
- You should create a new branch from develop that is clearly named with the feature that you want to add
- Avoid pushing directly to the `master` branch instead you will need to make a Pull Request (PR)
- There should be protections in place that prevent pushing directly to the main branch, but don't rely on it
- When you make a PR it will be tested against the continuous integration (ci) pipeline (see `continuous_integration.md`)
- It is not sufficient to just write some tests, they need to be incorporated into the ci pipeline to make sure they are run
- During a PR, you might receive feedback regarding your changes and you might need to make further commits to your branch


## Creating  and Adding Tests
see `testing.md`