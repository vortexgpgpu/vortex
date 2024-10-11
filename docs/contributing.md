# Contributing to Vortex

## Github
Vortex uses Github to host its git repositories.
There are a lot of ways to use the features on Github for collaboration.
Therefore, this documentation details the standard procedure for contributing to Vortex.
Development of Vortex is consolidated to this repo, `vortex` and any associated forks.
Previously, there was active work done on a private repo named `vortex-dev`.
`vortex-dev` has officially been deprecated and fully merged into this public repo, `vortex`.
If you are returning to this project and have legacy versions of Vortex, you can use the releases branches to access older versions.

## Contribution Process
In an effort to keep `vortex` organized, permissions to directly create branches and push code has been limited to admins.
However, contributions are strongly encouraged and keep the project moving forward! Here is the procedure for contributing:

1. Create a fork of `vortex`
2. In your fork, create a branch from `master` that briefly explains the work you are adding (ie: `develop-documentation`)
3. Make your changes on the new branch in your fork. You may create as many commits as you need, which might be common if you are making multiple iterations
4. Since you are the owner of your fork, you have full permissions to push commits to your fork
4. When you are satisfied with the changes on your fork, you can open a PR from your fork using the online interface
5. If you recently made a push, you will get automatically get a prompt on Github online to create a PR, which you can press
6. Otherwise, you can go to your fork on Github online and manually create a PR (todo)
(todo): how to name and format your PR, what information you should add to the PR, does not need to be too strict if you are attending the weekly meetings*
7. Github uses the following semantics: `base repository` gets the changes from your `head repository`
8. Therefore, you should set the `base repository` to `vortexgpgpu/vortex` and the `base` branch to `master` since the master branch is protected by reviewed PRs.
9. And you should assign the `head repository` to `<your-github-username>/vortex` (which represents your fork of vortex) and the `base` branch to the one created in step 2
10. Now that your intended PR has been specified, you should review the status. Check for merge conflicts, if all your commits are present, and all the modified files make sense
11. You can still make a PR if there are issues in step 10, just make sure the structure is correct according to steps 7-9
12. Once the PR is made, the CI pipeline will run automatically, testing your changes
13. Remember, a PR is flexible if you need to make changes to the code you can go back to your branch of the fork to commit and push any updates
14. As long as the `head repository`'s `base` branch is the one you edited, the PR will automatically get the most recent changes
15. When all merge conflicts are resolved, changes are made, and tests pass you can have an admin merge your PR

## What Makes a Good Contribution?
- If you are contributing code changes, then review [testing.md](./testing.md) to ensure your tests are integrated into the [CI pipeline](continuous_integration.md)
- During a PR, you should consider the advice you are provided by your reviewers. Remember you keep adding commits to an open PR!
- If your change aims to fix an issue opened on Github, please tag that issue in the PR itself