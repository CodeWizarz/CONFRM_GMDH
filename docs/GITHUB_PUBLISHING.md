# GitHub handoff after author review

The local Git repository has three reviewable commits on `main` and an `origin`
remote. The destination GitHub repository is public. The author has authorized
publishing to it; the push publishes the source files, supplied result artifacts,
and the GitHub account's identity. Confirm the journal's double-blind repository
rules before reusing the public link in a manuscript or response letter.

```sh
git status --short
gh auth status
git push -u origin main
```

The local history contains separate provenance, package, and documentation
commits, followed by the citation and license commit.
