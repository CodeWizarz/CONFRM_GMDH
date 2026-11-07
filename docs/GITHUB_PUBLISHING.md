# GitHub handoff after author review

The local Git repository has three reviewable commits on `main` and an `origin`
remote. It has not been pushed. GitHub currently reports the destination
repository as public. Do not push there while double-blind review or author
metadata is unresolved. After completing `CITATION.cff`, choosing a license,
and deciding the journal's anonymous review route, first use the approved
private or anonymized destination. Then run from this directory:

```sh
git status --short
gh auth status
git push -u origin main
```

The push command publishes the existing three-commit history only when the
author chooses to run it. If the journal requires an anonymous reviewer link,
follow its permitted repository anonymization process; do not expose author
identity or private data in a public repository.
