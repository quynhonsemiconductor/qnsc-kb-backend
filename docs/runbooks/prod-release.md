# Production release for kb (two tags, two promote PRs, one go-live window)

kb is the one product whose images come from **two repositories**: `qnsc-kb-backend` publishes
`kb-api` / `kb-worker` / `kb-migrator`, `qnsc-kb-frontend` publishes `kb-web`. Each repository has
its own Release Please stream, its own `vX.Y.Z` tags, and its own promote workflow
(`.github/workflows/release-promote.yml`); both write `values/kb/tags.prod.yaml` in `gitops` but
different keys of it (backend `services.{api,worker,migrator}`, frontend `web.image`). A tag
**rebuilds nothing**: it retags the digest already published for that commit and opens a reviewed
`gitops` PR. Merging that PR is the production gate.

This runbook is the go-live plan for the promote gate variable and the two release cuts. It assumes
the rest of the launch checklist (`sa-notes/reviews/kb-prod-readiness-2026-11-05.md`) is already
green: `gitops#255` merged, the `kb-prod` data layer live, the two Entra apps applied, and the ten
`kb-prod-*` secrets in `kv-qnsc-prod`.

## 0. The gate variable (owner, at cutover)

`KB_PROD_PROMOTE_ENABLED` is a **repository** variable, not an org one: the gate must be armable
per product, and kb is the only product that promotes its SPA from a second repository. While it is
unset, a release tag runs nothing (`if: vars.KB_PROD_PROMOTE_ENABLED == 'true'`); a clean skip, not
a red run.

```sh
gh variable set KB_PROD_PROMOTE_ENABLED --body true -R quynhonsemiconductor/qnsc-kb-backend
gh variable set KB_PROD_PROMOTE_ENABLED --body true -R quynhonsemiconductor/qnsc-kb-frontend
gh variable list -R quynhonsemiconductor/qnsc-kb-backend | grep KB_PROD_PROMOTE_ENABLED
gh variable list -R quynhonsemiconductor/qnsc-kb-frontend | grep KB_PROD_PROMOTE_ENABLED
```

Set it **in the go-live window, and only after all three are true**:

1. `values/kb/{prod.yaml,prod.secrets.yaml,tags.prod.yaml}` exist on `gitops` main (`gitops#255`).
   Before that the reusable stops with its own red `values/kb/tags.prod.yaml does not exist. Prod
   launch is phase P3` — an armed tag would turn into a failed run, not a PR.
2. The `kb-prod` namespace and `pg-kb` exist and the migrator's first run is understood: the promote
   PR is the *deploy* approval, and ArgoCD applies it to a live namespace.
3. The placeholders `<KB_PROD_CLIENT_ID>` / `<KB_PROD_SERVICE_APP_ID>` are replaced
   (`scripts/check-prod-placeholders.sh` passes in the PR that enables the appset row).

Unsetting the variable again is a clean disarm and never touches a PR already opened.

## 1. Cut the two releases in the same window

Merge the Release Please PR in each repository. Versions are independent (`qnsc-kb-backend` is on
`0.1.x`, `qnsc-kb-frontend` on `0.1.x` with its own sequence); what has to match is the **window**,
not the number. Both release PRs should be merged one after the other, and each must be merged from
a commit that GHCR · Publish has built — the promote workflow waits for that run and then retags
`sha-<commit>`.

Each merge is a push to `main`, so GHCR · Publish starts building the release commit; Release Please
then pushes the `vX.Y.Z` tag with its App token, which is what triggers Release · Promote (a tag
pushed with `GITHUB_TOKEN` would not trigger it — that is why both repos pass
`QNSC_AUTOMATION_PRIVATE_KEY`). The promote run's `wait-for-publish` job waits for the same
commit's publish run, up to 45 minutes.

## 2. What each promote PR carries

| Repository | Tag | `services:` | Images retagged | `tags.prod.yaml` keys |
|---|---|---|---|---|
| qnsc-kb-backend | its own `vX.Y.Z` | `api worker migrator` | `kb-api`, `kb-worker`, `kb-migrator` at the `sha-<commit>` digest of that tag's commit | `services.api`, `services.worker`, `services.migrator` |
| qnsc-kb-frontend | its own `vX.Y.Z` | `web` | `kb-web` at the `sha-<commit>` digest of that tag's commit | `web.image` |

Neither list can be partial: `release-retag` fails before writing any tag if one listed image is
not published for that commit, and it verifies each digest's build attestation against the source
repository and `refs/heads/main`. The two runs take different promotion branches
(`promote/kb-<source-repo>-<tag>`, ci v1.34.1) so they cannot collide or exit green without
promoting — this is why the four kb workflows pin the promote reusable at v1.34.1 and not v1.33.1.

`web` is deliberately **not** in the backend's list: the SPA image is built from the frontend's own
commit and a backend commit never published a `kb-web` digest.

## 3. Review and merge the two promote PRs (owner)

Both PRs are titled `chore(kb): promote prod to vX.Y.Z` (same title, different repositories of
origin). Before merging, check in each PR:

- The `## Digests` block lists exactly the services in the table above, with `sha256:` digests.
- The digests equal what GHCR holds for the release commit, for example:
  `gh api "orgs/quynhonsemiconductor/packages/container/kb-api/versions?per_page=100" --jq '.[] | select(.metadata.container.tags[]? == "sha-<commit>") | .name'`.
- The diff touches only `values/kb/tags.prod.yaml` (and `rendered/kb/prod.yaml` once the appset row
  is enabled — see below), and every `image.tag` it writes is a `vX.Y.Z`, never a `sha-`.

Merge order: **backend first, then frontend.** Once the appset row for kb prod is enabled,
`values/kb/tags.prod.yaml` has a companion `rendered/kb/prod.yaml`, and **both** promote runs rewrite
that render (each from the `main` it checked out). Merge one PR, then update the other's branch
(`gh pr update-branch <n> -R quynhonsemiconductor/gitops` or let the stale-render gate ask for it)
and re-check its render before merging. Merging both without the refresh can leave `main` with a
render that does not match its values.

After both merge, `values/kb/tags.prod.yaml` on `gitops` main must show four images, all on release
tags:

```sh
gh api repos/quynhonsemiconductor/gitops/contents/values/kb/tags.prod.yaml \
  --jq .content | base64 -d | grep -E 'api:|worker:|migrator:|web:|tag:|digest:'
```

If any of them is still `sha-<commit>` (the hand-seeded provisional digests in `gitops#255`), prod
is not on the release: that seed is only a placeholder that lets the chart render before the first
promotion, and **never** a go-live state.

## 4. Charts and prod values (what breaks a later bump)

`gitops#255` pins the kb prod row at chart **0.19.1**. Chart **0.19.2** made the data-egress
NetworkPolicy select the cache by pod label; from 0.19.2 on, rendering kb prod **fails** while
`values/kb/prod.yaml` has no `data.cache.instance`:

```
a Redis/Valkey URL is configured (REDIS_URL or VALKEY_URL) but data.cache.instance is unset:
allow-egress-data would render no 6379 rule and the app could not reach its cache.
```

The fix is one value (the prod Valkey's pod label, `platform/kb-prod/valkey-kb.yaml`):

```yaml
data:
  postgres: { ... }
  cache:
    instance: valkey-kb-prod
```

Verified 2026-11-05: with that added, chart 0.19.3 renders (four digests, the 6379 rule, and
`APP_VERSION=vX.Y.Z` on api/worker; release tags name no commit, so `APP_COMMIT` is absent by the
chart's own design). Until the value is in, keep the row on 0.19.1.

Two more prod values worth carrying from dev while `prod.yaml` is open: `DEPLOYMENT_ENV: prod`
(gitops#260 added `DEPLOYMENT_ENV: dev`; the app's OTel label reads it first and falls back to the
chart's `deployment.environment`, so prod is labelled correctly either way — but the two files
should not drift), and `APP_COMMIT_SHA` is not needed: this build reads the commit from
`APP_COMMIT` when the image tag is `sha-<commit>`, and reports "unknown" otherwise.

## 5. Failure modes and what they mean

| Symptom | Cause | Fix |
|---|---|---|
| Promote run skipped, no jobs | `KB_PROD_PROMOTE_ENABLED` unset in **that** repository | set the variable (step 0); the gate is per repo |
| `values/kb/tags.prod.yaml does not exist` | `gitops#255` not merged | merge it, then push the tag again (`gh run rerun` re-runs the promote for the same tag) |
| `kb-<svc>:sha-<commit> is not published for this commit` | The tag's commit has no GHCR · Publish run (a recovered/bootstrapped tag, or a release PR whose merge only touched Markdown and was path-ignored) | Cut the next release from a `main` commit that published; a release tag is never moved |
| `promote/kb-…` exists on gitops but no PR | An earlier run failed after pushing the branch | Delete the branch, re-run: a promotion branch is never repointed |
| `prod is already at vX.Y.Z; nothing to promote` | The digest and tag are already in `tags.prod.yaml` | Nothing to do; the previous promotion merged |

The recovered tags from the 2026-11-05 Release Please bootstrap (`v0.1.1`/`v0.1.2` backend,
`v0.1.1` frontend) are **not** go-live releases. They were created with the gate unset, and the
frontend's `v0.1.1` commit (`d897164`, 2026-08-25) has no `kb-web` image on GHCR at all — it would
fail at retag if it were ever re-run. The prod release is the next release PR merged after step 0.

## 6. Rollback

Revert the promotion PR (`git revert` on `gitops` main, or `gh pr revert`): prod points back at the
previous digests, ArgoCD syncs that, and nothing is rebuilt. Never delete a release tag and never
move one — `release-retag` refuses to repoint a published `vX.Y.Z`, by design.
