# Changelog

## [0.1.1](https://github.com/QNSC-VN/qnsc-kb-backend/compare/v0.1.0...v0.1.1) (2026-08-25)


### ✨ Features

* adopt the shared QNSC deploy pipeline for develop ([#7](https://github.com/QNSC-VN/qnsc-kb-backend/issues/7)) ([5b9990d](https://github.com/QNSC-VN/qnsc-kb-backend/commit/5b9990de6854121a92894ce322cc8a20eed00425))
* **auth:** name the Entra administrators instead of provisioning everyone as Staff ([#29](https://github.com/QNSC-VN/qnsc-kb-backend/issues/29)) ([745e1aa](https://github.com/QNSC-VN/qnsc-kb-backend/commit/745e1aa10e6128abf5485126815f2b3fbd7b93f0))
* **ci:** add zizmor Actions-security scan job ([#66](https://github.com/QNSC-VN/qnsc-kb-backend/issues/66)) ([9adf340](https://github.com/QNSC-VN/qnsc-kb-backend/commit/9adf340cf4684c6fbcd7817d49b67d944b718eb3))
* **connectors:** enable the Microsoft connector in develop, and keep its webhooks alive ([6353ab4](https://github.com/QNSC-VN/qnsc-kb-backend/commit/6353ab4406be9579ee8775629851bb49b5f884fd))
* **embeddings:** drop the ml group from shipped images — torch is build-only ([#54](https://github.com/QNSC-VN/qnsc-kb-backend/issues/54)) ([c6976b8](https://github.com/QNSC-VN/qnsc-kb-backend/commit/c6976b8b3c4d0c48b5cb3ac27b0f3bb097010def))
* **embeddings:** run bge-m3 on ONNX Runtime in develop, one copy of the weights ([#51](https://github.com/QNSC-VN/qnsc-kb-backend/issues/51)) ([049d9fe](https://github.com/QNSC-VN/qnsc-kb-backend/commit/049d9fed931771850bbfeb2fd0f0d36220f33081))
* **infra:** add quangld@qnsc.vn and hieuvbm@qnsc.vn to entra_admin_emails ([#61](https://github.com/QNSC-VN/qnsc-kb-backend/issues/61)) ([c2474c6](https://github.com/QNSC-VN/qnsc-kb-backend/commit/c2474c631079ac3c754e1ec0b1ad8b1f896d77aa))
* **infra:** move qnsc-kb develop onto the shared Valkey node ([#47](https://github.com/QNSC-VN/qnsc-kb-backend/issues/47)) ([a6c5fc3](https://github.com/QNSC-VN/qnsc-kb-backend/commit/a6c5fc301a3953a8776772d89648c51804db3a30))
* **infra:** run develop on working hours only ([#46](https://github.com/QNSC-VN/qnsc-kb-backend/issues/46)) ([4bf6887](https://github.com/QNSC-VN/qnsc-kb-backend/commit/4bf688712ab8b7fb22ee8f0f7ac1b58ec0597698))
* init project ([2b912ac](https://github.com/QNSC-VN/qnsc-kb-backend/commit/2b912ac31354baaf30a147eb656159c4615a1a95))
* phase 1 ([72e4cfa](https://github.com/QNSC-VN/qnsc-kb-backend/commit/72e4cfa9ec835ab2ada601a4b46f7915b6babd25))
* phase2 ([4641bb2](https://github.com/QNSC-VN/qnsc-kb-backend/commit/4641bb20779062a1941120bf0e07eca89621948e))
* phase3 ([0227ad8](https://github.com/QNSC-VN/qnsc-kb-backend/commit/0227ad80fea2bb9be509263f1086e6811d4f569e))
* **scripts:** create the first global administrator ([ba1afe9](https://github.com/QNSC-VN/qnsc-kb-backend/commit/ba1afe9c1a24ba2bdcd8db15cd502cb36cf0932c))
* sso ([#17](https://github.com/QNSC-VN/qnsc-kb-backend/issues/17)) ([5aaa1f0](https://github.com/QNSC-VN/qnsc-kb-backend/commit/5aaa1f0802fc4c45e7b80c68f2fbbea9e1d69396))
* update phase 1 ([1f4eeed](https://github.com/QNSC-VN/qnsc-kb-backend/commit/1f4eeed671f45ccfd762430cd27f5b042517bbcc))


### 🐛 Bug Fixes

* **auth:** bootstrap admin on a fresh database hit MissingGreenlet and never booted ([#55](https://github.com/QNSC-VN/qnsc-kb-backend/issues/55)) ([45ce0ca](https://github.com/QNSC-VN/qnsc-kb-backend/commit/45ce0ca4e35ab1271b4e2a33e92db90bcbc7b08f))
* **auth:** give refresh tokens a jti so rotation cannot collide ([65f902c](https://github.com/QNSC-VN/qnsc-kb-backend/commit/65f902c28a134a7c96e33575e7890d2ad42e3402))
* **auth:** re-fetch instead of refresh, so user responses can read roles ([a69ce62](https://github.com/QNSC-VN/qnsc-kb-backend/commit/a69ce6272e90999e4790b74108cadf5153a7065d))
* **ci:** restore deploy permissions broken by the zizmor excessive-permissions fix ([#68](https://github.com/QNSC-VN/qnsc-kb-backend/issues/68)) ([a26fed1](https://github.com/QNSC-VN/qnsc-kb-backend/commit/a26fed154a24752a1711db9959822ff4ad3a2b72))
* **ci:** scope permissions, set persist-credentials, enforce zizmor ([#67](https://github.com/QNSC-VN/qnsc-kb-backend/issues/67)) ([faeaad3](https://github.com/QNSC-VN/qnsc-kb-backend/commit/faeaad3b48387e141cfe49829565268bfd01cb02))
* **ci:** stop dependabot updating terraform providers ([#39](https://github.com/QNSC-VN/qnsc-kb-backend/issues/39)) ([6c4e162](https://github.com/QNSC-VN/qnsc-kb-backend/commit/6c4e162d5a32e487f8e994f468f579d4d970c5f3))
* **codeowners:** point at the team that actually exists ([1756a72](https://github.com/QNSC-VN/qnsc-kb-backend/commit/1756a72a0685f43d407794e040736e5fc81d572e))
* **db:** keep the tenant context across commits, and make it unleakable ([#26](https://github.com/QNSC-VN/qnsc-kb-backend/issues/26)) ([d24e0bd](https://github.com/QNSC-VN/qnsc-kb-backend/commit/d24e0bdd3a7934a034f5fe94731e97a05d0f23d4))
* **db:** resolve the migrator's connection through Settings, not os.getenv ([957e35b](https://github.com/QNSC-VN/qnsc-kb-backend/commit/957e35b8c58f525fb60830895616b6b97a9e3840))
* **deps:** clear every dependency CVE the scanner blocks on ([#21](https://github.com/QNSC-VN/qnsc-kb-backend/issues/21)) ([e832761](https://github.com/QNSC-VN/qnsc-kb-backend/commit/e8327614dfdc54a731c8b54f60049d045d1e9878))
* **docker:** install the OCR group with --only main,ocr ([68ef6f5](https://github.com/QNSC-VN/qnsc-kb-backend/commit/68ef6f5b3f292cb69dd4c72830231a7c244d44c7))
* **docker:** put /app on PYTHONPATH so the migrator can import src ([0f27928](https://github.com/QNSC-VN/qnsc-kb-backend/commit/0f279284d7883dc7aeb8bcc9451b2c1b70d435ff))
* **embeddings:** align the pgvector column width with the configured model ([f33ce27](https://github.com/QNSC-VN/qnsc-kb-backend/commit/f33ce274b518c2b97ee1ccd646f61af6272a239d))
* **embeddings:** use gemini-embedding-001 at 768 dimensions, normalised ([fb92f69](https://github.com/QNSC-VN/qnsc-kb-backend/commit/fb92f69c6aff1d2cbf2fcf0ca3bf6ce380a99bc7))
* **infra:** bump rds to v2.1.1 — unbreak the develop apply ([#42](https://github.com/QNSC-VN/qnsc-kb-backend/issues/42)) ([971de00](https://github.com/QNSC-VN/qnsc-kb-backend/commit/971de0045b4e317bab9a728298523f043694e1f2))
* **infra:** clamav startPeriod back to 300, the ECS maximum ([#28](https://github.com/QNSC-VN/qnsc-kb-backend/issues/28)) ([76d54b7](https://github.com/QNSC-VN/qnsc-kb-backend/commit/76d54b70e02d3e8d430f7321d187b8b857e7a585))
* **infra:** disable the startup admin bootstrap in deployed environments ([fab0dfe](https://github.com/QNSC-VN/qnsc-kb-backend/commit/fab0dfec0af9ece2e97d2b26bb5aeae1421d0d23))
* **infra:** give clamav the 2 GB its signature database needs ([#27](https://github.com/QNSC-VN/qnsc-kb-backend/issues/27)) ([d2815aa](https://github.com/QNSC-VN/qnsc-kb-backend/commit/d2815aaa1c944556ff0fbc9a408a10d47a516373))
* **infra:** give the tunnel its ingress rules ([973635e](https://github.com/QNSC-VN/qnsc-kb-backend/commit/973635ef40df56bf3a73e48b9e3472906c0cf8ad))
* **infra:** point the embedding model at what the code and image actually use ([#24](https://github.com/QNSC-VN/qnsc-kb-backend/issues/24)) ([5eb4506](https://github.com/QNSC-VN/qnsc-kb-backend/commit/5eb4506f17d86c6c00e4c10a3ff38abbffcfa4f4))
* **infra:** restore qnsc-kb develop's wake schedule ([#56](https://github.com/QNSC-VN/qnsc-kb-backend/issues/56)) ([5526eb8](https://github.com/QNSC-VN/qnsc-kb-backend/commit/5526eb807753b7df9072c5937faf9c1976ba7b82))
* **infra:** set MICROSOFT_LOGIN_REDIRECT_URI, which develop needs to boot ([ebeca05](https://github.com/QNSC-VN/qnsc-kb-backend/commit/ebeca0513b743312a54ea78c2965629a1164acc4))
* **llm:** make Gemini selectable, so a Gemini-only deployment gets real answers ([918455f](https://github.com/QNSC-VN/qnsc-kb-backend/commit/918455ff053eaec98fce7e93f55a79216db520df))
* **llm:** replace the retired Gemini generation model with a latest alias ([7160263](https://github.com/QNSC-VN/qnsc-kb-backend/commit/7160263bbc9161c57f34dd8ba2ad71c67b784cee))
* **locking:** release the article advisory lock after a failed index ([270a12e](https://github.com/QNSC-VN/qnsc-kb-backend/commit/270a12e14879d52560c58ee313f88f4ee8ed3a92))
* **models:** drop the connectors column that no migration ever created ([#50](https://github.com/QNSC-VN/qnsc-kb-backend/issues/50)) ([eb44e16](https://github.com/QNSC-VN/qnsc-kb-backend/commit/eb44e160cc3e852d3e6319a9ec9d86b37199ab5d))
* **models:** fill timestamps Python-side so ORM inserts drop RETURNING ([87e7fd4](https://github.com/QNSC-VN/qnsc-kb-backend/commit/87e7fd4268966bf6b08cb98e5f1310f8e8d12c8b))
* **repositories:** re-fetch after update so callers can read relationships ([3f1e3ae](https://github.com/QNSC-VN/qnsc-kb-backend/commit/3f1e3aebf7f0b7e8124978359edcd326b898829c))
* stop the idle production database daily, not weekly ([#31](https://github.com/QNSC-VN/qnsc-kb-backend/issues/31)) ([100b2cf](https://github.com/QNSC-VN/qnsc-kb-backend/commit/100b2cf9ec5e476fc6f5742f81ef3719ce00b3a0))


### ⚡ Performance

* **ci:** adopt qnsc-ci v1.9.1 — parallel builds and registry cache ([#41](https://github.com/QNSC-VN/qnsc-kb-backend/issues/41)) ([0d0a7aa](https://github.com/QNSC-VN/qnsc-kb-backend/commit/0d0a7aabcc10b6cef9bcbc6fc3232e968de03e54))
* **ci:** stop holding a 10.4 GB build cache in ECR ([#49](https://github.com/QNSC-VN/qnsc-kb-backend/issues/49)) ([bc6e459](https://github.com/QNSC-VN/qnsc-kb-backend/commit/bc6e459a49a87c6470bd19d3084a4b1b897d4849))
* **docker:** put the app code above the expensive layers, not below ([0f1afd6](https://github.com/QNSC-VN/qnsc-kb-backend/commit/0f1afd6ed9694e7bc920d444187fa25db31a2d13))
* move embeddings to a hosted API and shrink every image and task ([80e333d](https://github.com/QNSC-VN/qnsc-kb-backend/commit/80e333d9598e2bb55a017447138fd98a0b8348a3))


### ♻️ Refactors

* **embeddings:** split runtime from model, add an ONNX backend ([#23](https://github.com/QNSC-VN/qnsc-kb-backend/issues/23)) ([f02877a](https://github.com/QNSC-VN/qnsc-kb-backend/commit/f02877af1476d85cd2be48ba4c7e73910fd2f466))


### 📦 Dependencies

* drop the numpy&lt;2 ceiling and refresh every group now that torch is gone ([#59](https://github.com/QNSC-VN/qnsc-kb-backend/issues/59)) ([341e18a](https://github.com/QNSC-VN/qnsc-kb-backend/commit/341e18af1ebb313aa2aa236a56a63177642d5c30))
