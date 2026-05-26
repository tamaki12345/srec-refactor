---
description: "srec-refactorの実験運用ルール。比較実験のタイムアウト、ログ分離、copilot配下への履歴集約、日本語コミット方針を適用する。"
applyTo: "**/*"
---

# 実験運用ルール

- 比較実験は `QUICK_RUN=1` と `QUICK_TIMEOUT_SEC=60` を既定とし、長時間実行を避ける。
- 実験ログは実行ごとに分割し、履歴文書は `copilot/` 配下へ集約する。
- 実装作業は小さく区切って都度コミットし、コミットメッセージは日本語にする。
- 変更後は主要な結果（pair/s、比率、有効化された最適化の有無）を `copilot/implementation_log.md` に追記する。
