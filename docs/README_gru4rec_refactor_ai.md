# GRU4Rec Refactor: AI実装支援用サマリ

## このリポジトリの目的
- セッションベース推薦アルゴリズムGRU4RecのTheano実装を、PyTorch等の現代的なフレームワークで再現・高速化・保守性向上する。
- Theano実装との厳密な挙動一致・速度比較・最適化を通じて、現代GPU環境での最大性能を目指す。

## 主要な現状・課題
### 1. TheanoとPyTorchの速度差
- Theanoは全体グラフ最適化・一括カーネル化で圧倒的に高速。
- PyTorchは部分的なグラフ化（torch.compile, CUDA Graphs）やバッチ固定長化で改善できるが、まだ20倍以上の差が残る。

### 2. 主なボトルネック
- Pythonループ主導の細粒度ステップ実行（tail含む）
- ステップごとのCPU→GPUインデックス転送
- 行単位Adagrad更新のscatter/gather多発
- full-size step以外の非グラフ化
- PyTorchのグラフ最適化の限界

### 3. Theanoではなぜ速いか
- 全体グラフ最適化でPython制御・分岐を排除
- 全区間一括カーネル化
- scatter/gatherもグラフ内で最適化

### 4. PyTorchでの解決策
- 全step固定長化＋全区間CUDA Graphs化（実装難度高）
- カスタムCUDAカーネルやsparse optimizer導入
- ただし現状はTheano/JAX型の全体最適化には及ばない

### 5. 他技術の選択肢
- JAXやTensorFlow XLAはTheano型の全体グラフ最適化＋現代的な保守性を両立できる
- 長期的にはJAX等への移行が最も現実的

## 今後AIに実装させる際の指針
- 厳密比較時は `fixed_candidate_size: false` でTheanoと完全一致を目指す
- 速度最適化時は `fixed_candidate_size: true` ＋全区間グラフ化を目指す
- tailも含めて全stepを完全固定長化し、全区間をCUDA Graphsで回す実装がPyTorchでの最速案
- さらなる高速化・保守性・将来性を重視するならJAX等への移行も検討する

## 参考: 直近の比較結果（quick run例）
| Run | pair/s |
|---|---:|
| Torch strict parity | 1687.14 |
| Torch optimized | 3996.91 |
| Theano GPU | 85559.82 |

---
このファイルはAIによる自動実装・最適化のための設計・現状・課題・方針の共有用です。

## 実装ロードマップ（refactor_non_torchブランチ向け）

### 全体方針（結論）
- **短期（今すぐの実験最適化）**: PyTorch実装を継続し、固定長化・グラフ化・転送削減を優先して改善する
- **中長期（最大性能と保守性）**: JAX/XLAでPoCを作り、Theano相当の全体最適化に段階移行する
- **比較の軸**: 1) 厳密一致（Theanoとの挙動） 2) 速度（pair/s） 3) 実装保守コスト

### フェーズ分割

#### Phase 0: ベースライン固定（1-2日）
- 目的: 比較の再現性を担保する
- 実施:
	- `experiment/m4a_torch_theano_strict.yml`, `experiment/m4a_torch_optimized.yml`, `experiment/m4a_theano.yml` の3本を同一データ・同一指標で計測
	- `scripts/run_m4a_torch_theano_compare.sh` でログと要約を保存
	- 実験時のGPU/CPU条件、CUDAバージョン、環境情報を必ず記録
- 完了条件:
	- strict / optimized / theano の `pair/s` を同一フォーマットで再取得できる

#### Phase 1: PyTorch高速化の本命（3-7日）
- 目的: Pythonオーバーヘッドと転送コストを下げ、Theanoとの差を最小化する
- 実施:
	- tailを含む**全step固定長化**（可変長分岐を削減）
	- full-size step以外も含めた**全区間CUDA Graphs化**
	- stepごとのCPU→GPU転送を抑えるため、インデックスバッファ再利用を導入
	- scatter/gather更新が集中する箇所をプロファイルしてホットスポットを特定
- 完了条件:
	- optimized構成で `pair/s` が明確に改善し、かつ strict parity 指標を悪化させない

#### Phase 2: PyTorch延命オプション（必要時のみ、3-10日）
- 目的: Phase 1で頭打ちの場合の追加施策
- 実施:
	- `torch.compile` の適用範囲拡大
	- optimizer更新経路の見直し（可能ならsparse系・カーネル集約）
	- 必要であればカスタムCUDAカーネルを限定導入
- 完了条件:
	- 実装複雑化に見合う速度改善があること

#### Phase 3: JAX PoC（5-10日）
- 目的: 中長期の本命技術として成立するかを早期判定
- 実施:
	- GRU4Rec最小経路（forward + loss + update）をJAXで再実装
	- strict parity条件に寄せた比較実験を最小データで実施
	- XLA最適化での実効 `pair/s` を測定
- Go/No-Go判定:
	- Go: PyTorch optimized比で有意な高速化 + 保守可能なコード構造
	- No-Go: 性能優位が小さい、または実装コスト過大

### 実装時の設定ガイド
- 厳密一致優先: `fixed_candidate_size: false`
- 速度優先: `fixed_candidate_size: true` + 全step固定長 + 全区間グラフ化
- 比較時は、データ分割・seed・評価指標・batch条件を固定する

### コミット運用ルール（このブランチでの推奨）
- 作業は必ず「1目的1コミット」で分割する
- コミットメッセージは日本語で、以下形式を推奨:
	- `docs: GRU4Rec最適化ロードマップを追記`
	- `scripts: 比較実験の一括実行スクリプトを追加`
	- `gru4rec_torch: 全step固定長化の前処理を追加`
- 1コミットで混ぜない:
	- ドキュメント変更
	- 実験設定変更
	- 学習ロジック変更
	- 計測/可視化変更

### 直近の実装順（AI作業キュー）
1. ドキュメントへ本ロードマップ追記（完了）
2. 比較実験の実行性を高めるスクリプト/要約自動化を追加
3. PyTorchの全step固定長化に向けた前処理経路を実装
4. CUDA Graphs適用範囲をtail含めて拡張
5. ボトルネック再計測（pair/s + profiler）
6. JAX PoCブランチを切って最小再実装
