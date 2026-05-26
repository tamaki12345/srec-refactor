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
