# 実装・進捗履歴ログ

## 2026-05-26

### 3. PyTorchの全step固定長化に向けた前処理経路を実装
- fixed_candidate_size有効時、tailも含めてx/hiddenをactive_batch長でpaddingし、損失計算は実データのみ反映
- コミット: 900c945
- 検証: 1分タイムアウトでstrict/optimized/theano全て正常動作

### 4. CUDA Graphs適用範囲をtail含めて拡張
- tailステップ用forward専用CUDA Graphsを追加、fixed_candidate_size時に自動適用
- コミット: f2527f2
- 検証: ログに `Enabled CUDA Graphs for tail forward steps` を確認

### 1分タイムアウトでの再計測（RUNS=1）
- Torch strict: 1501.45 pair/s
- Torch optimized: 3512.74 pair/s
- Theano: 67386.61 pair/s
- optimized/strict: 2.3396

---

## 実験結果と現状の考察（2026-05-26）

- 1分タイムアウトでの再計測（RUNS=1）
    - Torch strict: 1501.45 pair/s
    - Torch optimized: 3512.74 pair/s
    - Theano: 67386.61 pair/s
    - optimized/strict: 2.34倍（前回2.32倍）
    - Theano比: 5.2%（前回5.3%）

**改善幅が小さい/頭打ちの要因分析**
- Pythonループ自体のオーバーヘッドや、scatter/gatherを含むoptimizer更新の細粒度性が依然としてボトルネック
- CUDA Graphs適用範囲をtailまで拡張しても、PyTorchの設計上「全区間一括グラフ化」や「全体カーネル化」には至らず、Theano型の高速化には根本的に及ばない
- 転送・バッファ再利用・固定長化・Graph化など「PyTorchでできる範囲」はほぼやり切った状態
- これ以上の大幅な改善には、optimizerのカスタムカーネル化や、Python制御自体の排除（JAX/XLA型）など、より抜本的なアプローチが必要

**残作業の割合・優先度**
- PyTorch最適化で得られる速度向上は残りごく僅か（1〜2割未満）
- 残りの大きな改善余地は「JAX PoC」や「optimizerの抜本的刷新」など、現状未着手の領域に集中
- したがって、今後の主軸は「JAX PoCブランチでの全体グラフ最適化」や「Theano型の全体カーネル化」に移行すべき段階
