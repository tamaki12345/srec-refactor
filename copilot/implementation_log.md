# 実装・進捗履歴ログ

## 2026-05-26 (ボトルネック再計測・3回平均)

### 比較実験（RUNS=3, 1分タイムアウト）
- Torch strict: 1482.67 pair/s
- Torch optimized: 3443.68 pair/s
- Theano: 67566.78 pair/s
- optimized/strict: 2.32倍
- Theano比: 5.1%

#### 各回のpair/s
| Run | Torch strict | Torch optimized | Theano |
|---|---:|---:|---:|
| 1 | 1481.31 | 3496.64 | 68774.26 |
| 2 | 1484.92 | 3415.34 | 64403.41 |
| 3 | 1481.77 | 3419.06 | 69522.68 |

#### プロファイラ出力（抜粋: m4a_torch_optimized_20260526_170226.log）
- 1stepあたり: 約3400~3500 pair/s
- CUDA Graphs, torch.compile, tail graph すべて有効化を確認
- 依然としてforward以外（データ準備・optimizer step等）のオーバーヘッドが支配的

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

## 2026-05-26 (GRU4RecTorch NaN修正)

### 変更内容
- `algorithms/gru4rec/gru4rec_torch.py` の既定 `loss` を `cross-entropy` から `bpr-max` に修正（Theano実装互換）。
- 学習時NaN発生後に評価で無条件 `Exception` を投げる箇所を、原因が分かる `RuntimeError` メッセージへ変更。

### 再現手順
- 実行コマンド: `uv run python run_config.py conf/example_next_torch.yml`
- 条件: `gru4rec_torch` / `n_epochs=3` / `batch_size=256` / `device=cuda`

### 主要結果
- 学習: Epoch0-2でNaNなし（loss: 0.645770 -> 0.546870 -> 0.490983）
- 評価: 正常完了（HitRate@20: 0.5426, MRR@20: 0.2703）
- 学習速度（概算）: 約22733 pair/s（52484 pair / 2.3087 s）
- 比率: N/A（単独実行のため strict/optimized 比較なし）
- 有効化最適化: `torch.compile=False`, `use_cuda_graphs=False`（本設定では未有効）

## 2026-05-26 (ボトルネック再計測・3回平均・新環境)

### 比較実験（RUNS=3, 1分タイムアウト）
- Torch strict: 5846.98 pair/s
- Torch optimized: 16805.80 pair/s
- Theano: 83213.57 pair/s
- optimized/strict: 2.87倍
- Theano比: 20.2%

#### 各回のpair/s
| Run | Torch strict | Torch optimized | Theano |
|---|---:|---:|---:|
| 1 | 5862.24 | 16852.26 | 86075.81 |
| 2 | 5878.06 | 16802.31 | 83037.26 |
| 3 | 5800.65 | 16762.82 | 80527.64 |

#### 実行条件・出力
- コマンド: `RUNS=3 QUICK_RUN=1 QUICK_TIMEOUT_SEC=60 USE_GPU=0 bash scripts/run_m4a_compare_batch.sh`
- 集計: `results/compare/aggregate_20260526_224944.md`
- CSV: `results/compare/aggregate_20260526_224944.csv`

#### 有効化された最適化（optimized設定）
- `torch.compile=True`（`compile_backend=aot_eager`, `compile_mode=reduce-overhead`）
- `use_cuda_graphs=True`（full-size graph + tail forward graph）

## 2026-05-27 (Theano環境再修正: BLAS設定と依存補完)

### 変更内容
- `scripts/setup_theano_env.sh` の `THEANO_FLAGS_BASE` に `blas.ldflags=` を追加し、Theano起動時の `blas_opt_info` 参照例外を回避。
- 同スクリプトに `scikit-optimize` と `dill` の導入を追加（`run_config.py` の必須importを満たすため）。

### 再現手順
- 実行コマンド: `bash scripts/setup_theano_env.sh`
- 追加検証コマンド:
    - `THEANO_FLAGS="device=cuda0,floatX=float32,dnn.enabled=False,force_device=True,blas.ldflags=,print_active_device=True" python - <<'PY' ... PY`
    - `QUICK_RUN=1 QUICK_TIMEOUT_SEC=60 THEANO_FLAGS="device=cuda0,floatX=float32,dnn.enabled=False,force_device=True,blas.ldflags=" timeout 120s python run_config.py experiment/m4a_theano.yml`

### 主要結果
- BLAS例外（`No section: 'blas'` / `numpy.distutils.__config__.blas_opt_info`）は解消。
- `run_config.py experiment/m4a_theano.yml` はデータロードと学習開始まで到達（`~53k pair/s` を観測）。
- 一方で `pygpu` 初期化時に `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` 警告は継続（ただし実行は継続）。
- 比率: N/A（今回の確認はTheano単独起動の復旧確認）
- 有効化最適化: `force_device=True`, `blas.ldflags=`（BLAS自動検出回避）

## 2026-05-27 (Theano GPU実利用の復旧)

### 原因
- `/usr/local/cuda` が CUDA 12.5 を指しており、driver 535 系との組み合わせで `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` が発生。
- その結果、Theano は `pygpu` 初期化に失敗してCPU演算へフォールバックしていた。

### 変更内容
- `scripts/setup_theano_env.sh`
    - 既存envの再利用を追加（再実行時に失敗しないよう改善）。
    - `cudatoolkit=11.8` を conda env に導入。
    - `LD_LIBRARY_PATH` を `${CONDA_PREFIX}/lib` 優先に変更し、`libnvrtc.so.11.8` を優先利用。
- `scripts/run_m4a_torch_theano_compare.sh`
    - Theano実行envを `THEANO_ENV`（既定: `srec_theano_gpu`）で指定可能に変更。
    - GPU実行時の `THEANO_FLAGS` に `force_device=True,blas.ldflags=,dnn.enabled=False` を設定。
    - `LD_LIBRARY_PATH` を `${CONDA_PREFIX}/lib` 優先に変更。

### 検証
- GPU演算子確認コマンドで `GpuDot22` を確認（`has_gpu_op: True`）。
- 比較実験（`QUICK_RUN=1 QUICK_TIMEOUT_SEC=60 USE_GPU=1 THEANO_ENV=srec_theano_gpu bash scripts/run_m4a_torch_theano_compare.sh`）で Theano 実行時に
    - `Mapped name None to device cuda0: NVIDIA GeForce RTX 3090` を確認
    - `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` は非再発

### 主要結果（summary_20260527_003838.md）
- Torch strict: 5871.58 pair/s
- Torch optimized: 16814.41 pair/s
- Theano (GPU): 167711.69 pair/s
- optimized/strict: 2.86倍
- Theano比（optimized/Theano）: 10.0%
