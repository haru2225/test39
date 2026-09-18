# test39: 周期セル内の完全ノイズから生成する粘土CG実験

`test39.py`はtest38の時間条件付きNequIP・CG種別・データ形式(`test38.py`をそのまま
同梱)を再利用しますが、目的が異なります。test38は「参照構造に少しノイズを
加えたところから復元する」局所デノイザーです。test39は前向き過程を周期セル上の
ブラウン運動にして、ノイズが十分大きいときの終端分布をセル内の一様分布にします。
生成は同じセル内の一様ランダム配置から逆SDEを積分します。test38のチェックポイント
は学習目標が違うため、test39の生成には使えません。

## Si・Alの周囲の酸素を省く粗視化(1292→396粒子)

学習前に、既定でジョブ内から粘土の酸素(ob, obos, oh, ohs)とOHの水素(ho)を
取り除く粗視化を実行します(`CG_MODE=cations`、既定)。Si(st)・Al(ao)・
Mg(mgo)・Na・Caの中心位置だけを残し、同梱の1292粒子×10,000フレームデータから
396粒子のデータセットを作ります。これは中心原子の座標を残す粗視化であり、
SiO4/AlO6の重心への置換や共有酸素の質量配分は行いません。省いた酸素の位置は
このモデルから復元できません。

酸素だけを除いてOHの水素を残す場合は524粒子になります(`CG_MODE=oxygen-only`)。
酸素を残す従来の1292粒子モデルは`CG_MODE=none`です。396粒子・524粒子モデルは
1292粒子のチェックポイントから再開できません。

## 学習品質の改善オプション(すべてオプトイン、既定は変更なし)

学習が進んでも生成が収束しない(ランダムな配置のまま変化しない)場合の
診断で見つかった問題への対策を、**すべて既定オフの追加フラグ**として用意しています。
既に走っているジョブは何も指定しなければ`--resume`にそのまま使えます。

- **`--num-neighbors auto`**: `architecture()`内の近傍数の正規化定数は既定で
  `12`固定ですが、これは酸素を含む1292粒子系(test38)向けの値です。396粒子の
  カチオンのみモデルでは実際の10Å以内の平均近傍数は約73と大きく異なり、
  メッセージパッシングのスケールがずれたまま学習することになります。
  `auto`を指定するとデータセットの最初のフレームから実際の平均近傍数を
  計算して使います。数値を直接指定することもできます。
- **`--sigma-sampling log-normal`**: 生成を完全ランダムから始めるにはsigma_max
  (セル最長辺)が本当に必要ですが、既定の対数一様サンプリングだと
  sigma_minからsigma_maxまでの3桁近い範囲に学習の勾配更新が均等に散らばり、
  実際の粒子間距離(≈3Å)付近という「構造形成に一番効くが一番難しい」
  帯域への配分が薄くなります。`log-normal`にすると、sigma_maxはそのまま
  維持しつつ(終端分布が一様になる保証は変わらない)、学習時にサンプルする
  sigmaの分布だけをデータの典型スケール付近に集中させます。中心は既定で
  データセットの最近接距離の中央値を自動推定しますが、`--sigma-log-mean`/
  `--sigma-log-std`で手動指定もできます。
- **`--validation-frames`(既定8)**: `training.json`の`validation_mse`
  (small/middle/terminal)は元々検証フレーム1個だけの評価値でノイズが
  大きかったため、既定で複数フレームの平均に変更しました。こちらは
  チェックポイントの`settings`に影響しないため、`--resume`の互換性には
  影響しません。
- **`--irreps-hidden`/`--num-convs`**: `architecture()`の既定値は現在
  `irreps_hidden="128x0e + 64x1e + 32x2e + 16x3e + 8x4e + 4x5e"`(l=5まで
  拡張)・`num_convs=3`です(旧: `64x0e + 32x1e`・3層)。この粘土鉱物の
  主要な配位構造であるSi四面体(Td対称性、最初の非自明な多重極項はl=3)
  とAl/Mg八面体(Oh対称性、最初の非自明な多重極項はl=4)を、エッジの
  球面調和展開だけで直接表現できるようにするための拡張です。これらの
  フラグはこの既定値をさらに上書きしたい場合に使います。
- **`--gradient-checkpointing`(既定on)**: 上記の容量増強・l=5拡張により
  GPUメモリ使用量が大きく増え、CUDA out of memoryが発生することを確認
  しました。各Interaction+Gate層の順伝播結果をメモリに保持せず、
  backward時に再計算することでピークメモリを削減します(再計算コストは
  増えますが、モデルの出力・精度は完全に同一であることを検証済みです)。
  メモリに余裕がある環境で再計算コストを避けたい場合は
  `--no-gradient-checkpointing`で無効化できます。

これらは新規学習でのみ有効にしてください。既存のチェックポイントを
`--num-neighbors`・`--sigma-sampling`・`--irreps-hidden`・`--num-convs`を
変えて再開しようとすると、設定が一致しないため明示的にエラーになります
(黙って壊れた状態にはなりません)。`--gradient-checkpointing`は出力に
影響しないため、`--resume`の互換性には影響しません。

```bash
qsub -P <ProjectGroup_ID> \
  -v NUM_NEIGHBORS=auto,SIGMA_SAMPLING=log-normal \
  run_test39.pbs
# GPUメモリに余裕があり再計算コストを避けたい場合
qsub -P <ProjectGroup_ID> -v GRADIENT_CHECKPOINTING=0 run_test39.pbs
```

## スパコンで実行(PBS)

```bash
git clone https://github.com/haru2225/test39.git
cd test39
# 施設で必要なSingularity / Apptainerモジュールを先にロードしてください。
singularity build --fakeroot test39.sif Singularity.test39.def
qsub -P <ProjectGroup_ID> run_test39.pbs
```

`<ProjectGroup_ID>`は自分の課題番号に置き換えます。Apptainer環境では
ビルドコマンドの`singularity`を`apptainer`に置き換えてください。ジョブ
スクリプトはApptainerを自動検出します。ビルドにはネットワークとfakeroot
対応の環境が必要です。施設がビルドを許可するノードで事前に作成するか、
別環境で作成したSIFを転送してください。

既定は**sg8・GPU 1台・CPU 8個・32 GB・20時間**、学習時間予算19.5時間です。
キュー・資源指定は施設に合わせて変更してください。単一GPU実行です。
CUDA 12.4対応のホストGPUドライバーが必要です。

既定では`input/test36-dataset`(1292粒子×10,000フレーム、test37/test38と
同じデータ)から、ジョブ内で`CG_MODE=cations`の粗視化(396粒子)を実行してから
30,000更新を学習します。`positions.npy`はGitHubのファイルサイズ制限を避ける
ため2分割して保存し、PBSスクリプトが実行時に結合します。粗視化済みデータは
`input/test39-cations/`に保存され、2回目以降は元データと設定とハッシュが
一致すれば再利用します。結果は`results/test39-cations/train/`に保存されます。

```bash
# 学習終了後に構造生成(結果: results/test39-cations/generated/)
qsub -P <ProjectGroup_ID> -v STAGE=generate run_test39.pbs

# 時間切れなどで中断した学習・生成を再開
qsub -P <ProjectGroup_ID> -v RESUME=1 run_test39.pbs
qsub -P <ProjectGroup_ID> -v STAGE=generate,RESUME=1 run_test39.pbs

# 学習更新数とバッチサイズを指定
qsub -P <ProjectGroup_ID> -v UPDATES=30000,BATCH_SIZE=2 run_test39.pbs

# 酸素だけ省いてOHのHを残す(524粒子)場合
qsub -P <ProjectGroup_ID> -v CG_MODE=oxygen-only run_test39.pbs
qsub -P <ProjectGroup_ID> -v CG_MODE=oxygen-only,STAGE=generate run_test39.pbs

# 酸素を残した従来の1292粒子モデル
qsub -P <ProjectGroup_ID> -v CG_MODE=none run_test39.pbs
```

再開時はデータ・デバイス・学習/生成設定と`CG_MODE`を初回と同じにしてください。
学習の`UPDATES`は総更新数で、再開時に延長できます。正常終了は終了コード0、
中断してチェックポイントを保存した場合は75です。生成は学習完了を確認してから
投入してください。同じ出力先への同時投入は避けてください。新しい実験では
`TRAIN_DIR` / `GENERATED_DIR`に新しいディレクトリを指定します。

## 自分のデータを使用

`DATASET_PATH`にtest36/37/38で準備したデータセット(`metadata.json`,
`positions.npy`, `cells.npy`)を指定できます。読み込み時にSHA-256を検証します。
同梱データはtest37/test38リポジトリの`input/test36-dataset`と同じ10,000フレーム
のデータです。本番データに置き換える場合も同じ形式を使用してください。データ
セット作成コマンドはこのリポジトリには含みません。粗視化後のデータセットを
直接指定したい場合は`CG_DATASET_PATH`で保存先を、`CG_MODE=none`で
`DATASET_PATH`をそのまま使うよう指定してください。

```bash
qsub -P <ProjectGroup_ID> \
  -v DATASET_PATH=input/my-dataset,CG_DATASET_PATH=input/my-dataset-cations \
  run_test39.pbs
```

データと出力は原則リポジトリ内に配置します。外部ストレージを使う場合は絶対
パスと`EXTRA_BIND=/scratch:/scratch`などの追加マウントを指定してください。
既存コンテナは`SIF_IMAGE`、ランタイムは`CONTAINER_RUNTIME`で変更できます。
その他の設定変数は`run_test39.pbs`に記載しています。

## Pythonで直接実行

依存パッケージがインストールされた環境では以下でも実行できます。

```bash
# 粗視化だけを実行(1292 -> 396粒子)
python test39.py coarse-grain --dataset input/test36-dataset \
  --output input/test39-cations --remove-species ob obos oh ohs ho

# 学習
python test39.py train --dataset input/test39-cations \
  --output results/test39-cations/train --device cuda

# 生成(周期セル内の一様ランダム配置から開始)
python test39.py generate --checkpoint results/test39-cations/train/checkpoint.pt \
  --output results/test39-cations/generated --device cuda

python test39.py train --help
python test39.py generate --help
python test39.py coarse-grain --help
```

`--device cpu`でCPU実行も可能です。`train`は渡されたデータセットをそのまま
使います。前処理(粗視化)の自動実行はPBS側でのみ行われます。生成結果は
`positions.npy`, `generation.json`, `generation_restart.pt`, `final.extxyz`,
`generation.xyz`に保存されます。途中の軌跡は`generation.json`の
`valid_frames`までが有効です。

## 学習途中のチェックポイントから、ランダム配置から生成される過程を確認する

`run_test39.pbs`での学習を止めずに、その時点で保存済みの`checkpoint.pt`だけを
使って生成できます。`test39_analysis.pbs`は現在の`checkpoint.pt`を別ディレクトリ
へコピーしてから(atomicなrenameで保存されているため、学習が同時に書き換えて
いても安全にコピーできます)、そのスナップショットで`test39.py generate`を
実行します。

```bash
git pull --ff-only origin main
qsub -P <ProjectGroup_ID> test39_analysis.pbs
```

test39の生成は元々**必ず周期セル内の一様ランダムな配置から**始まります
(test38と違い、参照構造やノイズ幅を指定する`--initial-state`のような
オプションはありません。これがtest39の設計そのものです)。生成が完了または
時間切れで中断すると、`test39.py`自身が`generation.xyz`という拡張XYZ形式の
軌道ファイルを書き出します。1フレーム目(`generation_step=0`)がランダムな
初期配置、最終フレームが`final.extxyz`と同じ生成結果です。OVITOやVMDでこの
ファイルを開いて再生すると、ランダムな配置から粘土構造らしきものへ収束して
いく過程(またはまだ収束していない途中経過)を確認できます。ステップは生成
計算の番号であり、MDの物理時間ではありません。

既定では`results/test39-cations/train/checkpoint.pt`(`CG_MODE=cations`)を
読みます。`CG_MODE=oxygen-only`や`CG_MODE=none`で学習した場合は同じ変数を
指定してください。

```bash
qsub -P <ProjectGroup_ID> -v CG_MODE=oxygen-only test39_analysis.pbs
```

出力は毎回新しい`results/test39-cations/intermediate/run.XXXXXXXX/`に作成
されます。`checkpoint.pt`がコピーした重み、`generated/`が生成結果です。
実際のパスはジョブログに表示します。ジョブ開始時点の保存済みチェックポイント
を固定するため、その後も学習が進んでいてもこの生成結果は変わりません。
まだチェックポイントが保存されていない場合はエラーで終了します。

生成ステップ数(既定300)や時間予算は変数で変更できます。

```bash
qsub -P <ProjectGroup_ID> -v REVERSE_STEPS=100,GENERATE_TIME_HOURS=2 test39_analysis.pbs
```

学習ジョブとは別のGPU割り当てを待つため、空き状況によっては待機します。

## 範囲・出典

これは変位/スコアデノイザーによる構造生成であり、エネルギー・力場・物理的な
時間を持つMDや平衡分布の検証を提供するものではありません。周期セル上の
ブラウン運動という前向き過程を使いますが、終端分布が一様であることは
セル形状・sigma-maxの設定でのみ保証され、生成された構造が物理的に妥当な
粘土構造であることは保証しません。同梱データも平衡性を保証しません。
生成パラメーターは`test39.py`の実装・既定値を使用しています。

`test38.py`にはDM2([digital-synthesis-lab/DM2](https://github.com/digital-synthesis-lab/DM2))
由来のNequIPモデルコードが(DM2のチェックアウトを必要とせず動作するよう)
移植されています。出典は`test38.py`本体のコメントに、ライセンスは
[licenses/DM2-LICENSE](licenses/DM2-LICENSE)に同梱しています。GPU・PBS・
コンテナ実行は実際のスパコンでの確認が必要です。

学習・生成プログラムが動くことと、粘土の結合・組成・層構造・RDFが正しいことは
別の検証です。現在のネットワークは10 Åの局所グラフなので、長距離秩序を
再現できる保証もありません。検証用フレームと生成構造を比較してください。
