#!/usr/bin/env python3
"""test38: clay CG denoiser using a sigma-conditioned NequIP and a reverse-SDE
sampler, not a hand-tuned Langevin fraction.

test37.py (and test36.py before it) faithfully reused DM2's *unconditional*
denoising-autoencoder scheme (Hsu et al. 2024) -- a single sigma-agnostic NequIP,
applied every generation step with the model's full predicted correction plus a
fresh full-size noise draw ("legacy" sampler). That scheme is DM2's own
generation code too (denoise_snapshot_with_noise_gpu in
DM2/demo/demo_generating/denoise_generate_unconditional.py) -- it is not a
test36/37 bug. A controlled ablation (test37_variant_experiment.py) showed the
legacy sampler collapses this clay system's local bonding around step 55-60
*regardless of training* (undertrained, fully-trained, or trained with a
smaller sigma_max all failed the same way), while switching only the sampler
to an annealed Langevin update (fixed step-size fraction) kept it stable for
the full 100-step test.

This file replaces the hand-picked "fraction" with a sigma-conditioned model
class and a reverse variance-exploding SDE integrator, which derive the step
size from the noise schedule itself instead of a manually tuned constant:

  - Model: `NequIP_TimeEmbed` below is vendored from DM2's own
    DM2/src/graphite/nn/models/e3nn_nequip.py (class NequIP_TimeEmbed) and its
    dependency DM2/src/graphite/nn/conv/e3nn_nequip.py (class Interaction),
    with identical irreps/interaction/tensor-product logic -- same irreps
    stack as the plain NequIP test32/36/37 already use, plus a
    `t_embed`/`t_projection` pair that turns `t = sigma / sigma_max_train`
    into a per-node feature added at every conv layer. `DownselectEdges` and
    `RattleParticles` are likewise vendored from DM2/src/graphite/transforms/
    with identical logic. (Commented-out alternative implementations and
    decorative markers from the original source files were not carried over;
    everything that executes is unchanged.) This file no longer imports
    `test37` or inserts DM2's `src/` onto `sys.path` at all -- it is
    self-contained modulo the ordinary third-party packages (torch,
    torch_geometric, e3nn, ase, numpy) DM2 itself depends on, so it runs
    without a DM2 checkout or a `DM2_ROOT` environment variable present.
  - Update rule: between noise levels sigma -> next_sigma,
        variance_drop = max(sigma**2 - next_sigma**2, 0)
        score_step    = variance_drop / sigma**2
        pos -= score_step * model(pos, sigma) ; pos += N(0,1)*sqrt(variance_drop)
    for the stochastic (annealed Langevin) steps, then a deterministic DDIM
    tail (`pos -= (1 - next_sigma/sigma) * model(pos, sigma)`) for the last
    few steps, matching the "noisy, then polish" split every prior file in
    this project has used. NOTE there is no `demo_probing` directory or
    `graphite.probing` module anywhere in DM2's history, so unlike the model
    class above, this *update rule* is not a port of DM2 demo/training code --
    it is a standard annealed-Langevin / variance-exploding reverse-diffusion
    step (Song & Ermon 2019), written out directly against `NequIP_TimeEmbed`.

  Dataset loading, graph construction and species handling below are also
  vendored from test37.py (functions `load_dataset`, `graph`,
  `atoms_from_meta`, `InitialEmbedding`, `architecture`, plus small argparse/
  checkpoint/RNG helpers) rather than reusing DM2's own species-from-atomic-
  number helper, because the clay CG mapping's four oxygen-role species
  (ob/obos/oh/ohs) share atomic number 8 by design (they are exported as real
  oxygen for masses/visualization) and must stay distinct model inputs, keyed
  by `dataset_metadata["type_ids"]` as test36/37 already do.

train warm-starts from an already-trained test37 (or test36) plain-NequIP
checkpoint: the shared interaction/output layers are copied over unchanged,
and the new time-projection layer is zero-initialized so the warm-started
model is bitwise-identical to the source checkpoint until further training
moves it (same trick test37_variant_experiment.py's ad hoc SigmaEmbedding
used, applied here to DM2's own NequIP_TimeEmbed instead of a bespoke class).
Training from scratch (no --warm-start) is also supported, matching DM2's own
train_sio2_time_denoiser.py exactly.

    python test38.py train --dataset test36-aa/dataset-pilot \
        --warm-start clay-mineral-test2/checkpoint.pt \
        --output clay-mineral-test3 --device cuda
    python test38.py generate --checkpoint clay-mineral-test3/checkpoint.pt \
        --output clay-mineral-test3/generated --device cuda

Everything test37.py's own CAVEAT says still applies: no scalar energy, no
equilibrium claim, no physical clock. This file only replaces *how* the
reverse process is integrated; it does not add anything DM2's own codebase
does not already implement elsewhere for a different system (SiO2).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from functools import partial
from pathlib import Path

import ase.io
import numpy as np
import torch

torch.serialization.add_safe_globals([slice])  # e3nn loads its own constants.pt with torch.load

from ase import Atoms
from ase.neighborlist import primitive_neighbor_list
from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import scatter

# ===== 基本設定・定数 =====
ROOT = Path(__file__).resolve().parent
FORMAT = "test38-clay-cg-time-denoiser-v1"  # このtest38用チェックポイントの識別子(test37のものとは別)
DATASET_FORMATS = {"test37-clay-cg-denoiser-v1", "test36-clay-cg-denoiser-v1"}  # 読み込めるデータセット形式(test36/37と共通)
CAVEAT = (
    "test38 sigma-conditioned displacement denoiser with a reverse variance-exploding "
    "SDE sampler (a sigma-conditioned NequIP, vendored from DM2, plus an annealed-Langevin "
    "update rule, ported to test37's species/graph handling). Not a scalar energy or a "
    "temperature-conditioned equilibrium score. No energy/virial, rigid platelets, explicit "
    "electrostatics, pressure, shear response or physical kinetics are provided. One model "
    "per condition; generation frames are not equilibrium MD data."
)
STOP = False  # SIGINT/SIGTERMを受け取ったらTrueにして、train/generateループを安全に中断させるフラグ


def request_stop(signum, frame):
    # シグナルハンドラ: Ctrl-CやHPCのジョブ時間切れ通知を受けたときに呼ばれる。
    # ここで即座に終了せず、ループ側にSTOPを見てもらってから
    # チェックポイントを保存してから終了する(生成/学習の再開性を保つため)。
    global STOP
    STOP = True


def positive(value):
    # argparseの型変換関数: 正の有限値であることを保証する(sigmaや学習率など)。
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def count(value):
    # argparseの型変換関数: 1以上の整数であることを保証する(ステップ数など)。
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_count(value):
    # argparseの型変換関数: 0以上の整数であることを保証する(deterministic-stepsなど、0も許す値)。
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def digest(path):
    # ファイルのSHA-256ハッシュを計算する。データセットやチェックポイントが
    # 途中で書き換わっていないか(再現性)を確認するために使う。
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, obj):
    # JSONを一時ファイルに書いてからatomicにrename。書き込み途中でプロセスが
    # 落ちても、既存の正しいファイルが壊れた中途半端な内容で上書きされない。
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path, obj):
    # チェックポイント(torch.save)も同様にtmp書き込み→rename でatomicに保存する。
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(obj, temporary)
    temporary.replace(path)


def new_output(path):
    # 出力先ディレクトリを新規作成する。既に中身がある場合はエラーにして、
    # 別の実行結果を誤って上書き・混在させないようにする。
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError(f"Use a new or empty output directory: {path}")
    return path


def device_for(name):
    # "auto"ならCUDA→MPS→CPUの優先順で自動選択。明示指定されたデバイスが
    # 実際には使えない場合はここでエラーにする(黙って別デバイスに落とさない)。
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(name)


def rng_state():
    # numpy/torch(+CUDA/MPSがあれば)の乱数状態をまとめて保存用に取得する。
    # --resumeで学習・生成を再開したときに、乱数列を継続させるために使う。
    state = {"numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    # rng_state()で保存した乱数状態を復元する。
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


# ===== test37から移植したデータ・グラフ・種(species)まわりのヘルパー =====

def bessel(x, start=0.0, end=1.0, num_basis=8, eps=1e-5):
    """Vendored from DM2/src/graphite/nn/basis.py (function bessel)."""
    # スカラー値(ここではエッジの距離)を、複数のBessel基底関数の値に展開する。
    # NequIPが距離をそのまま数値として使うのではなく、周波数の異なる
    # sin波の重ね合わせとして表現することで、距離依存性を学習しやすくする。
    x = x[..., None] - start + eps
    c = end - start
    n = torch.arange(1, num_basis + 1, dtype=x.dtype, device=x.device)
    return ((2 / c) ** 0.5) * torch.sin(n * torch.pi * x / c) / x


class InitialEmbedding(nn.Module):
    """Same embedding as test32.py/test37.py: two species embeddings and Bessel edges."""
    # ノード(原子)の「種」を2種類の埋め込みベクトルに変換し、エッジ(ボンド)の
    # 距離をBessel基底に変換する、NequIPモデルの最初の入力層。
    def __init__(self, num_species, cutoff):
        super().__init__()
        self.embed_node_x = nn.Embedding(num_species, 8)  # 更新されていくノード特徴量の初期値
        self.embed_node_z = nn.Embedding(num_species, 8)  # モデル全体を通して固定される補助的なノード特徴量
        self.embed_edge = partial(bessel, start=0.0, end=cutoff, num_basis=16)

    def forward(self, data):
        data.h_node_x = self.embed_node_x(data.x)
        data.h_node_z = self.embed_node_z(data.x)
        data.h_edge = self.embed_edge(data.edge_attr.norm(dim=-1))
        return data


def architecture(num_species, cutoff):
    # モデルの構造(irreps=e3nnの回転等変な特徴量の型、畳み込み層の数など)を
    # 1つの辞書にまとめたもの。チェックポイントに保存しておき、生成時に
    # 同じ構造のモデルを再構築するために使う。
    #
    # irreps_hidden/num_convsは元は test37 と同じ("64x0e + 32x1e", 3層)だった。
    # これはtest38の「参照構造にわずかなノイズを加えたところから戻す」局所
    # デノイザー用の規模で、test39が要求する「セル内で完全にランダムな配置
    # から周期的な結晶格子を再構築する」というはるかに難しい生成タスクには
    # 小さすぎると判断し、モデルクラス(NequIP)自体は変えずに容量だけを
    # 増やした。
    #
    # l_max=5まで拡張しているのは、この粘土鉱物の主要な配位構造(Si四面体
    # (Td対称性、最初の非自明な多重極項はl=3)、Al/Mg八面体(Oh対称性、
    # 最初の非自明な多重極項はl=4))を、エッジの球面調和展開だけで直接
    # 表現できるようにするため。l=2までしかないと、これらの配位構造は
    # 複数層のテンソル積を重ねて間接的にしか再構成できない。num_convsは
    # その分、間接的な多体相関の再構成に頼らなくてよくなった分だけ3層に
    # 戻している。
    return dict(num_species=num_species, cutoff_angstrom=cutoff,
                irreps_node_x="8x0e", irreps_node_z="8x0e",
                irreps_hidden="128x0e + 64x1e + 32x2e + 16x3e + 8x4e + 4x5e",
                irreps_edge="4x0e + 4x1e + 4x2e + 2x3e + 2x4e + 1x5e",
                irreps_out="1x1e", num_convs=3, radial_neurons=[16, 64], num_neighbors=12)


def graph(positions, cell, type_ids, cutoff, device):
    # Same periodic neighbor construction as test32/37; graph vectors are not
    # differentiable w.r.t. positions. This is deliberate for a dx model.
    # 周期境界条件(PBC)付きで、cutoff半径以内の原子対(i, j)とその変位ベクトルvecを列挙し、
    # PyTorch Geometricの`Data`グラフオブジェクトを組み立てる。
    i, j, vec = primitive_neighbor_list("ijD", [True] * 3, cell, positions, cutoff=cutoff)
    if not len(i):
        raise ValueError("No graph edges: check box, units and cutoff")
    return Data(x=torch.as_tensor(type_ids, dtype=torch.long, device=device),
                pos=torch.as_tensor(np.asarray(positions).copy(), dtype=torch.float32, device=device),
                edge_index=torch.as_tensor(np.stack((i, j)), dtype=torch.long, device=device),
                edge_attr=torch.as_tensor(vec, dtype=torch.float32, device=device))


def load_dataset(folder):
    # test36/37形式のデータセット(positions.npy, cells.npy, metadata.json)を読み込む。
    # メタデータのフォーマット・単位・配列の形状・保存後の改ざん有無(sha256)を検証してから返す。
    folder = Path(folder).resolve()
    meta = json.loads((folder / "metadata.json").read_text())
    if meta.get("format") not in DATASET_FORMATS or meta.get("length_unit") != "angstrom":
        raise ValueError("Expected a test36/37 dataset with explicit angstrom units")
    pos = np.load(folder / "positions.npy", mmap_mode="r", allow_pickle=False)
    cells = np.load(folder / "cells.npy", mmap_mode="r", allow_pickle=False)
    if pos.shape != (meta["frames"], len(meta["type_ids"]), 3) or cells.shape != (len(pos), 3, 3):
        raise ValueError("Dataset shapes disagree with metadata")
    for name in ("positions.npy", "cells.npy"):
        if digest(folder / name) != meta["sha256"][name]:
            raise ValueError(f"Dataset modified after preparation: {name}")
    return pos, cells, meta


def atoms_from_meta(positions, cell, meta):
    # モデル用のtype_id配列(粘土CGマッピングにおける「役割」ごとの疑似的な原子種)を、
    # 可視化・エクスポート用にase.Atomsオブジェクト(実際の原子番号・質量を持つ)へ変換する。
    # 酸素の役割ごとの4種(ob/obos/oh/ohs)は物理的にはどれも酸素(原子番号8)だが、
    # モデル入力上は別種として扱う必要があるため、"cg_type"配列に元のtype_idを残しておく。
    ids = np.asarray(meta["type_ids"])
    atoms = Atoms(numbers=[meta["species"][i]["atomic_number"] for i in ids],
                  positions=positions, cell=cell, pbc=True,
                  masses=[meta["species"][i]["mass_amu"] for i in ids])
    atoms.set_array("cg_type", ids.copy())
    return atoms


# --- Vendored from DM2/src/graphite (nn/conv/e3nn_nequip.py, nn/models/e3nn_nequip.py,
# transforms/downselect_edges.py, transforms/rattle_particles.py) so that this file does
# not import DM2 or require a DM2 checkout / DM2_ROOT to be present. -----------------

def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    # 2つのirreps(既約表現)のテンソル積が、指定した出力既約表現ir_outを
    # 生成しうるかどうかを判定する。e3nnのGate/TensorProductを組み立てる際に、
    # 「この組み合わせは数学的に意味があるか」を事前にチェックするために使う。
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)
    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


class Compose(nn.Module):
    # 2つのモジュール(例: Interaction畳み込み層とGate活性化層)を直列に繋げるだけの
    # 薄いラッパー。NequIP_TimeEmbedの各層は Compose(Interaction, Gate) として作られる。
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        return self.second(x)


class GaussianBasisEmbedding(nn.Module):
    """Embeds a scalar value in [0,1] using a Gaussian basis set followed by a dense layer."""
    # sigma/time条件付けの核心部分: スカラー値t(=sigma/sigma_max_train, 0〜1の値)を
    # 複数のガウス基底関数に展開し(bessel関数と同じ発想)、2層のMLPで
    # ベクトル特徴量に変換する。NequIP_TimeEmbedがこの出力(h_node_t)を
    # 各畳み込み層のノード特徴量に足し込むことで、モデルが「今どれくらいの
    # ノイズレベルを相手にしているか」を認識できるようになる。
    def __init__(self, num_basis=12, embedding_dim=32, min_sigma=0.1,
                 learn_means=False, learn_sigmas=False, min_value=0, max_value=1):
        super().__init__()
        means = torch.linspace(min_value, max_value, num_basis)  # 各ガウス基底の中心位置
        if learn_means:
            self.means = nn.Parameter(means)
        else:
            self.register_buffer('means', means)
        sigmas = torch.ones_like(means) * max(min_sigma, 1.0 / (num_basis - 1))  # 各ガウス基底の幅
        if learn_sigmas:
            self.sigmas = nn.Parameter(sigmas)
        else:
            self.register_buffer('sigmas', sigmas)
        hidden_dim = max(embedding_dim * 2, num_basis)
        self.layer1 = nn.Linear(num_basis, hidden_dim)
        self.activation = nn.Softplus()
        self.layer2 = nn.Linear(hidden_dim, embedding_dim)

    def gaussian_basis(self, x):
        # tの値を、各ガウス基底中心からの距離に応じた「近さ」のベクトルに変換する。
        if x.dim() == 1:
            x = x.unsqueeze(1)
        x_expanded = x.expand(-1, self.means.shape[0])
        return torch.exp(-0.5 * ((x_expanded - self.means) / self.sigmas) ** 2)

    def forward(self, x):
        basis_activation = self.gaussian_basis(x)
        hidden = self.activation(self.layer1(basis_activation))
        return self.layer2(hidden)


class Interaction(nn.Module):
    """Equivariant `Interaction` layer from NequIP (https://arxiv.org/pdf/2101.03164.pdf)."""
    # NequIPの中核となる1つの畳み込み(メッセージパッシング)層。
    # 回転・並進に対して等変(equivariant)なテンソル積を使って、
    # 各原子の近傍からの情報を集約し、ノード特徴量を更新する。
    def __init__(self, irreps_in, irreps_node, irreps_edge, irreps_out,
                 radial_neurons=[16, 64], num_neighbors=1):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_edge = o3.Irreps(irreps_edge)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_neighbors = num_neighbors

        # irreps_in(ノード特徴量)とirreps_edge(球面調和関数)のテンソル積のうち、
        # 最終的にirreps_outとして使えるものだけを集めて、中間表現irreps_midを組み立てる。
        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_in):
            for j, (_, ir_edge) in enumerate(self.irreps_edge):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, 'uvu', True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()
        assert irreps_mid.dim > 0, (
            f"irreps_in={self.irreps_in} times irreps_edge={self.irreps_edge} "
            f"produces nothing in irreps_out={self.irreps_out}."
        )
        instructions = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        self.sc = o3.FullyConnectedTensorProduct(self.irreps_in, self.irreps_node, self.irreps_out)  # 自己結合(残差)経路
        self.lin1 = o3.FullyConnectedTensorProduct(self.irreps_in, self.irreps_node, self.irreps_in)  # メッセージパッシング前の線形変換
        self.conv = o3.TensorProduct(
            self.irreps_in, self.irreps_edge, irreps_mid, instructions,
            internal_weights=False, shared_weights=False,  # 重みは下のmlp(ボンド長依存)から供給される
        )
        self.lin2 = o3.FullyConnectedTensorProduct(irreps_mid, self.irreps_node, self.irreps_out)  # メッセージパッシング後の線形変換
        self.mlp = FullyConnectedNet(radial_neurons + [self.conv.weight_numel], torch.nn.functional.silu)  # ボンド長(Bessel基底)からconvの重みを生成するMLP

        # SkipInit mechanism inspired by https://arxiv.org/pdf/2002.10444.pdf
        # 学習開始時点ではconvパスの寄与をゼロにしておき(alpha=0スタート)、
        # 学習が進むにつれて徐々にconvパスの影響を強めていく安定化トリック。
        self.alpha = o3.FullyConnectedTensorProduct(irreps_mid, self.irreps_node, "0e")
        with torch.no_grad():
            self.alpha.weight.zero_()
        assert self.alpha.output_mask[0] == 1.0, (
            f"irreps_mid={irreps_mid} and irreps_node={self.irreps_node} are not able to generate scalars."
        )

    def forward(self, x, node_attr, edge_index, edge_attr, edge_len_emb):
        i, j = edge_index
        num_nodes = x.size(0)
        node_self_connection = self.sc(x, node_attr)  # 残差(自己)経路の出力
        node_features = self.lin1(x, node_attr)
        # 各エッジについて、送り手ノードiの特徴量とエッジの球面調和関数edge_attrとの
        # テンソル積を、ボンド長依存の重み(self.mlp(edge_len_emb))で計算する。
        edge_features = self.conv(node_features[i], edge_attr, weight=self.mlp(edge_len_emb))
        # 受け手ノードjごとにエッジメッセージを合計(scatter)し、近傍数で正規化する。
        node_features = scatter(edge_features, j, dim=0, dim_size=num_nodes).div(self.num_neighbors ** 0.5)
        node_conv_out = self.lin2(node_features, node_attr)
        alpha = self.alpha(node_features, node_attr)
        m = self.sc.output_mask
        alpha = (1 - m) + alpha * m
        # 残差経路 + alphaでスケールした畳み込み経路、を足し合わせて出力する。
        return node_self_connection + alpha * node_conv_out


class NequIP_TimeEmbed(nn.Module):
    """Sigma/time-conditioned NequIP (https://arxiv.org/pdf/2101.03164.pdf), vendored from
    DM2/src/graphite/nn/models/e3nn_nequip.py.

    Args:
        init_embed (function): Initial embedding function/class for nodes and edges.
        irreps_node_x (Irreps or str): Irreps of input node features.
        irreps_node_z (Irreps or str): Irreps of auxiliary node features (not updated throughout model).
        irreps_hidden (Irreps or str): Irreps of node features at hidden layers.
        irreps_edge (Irreps or str): Irreps of spherical_harmonics.
        irreps_out (Irreps or str): Irreps of output node features.
        num_convs (int): Number of interaction/conv layers. Must be more than 1.
        radial_neurons (list of ints): Number of neurons per layers in the MLP that learns from bond distances.
        num_neighbors (float): Typical or average node degree (used for normalization).
    """
    def __init__(self, init_embed, irreps_node_x='8x0e', irreps_node_z='8x0e',
                 irreps_hidden='64x0e + 32x1e + 32x2e', irreps_edge='1x0e + 1x1e + 1x2e',
                 irreps_out='1x1e', num_convs=3, radial_neurons=[16, 64], num_neighbors=12):
        super().__init__()
        self.init_embed = init_embed
        self.irreps_node_x = o3.Irreps(irreps_node_x)
        self.irreps_node_z = o3.Irreps(irreps_node_z)
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_edge = o3.Irreps(irreps_edge)
        self.num_convs = num_convs

        act_scalars = {1: nn.functional.silu, -1: torch.tanh}
        act_gates = {1: torch.sigmoid, -1: torch.tanh}

        # num_convs層分のInteraction+Gateを積み重ねる。各層で、スカラー成分と
        # ベクトル/テンソル成分をGateで非線形活性化しながら特徴量を更新していく。
        irreps = self.irreps_node_x
        self.interactions = nn.ModuleList()
        for _ in range(num_convs):
            irreps_scalars = o3.Irreps([(m, ir) for m, ir in self.irreps_hidden
                                         if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge, ir)])
            irreps_gated = o3.Irreps([(m, ir) for m, ir in self.irreps_hidden
                                       if ir.l > 0 and tp_path_exists(irreps, self.irreps_edge, ir)])

            if irreps_gated.dim > 0:
                if tp_path_exists(irreps_node_z, self.irreps_edge, "0e"):
                    ir = "0e"
                elif tp_path_exists(irreps_node_z, self.irreps_edge, "0o"):
                    ir = "0o"
                else:
                    raise ValueError(f"irreps={irreps} times irreps_edge={self.irreps_edge} is unable "
                                      f"to produce gates needed for irreps_gated={irreps_gated}.")
            else:
                ir = None
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

            gate = Gate(
                irreps_scalars, [act_scalars[ir.p] for _, ir in irreps_scalars],
                irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],
                irreps_gated,
            )
            conv = Interaction(
                irreps_in=irreps, irreps_node=self.irreps_node_z, irreps_edge=self.irreps_edge,
                irreps_out=gate.irreps_in, radial_neurons=radial_neurons, num_neighbors=num_neighbors,
            )
            irreps = gate.irreps_out
            self.interactions.append(Compose(conv, gate))

        self.out = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps, irreps_in2=self.irreps_node_z, irreps_out=self.irreps_out,
        )

        # ここがtest38の要: sigma/time条件付けのための層を追加する。
        # 各畳み込み層のスカラー隠れ次元数(size_embed)に合わせたGaussianBasisEmbeddingで
        # t(=sigma/sigma_max_train)を埋め込み、t_projectionで各ノード特徴量の次元数に線形変換する。
        size_embed = int(str(irreps).split("x")[0])
        self.t_embed = GaussianBasisEmbedding(embedding_dim=size_embed)
        t_embed_dim = self.t_embed.layer2.out_features
        self.t_projection = nn.Linear(t_embed_dim, irreps.dim)

    def forward(self, data, t):
        data = self.init_embed(data)
        edge_index, edge_attr = data.edge_index, data.edge_attr
        h_node_x, h_node_z, h_edge = data.h_node_x, data.h_node_z, data.h_edge

        # スカラー値t(このバッチ全体で1つの値)を埋め込み、全ノードに同じベクトルとして
        # ブロードキャストする。1バッチ=1つのグラフしか正しく条件付けできない点に注意
        # (下のtrain()内のrattle_atのコメント参照)。
        h_node_t = self.t_embed(t)
        h_node_t = h_node_t.expand(h_node_x.shape[0], -1)
        h_node_t = self.t_projection(h_node_t)

        # エッジベクトルを球面調和関数に変換してから、各Interaction+Gate層を通し、
        # 毎層h_node_tを足し込むことでsigma情報を伝え続ける。
        edge_sh = o3.spherical_harmonics(self.irreps_edge, edge_attr, normalize=True, normalization='component')
        for layer in self.interactions:
            h_node_x = layer(h_node_x, h_node_z, edge_index, edge_sh, h_edge)
            h_node_x = h_node_x + h_node_t

        # 最終的に3次元ベクトル(irreps_out='1x1e')、つまり各原子の変位予測dxを出力する。
        return self.out(h_node_x, h_node_z)


class DownselectEdges(BaseTransform):
    """Vendored from DM2/src/graphite/transforms/downselect_edges.py."""
    # graph()はtraining用に少し大きめのlarge_cutoffで候補エッジを作っておき、
    # このDownselectEdgesで実際のモデルcutoff以内のエッジだけに絞り込む。
    # (RattleParticlesでノイズを加えた後、距離が変化してからこの絞り込みを行うことで、
    # ノイズ後もcutoff以内に収まっているエッジだけを使う。)
    def __init__(self, cutoff, cell=None):
        super().__init__()
        self.cutoff = cutoff
        self.cell = cell

    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        mask = (edge_attr[:, :3].norm(dim=1) <= self.cutoff)
        data.edge_index = edge_index[:, mask]
        data.edge_attr = edge_attr[mask]
        return data

    def forward(self, data):
        return self.__call__(data)

    def __repr__(self):
        return f'{self.__class__.__name__}(cutoff={self.cutoff})'


class RattleParticles(BaseTransform):
    """Vendored from DM2/src/graphite/transforms/rattle_particles.py. Applies a random
    Gaussian noise to particle positions, with standard deviation drawn uniformly from
    [sigma_min, sigma_max]."""
    # 学習時に「正解の構造」にガウスノイズを加えて壊し(corrupt)、モデルには
    # 「加えられたノイズdxを予測して元に戻す」タスクを学習させる。これが
    # denoising score matching(スコアベース生成モデル)の学習の基本形。
    def __init__(self, sigma_max, sigma_min=0.001):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, data):
        if data.batch is not None:
            # バッチ内のグラフごとに別々のsigmaを[sigma_min, sigma_max]から一様サンプルする
            # (test38ではtrain()側でsigma_min=sigma_maxに固定して呼ぶため、実質バッチ全体で1つのsigmaになる)。
            sigma = torch.empty(data.num_graphs, device=data.pos.device).uniform_(
                self.sigma_min, self.sigma_max)
            sigma = sigma[data.batch, None]
        else:
            sigma = torch.empty(1, device=data.pos.device).uniform_(self.sigma_min, self.sigma_max)

        eps = torch.randn_like(data.pos)  # 標準正規分布ノイズ
        data.dx = sigma * eps  # モデルが予測すべき正解の変位(ノイズそのもの)
        data.pos = data.pos + data.dx  # 座標を実際に壊す

        if data.edge_attr is not None:
            # 座標を動かしたので、既存のエッジベクトル(相対変位)も整合するように更新する。
            i, j = data.edge_index
            data.edge_attr = data.edge_attr + data.dx[j] - data.dx[i]

        data.sigma = sigma  # 後で参照できるように保存(このtest38ではdata.sigmaは直接は使わずtを別途渡す)
        data.eps = eps
        return data

    def forward(self, data):
        return self.__call__(data)


# --- test38-specific code -------------------------------------------------------------

def build_time_model(config: dict, device: torch.device) -> nn.Module:
    # architecture()で作った構成辞書からNequIP_TimeEmbedモデルを組み立てる。
    values = {k: v for k, v in config.items() if k not in ("num_species", "cutoff_angstrom")}
    model = NequIP_TimeEmbed(
        init_embed=InitialEmbedding(config["num_species"], config["cutoff_angstrom"]),
        **values,
    )
    return model.to(device)


def warm_start_from_plain_checkpoint(model: NequIP_TimeEmbed, plain_state_dict: dict) -> None:
    """Copy every weight NequIP_TimeEmbed shares with plain NequIP, then zero the new
    time-projection layer so the warm-started model equals the source checkpoint at
    t=anything until training moves it."""
    # test37(sigma条件付けなしのplain NequIP)で既に学習済みのチェックポイントから
    # 重みを引き継ぐための関数。NequIP_TimeEmbedはplain NequIPと全く同じ
    # Interaction/Gate/出力層を持つので、それらの重みはそのままコピーできる。
    # 新規に追加されたt_embed/t_projection(sigma条件付け用の層)だけは
    # plain側チェックポイントに存在しないので、ここではまだ扱わない。
    own_state = model.state_dict()
    missing = [k for k in own_state if k not in plain_state_dict]
    unexpected = [k for k in plain_state_dict if k not in own_state]
    if unexpected:
        raise ValueError(f"--warm-start checkpoint has unexpected keys for this architecture: {unexpected}")
    if any(not k.startswith(("t_embed.", "t_projection.", "time_scalar_mask")) for k in missing):
        raise ValueError(f"--warm-start checkpoint is missing non-time-conditioning keys: {missing}")
    own_state.update(plain_state_dict)
    model.load_state_dict(own_state)
    # t_projectionの重み・バイアスをゼロにすることで、h_node_t(sigma由来の特徴量)が
    # 各層の出力に何も足さない状態にする。つまりウォームスタート直後は
    # 「sigmaを完全に無視するモデル」= 元のplainチェックポイントと数値的に全く同じ
    # 挙動になり、そこから学習を進めるにつれて徐々にsigma依存性を獲得していく。
    nn.init.zeros_(model.t_projection.weight)
    nn.init.zeros_(model.t_projection.bias)


def checkpoint_time_model(path, device):
    # test38形式のチェックポイントを読み込み、保存されていたarchitectureから
    # モデルを再構築して重みを復元する(生成時に使う)。
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") != FORMAT:
        raise ValueError("Expected a test38 sigma-conditioned checkpoint")
    model = build_time_model(ck["architecture"], device)
    model.load_state_dict(ck["model_state_dict"])
    return model.eval(), ck


def train(args):
    # --- 入力チェックとセットアップ ---
    positions, cells, meta = load_dataset(args.dataset)
    if len(positions) < 3:
        raise ValueError("Training requires at least three frames")
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("validation-fraction must be between zero and 0.5")
    if args.sigma_max < 0.001 or args.large_cutoff < args.cutoff:
        raise ValueError("Require sigma-max >= 0.001 and large-cutoff >= cutoff")
    device = device_for(args.device)
    output = args.output.resolve() if args.resume else new_output(args.output)
    checkpoint = output / "checkpoint.pt"
    if args.resume and not checkpoint.is_file():
        raise ValueError("--resume requires output/checkpoint.pt")

    # フレームを学習用/検証用に分割(先頭split個が学習、残りが検証)。
    split = max(1, int(len(positions) * (1 - args.validation_fraction)))
    # この実行の設定をすべて記録しておく。--resumeで再開する際、設定が
    # 完全一致するかどうかの検証にも使う(途中でハイパラを変えて再開させない)。
    settings = dict(
        dataset_sha256=meta["sha256"], metadata_sha256=digest(args.dataset / "metadata.json"),
        cutoff=args.cutoff, large_cutoff=args.large_cutoff,
        sigma_min=args.sigma_min, sigma_max=args.sigma_max,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        seed=args.seed, split_frame=split, device=str(device), log_every=args.log_every,
        warm_start_sha256=(digest(args.warm_start) if args.warm_start else None),
    )
    config = architecture(len(meta["species"]), args.cutoff)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = build_time_model(config, device)
    if not args.resume and args.warm_start is not None:
        # test37/36で学習済みのplainチェックポイントからウォームスタートする場合。
        # アーキテクチャ(species数・cutoffなど)が一致していることを確認してから、
        # 共有できる重みだけをコピーする。
        source_ck = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        if source_ck.get("architecture") != config:
            raise ValueError("--warm-start checkpoint architecture does not match this dataset")
        warm_start_from_plain_checkpoint(model, source_ck["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history, completed = [], 0
    if args.resume:
        # 既存のtest38チェックポイントから学習状態(重み・optimizer・乱数状態・履歴)を復元する。
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if ck.get("format") != FORMAT or ck["settings"] != settings:
            raise ValueError("Resume settings/data/device differ from checkpoint")
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        restore_rng(ck["rng"])
        history, completed = ck["history"], ck["completed_updates"]
    downselect = DownselectEdges(cutoff=args.cutoff)

    def rattle_at(data, sigma_value):
        # NequIP_TimeEmbed broadcasts a single scalar `t` to every node in the call
        # (`h_node_t.expand(n_nodes, -1)`); it does not support one sigma per graph
        # within a batch the way RattleParticles' own per-graph sigma mechanism
        # assumes (which is also silently dropped by PyG's Batch storage on a
        # multi-graph batch). So every graph in a training step is rattled with the
        # SAME drawn sigma, matching what the model can actually condition on; sigma
        # still varies step to step across the full [sigma_min, sigma_max] range
        # over the course of training.
        return RattleParticles(sigma_min=sigma_value, sigma_max=sigma_value)(data)
    deadline = time.monotonic() + args.time_budget_hours * 3600  # HPCジョブの時間切れ前に安全に停止するための締切

    def save():
        # 学習の途中経過(重み・optimizer状態・乱数状態・履歴)をチェックポイントに、
        # 進捗の要約をtraining.jsonに、それぞれ書き出す。
        save_checkpoint(checkpoint, dict(
            format=FORMAT, architecture=config, settings=settings,
            model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
            optimizer=optimizer.state_dict(), rng=rng_state(), history=history,
            completed_updates=completed, requested_updates=args.updates,
            dataset_metadata=meta, large_cutoff=args.large_cutoff,
            start_positions_angstrom=np.asarray(positions[0]).copy(),
            cell_angstrom=np.asarray(cells[0]).copy(), scientific_caveat=CAVEAT,
        ))
        save_json(output / "training.json", dict(
            completed_updates=completed, requested_updates=args.updates,
            history=history, scientific_caveat=CAVEAT,
        ))

    print(f"train (test38, sigma-conditioned): device={device}, frames={len(positions)}, "
          f"train/validation={split}/{len(positions) - split}, "
          f"warm_start={'yes' if args.warm_start else 'no'}", flush=True)
    # --- 学習ループ本体 ---
    for step in range(completed + 1, args.updates + 1):
        if STOP or time.monotonic() >= deadline:
            # 中断シグナルか時間切れなら、その場でチェックポイントを保存して終了コード75を返す
            # (HPCジョブスケジューラに「再投入すれば続きから再開できる」ことを伝える慣習的な値)。
            save()
            print("Training paused; resume with --resume", flush=True)
            return 75
        model.train()
        # 学習フレームからランダムにbatch_size個選び、
        indices = np.random.randint(split, size=args.batch_size)
        # このステップで使うsigmaを[sigma_min, sigma_max]から1つだけサンプルする
        # (バッチ内の全グラフに同じsigmaを使う。理由は下のコメント参照)。
        sigma_value = float(np.random.uniform(args.sigma_min, args.sigma_max))
        batch = Batch.from_data_list([
            graph(positions[i], cells[i], meta["type_ids"], args.large_cutoff, device)
            for i in indices
        ])
        # ノイズを加えてから(rattle_at)、モデルのcutoffに絞り込む(downselect)。
        batch = downselect(rattle_at(batch, sigma_value))
        # モデルに渡すt(正規化されたsigma)を計算し、順伝播・損失計算・逆伝播。
        t = torch.tensor([sigma_value / args.sigma_max], device=device, dtype=batch.pos.dtype)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch, t)
        # 目的関数: 加えたノイズ(batch.dx)をどれだけ正確に予測できたかのMSE(denoising score matching)。
        loss = torch.nn.functional.mse_loss(prediction, batch.dx)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        completed = step
        if step == 1 or step % args.log_every == 0 or step == args.updates:
            # 定期的に検証データでの損失もログに出す。乱数状態を退避・復元することで、
            # 検証評価が学習側の乱数列(データ選択やノイズ)に影響を与えないようにしている。
            model.eval()
            saved_rng = rng_state()
            losses = []
            valid_rng = np.random.default_rng(args.seed + 1)
            for i in range(split, min(split + 4, len(positions))):
                valid_sigma = float(valid_rng.uniform(args.sigma_min, args.sigma_max))
                valid = graph(positions[i], cells[i], meta["type_ids"], args.large_cutoff, device)
                valid = downselect(rattle_at(valid, valid_sigma))
                valid_t = torch.tensor([valid_sigma / args.sigma_max], device=device, dtype=valid.pos.dtype)
                with torch.no_grad():
                    losses.append(torch.nn.functional.mse_loss(
                        model(valid, valid_t), valid.dx
                    ).item())
            restore_rng(saved_rng)
            val_loss = float(np.mean(losses))
            row = dict(step=step, train_mse_A2=float(loss.detach().cpu()), valid_mse_A2=val_loss)
            history.append(row)
            print(json.dumps(row), flush=True)
        if step % args.checkpoint_every == 0:
            save()
    save()
    print(f"Checkpoint: {checkpoint}")


@torch.no_grad()
def generate(args):
    # --- セットアップ: チェックポイント読み込みと出力先準備 ---
    if args.start_sigma < 0.001:
        raise ValueError("start-sigma must be at least 0.001 angstrom")
    device = device_for(args.device)
    model, ck = checkpoint_time_model(args.checkpoint, device)
    meta, cell = ck["dataset_metadata"], ck["cell_angstrom"]
    cutoff = ck["architecture"]["cutoff_angstrom"]
    sigma_max_train = ck["settings"]["sigma_max"]  # tの正規化に使う、学習時のsigma_max
    output = args.output.resolve() if args.resume else new_output(args.output)
    settings = dict(
        checkpoint_sha256=digest(args.checkpoint), reverse_steps=args.reverse_steps,
        deterministic_steps=args.deterministic_steps, start_sigma=args.start_sigma,
        sigma_min=args.sigma_min, thermal_scale=args.thermal_scale,
        seed=args.seed, device=str(device), cutoff_angstrom=cutoff,
    )
    total = args.reverse_steps
    state_path = output / "generation_restart.pt"
    if args.resume:
        # 中断していた生成を再開する場合: 直前の座標・乱数状態を復元し、
        # 既存のpositions.npyメモリマップを追記モードで開く。
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["settings"] != settings:
            raise ValueError("Generation resume settings differ; use a new output directory")
        pos, completed = state["positions"].to(device), state["step"]
        restore_rng(state["rng"])
        trajectory = np.lib.format.open_memmap(output / "positions.npy", mode="r+")
    else:
        # 新規に生成を開始する場合: 学習データセットの最初のフレーム(訓練時に
        # 保存しておいたstart_positions_angstrom)を出発点にする。
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        pos = torch.tensor(ck["start_positions_angstrom"], dtype=torch.float32, device=device)
        completed = 0
        trajectory = np.lib.format.open_memmap(
            output / "positions.npy", mode="w+", dtype=np.float32, shape=(total + 1, len(pos), 3)
        )
        trajectory[0] = pos.cpu().numpy()

    # --- sigmaスケジュールの構築 ---
    # start_sigmaからsigma_minまで、対数スケールでreverse_steps段階に等分割する
    # (test37_variant_experiment.pyのlangevin ablationにおける
    # geomspace(0.1, 0.003, 20)と同じ発想: 段数を細かく刻むほど、1ステップあたりの
    # variance_drop(=分散の減少量)が小さくなり、歩幅も自然に小さくなる)。
    # 最後にsigma=0のダミー要素を足しておき、最終ステップのnext_sigmaが0になるようにする。
    first_sigma = max(args.start_sigma, args.sigma_min)
    if args.reverse_steps == 1:
        sigma_schedule = torch.tensor([first_sigma, 0.0], device=device)
    else:
        positive_levels = torch.logspace(
            np.log10(first_sigma), np.log10(args.sigma_min), args.reverse_steps, device=device,
        )
        sigma_schedule = torch.cat((positive_levels, positive_levels.new_zeros(1)))
    # 全reverse_stepsのうち、後半deterministic_steps回は確率的更新をやめて
    # 決定論的なDDIM風更新に切り替える(「ノイズありで大まかに→仕上げは決定論的に」という設計)。
    stochastic_steps = args.reverse_steps - args.deterministic_steps
    deadline = time.monotonic() + args.time_budget_hours * 3600

    def save():
        # 生成中の座標(positions.npy)と再開用チェックポイント、進捗JSONを保存する。
        trajectory.flush()
        save_checkpoint(state_path, dict(
            settings=settings, step=completed, positions=pos.detach().cpu(), rng=rng_state(),
        ))
        save_json(output / "generation.json", dict(
            completed_steps=completed, requested_steps=total, valid_frames=completed + 1,
            complete=completed == total, length_unit="angstrom", settings=settings,
            cell_angstrom=np.asarray(cell).tolist(), dataset_metadata=meta,
            is_equilibrium_trajectory=False, scientific_caveat=CAVEAT,
        ))

    # --- 生成(逆拡散)ループ本体 ---
    for step in range(completed, total):
        if STOP or time.monotonic() >= deadline:
            save()
            print("Generation paused; resume with --resume", flush=True)
            return 75
        sigma, next_sigma = sigma_schedule[step], sigma_schedule[step + 1]
        # 現在の座標からグラフを再構築し(distance依存のedge_attrを使うため、
        # 座標が動くたびにグラフを作り直す必要がある)、モデルにt=sigma/sigma_max_trainを渡して
        # 「このノイズレベルにおける、加えられたノイズの予測」を得る。
        data = graph(pos.cpu().numpy(), cell, meta["type_ids"], cutoff, device)
        t = torch.full((1,), float(sigma / sigma_max_train), device=device, dtype=pos.dtype)
        predicted = model(data, t)
        if step < stochastic_steps:
            # Annealed-Langevin / variance-exploding reverse-diffusion step: the
            # step size and injected-noise scale both come directly from how much
            # variance this sigma interval removes, not from a hand-picked constant.
            # variance_dropは「このsigma区間で本来除去されるはずのノイズ分散」。
            # これをsigma**2で割ったscore_stepが、モデル予測に対する重み(歩幅)になる。
            # sigma, next_sigmaが近い(=スケジュールが細かい)ほどscore_stepは小さくなり、
            # legacyサンプラーのような「毎回フルサイズで補正+フルサイズで再ノイズ」を避けられる。
            variance_drop = torch.clamp(sigma.square() - next_sigma.square(), min=0.0)
            score_step = variance_drop / sigma.square()
            pos = pos - score_step * predicted  # モデル予測方向への小さいドリフト(スコアに沿った移動)
            # 揺動散逸的にバランスの取れたノイズを注入する(sqrt(variance_drop)倍)。
            # thermal_scaleでこのノイズの強さを実験的に調整できる。
            pos = pos + torch.randn_like(pos) * torch.sqrt(variance_drop) * args.thermal_scale
        else:
            # DDIM-style deterministic tail for the final polish steps.
            # 最後の仕上げ区間はノイズを加えず、決定論的に少しずつ補正していく。
            ddim_step = 1.0 - next_sigma / sigma
            pos = pos - ddim_step * predicted
        if not torch.isfinite(pos).all():
            raise RuntimeError("Non-finite generated positions")
        completed = step + 1
        trajectory[completed] = pos.cpu().numpy()
        if completed % args.checkpoint_every == 0:
            save()
            print(f"generation {completed}/{total} (sigma={float(sigma):.4f})", flush=True)
    save()
    # 生成が完了したら、最終フレームをase.Atomsに変換してextxyzファイルとしても書き出す。
    atoms = atoms_from_meta(pos.cpu().numpy(), cell, meta)
    atoms.wrap()
    ase.io.write(output / "final.extxyz", atoms)
    print(f"Generated {total + 1} frames: {output}")


def parser():
    # コマンドラインインターフェース定義: "train"と"generate"の2つのサブコマンドを持つ。
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("train", help="sigma-conditioned NequIP_TimeEmbed + RattleParticles(sigma_min, sigma_max) + displacement MSE")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--warm-start", type=Path, default=None,
                   help="Plain (non-time-conditioned) test37 checkpoint to initialize shared weights from")
    p.add_argument("--updates", type=count, default=6000)  # 勾配更新の総回数
    p.add_argument("--batch-size", type=count, default=16)
    p.add_argument("--learning-rate", type=positive, default=2.e-4)
    p.add_argument("--cutoff", type=positive, default=10.0)  # モデルが実際に使うグラフcutoff
    p.add_argument("--large-cutoff", type=positive, default=10.0)  # ノイズを加える前に候補として作っておくcutoff(cutoff以上必須)
    p.add_argument("--sigma-min", type=positive, default=0.001)
    p.add_argument("--sigma-max", type=positive, default=0.75)  # 学習時に使うノイズ幅の上限(生成時のt正規化にも使われる)
    p.add_argument("--validation-fraction", type=positive, default=0.1)
    p.add_argument("--log-every", type=count, default=100)
    p.set_defaults(handler=train)

    p = sub.add_parser("generate", help="annealed-Langevin / variance-exploding reverse-SDE sampler with a DDIM polish tail")
    p.add_argument("--reverse-steps", type=count, default=300)  # sigmaスケジュールの段数(細かいほど1歩あたりの補正が小さくなる)
    p.add_argument("--deterministic-steps", type=nonnegative_count, default=30)  # 末尾何ステップをDDIM風の決定論的更新にするか
    p.add_argument("--start-sigma", type=positive, default=0.75)  # 生成開始時のノイズレベル
    p.add_argument("--sigma-min", type=positive, default=0.001)  # 確率的ステップを終える下限ノイズレベル
    p.add_argument("--thermal-scale", type=positive, default=1.0)  # 注入ノイズの強さを調整する倍率
    p.set_defaults(handler=generate)

    # train/generate共通の引数(出力先・デバイス・乱数シード・時間予算・再開オプションなど)。
    for name, p in sub.choices.items():
        p.add_argument("--output", type=Path, required=True)
        if name == "generate":
            p.add_argument("--checkpoint", type=Path, required=True)
        p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
        p.add_argument("--seed", type=int, default=1337)
        p.add_argument("--time-budget-hours", type=positive, default=11.5)  # この時間を超えたら自動で中断・保存する(HPCジョブの壁時計制限対策)
        p.add_argument("--resume", action="store_true")
        p.add_argument("--checkpoint-every", type=count, default=25)
    return root


def main():
    # SIGTERM/SIGINTを受けたらrequest_stop()でSTOPフラグを立てるようにしてから、
    # 指定されたサブコマンド(train/generate)のハンドラを実行する。
    import signal
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    args = parser().parse_args()
    print(CAVEAT, flush=True)
    return args.handler(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
