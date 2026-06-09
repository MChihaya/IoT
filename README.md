# Collaborative On-Device Learning at the Edge<br><sub>協調オンデバイス学習：デバイス間で「何を共有するか」</sub>

AIネイティブなエッジ機器どうしが協調して学習するとき、**重み**を共有するか **予測（ロジット）** を
共有するかを、自作の小規模実験で比較します。深層学習フレームワークは使わず **NumPy のみ** で実装し、
通信量を 1 バイト単位で計上します。

> 主張：予測共有（連合蒸留）は、重み共有（FedAvg / ゴシップ）に対して通信量がモデルサイズに非依存で、
> 異種アーキテクチャ混在でも動作する。

## 実験の内容
- 10 台の機器を模擬：ラベル非IID（Dirichlet α=0.3）＋ 機器ごとのセンサ・ドメインシフト。
- 比較する方式：`Local`（孤立）/ `Central`（全データ集約・上限）/ `FedAvg`（重み共有）/
  `Gossip`（分散重み共有）/ `FedDistill`（予測共有・連合蒸留）。
- 評価：各機器の自環境での精度（local）と、正準分布での汎化精度（global）、および総通信量。

## 必要環境
- [uv](https://docs.astral.sh/uv/)（Python 環境とパッケージの管理）。Python 3.10 以上。
- 依存パッケージ（`pyproject.toml` に記載）：numpy / scipy / scikit-learn / matplotlib。
- データセットは scikit-learn 同梱の手書き数字（ダウンロード不要）。

## 実行方法
```bash
# 初回は自動で仮想環境を作成し、依存をインストールしてから実行されます（約30秒, CPU）
uv run python fl_edge_experiment.py
```
明示的に環境を作る場合：
```bash
uv sync                               # .venv を作成し依存をインストール
uv run python fl_edge_experiment.py
```

## 出力
- `figures/` … 図 9 点（収束曲線、精度対通信量、スケーリング、異種構成、量子化 ほか）
- `results.json` … 全方式の最終精度・通信量などの数値

（どちらも実行時に生成され、リポジトリには含めません。）

## License
MIT
