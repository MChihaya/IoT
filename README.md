# Collaborative On-Device Learning at the Edge<br><sub>協調オンデバイス学習：デバイス間で「何を共有するか」</sub>
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
