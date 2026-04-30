# visual-map-localizer

[![CI](https://github.com/rsasaki0109/visual-map-localizer/actions/workflows/ci.yml/badge.svg)](https://github.com/rsasaki0109/visual-map-localizer/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)

COLMAP で構築した SfM 地図に対して、**1 枚の query 画像から 6DoF カメラ姿勢を推定する**
Visual Positioning System (VPS) 実装です。
内部では [hloc (Hierarchical-Localization)](https://github.com/cvg/Hierarchical-Localization)
の SuperPoint / LightGlue / NetVLAD パイプラインを薄くラップしつつ、CLI と
クリーンな Python API を被せています。

```
query.jpg ─▶ NetVLAD top-K 検索 ─▶ SuperPoint+LightGlue マッチング
                                  ─▶ 2D-3D 対応構築 ─▶ PnP+RANSAC
                                  ─▶ 6DoF pose (R, t)
```

## 特徴

* **CLI ファースト**: `visual-map-localizer build-map` / `... localize`
* **モジュール分割**: `retrieval / matching / localization / mapping / io / cli`
* **GPU あり/なし両対応** (CPU でも動くが、推論は遅め)
* **JSON 出力**: pose・inlier 数・reprojection error・retrieval Top-K
* **PnP は pycolmap が第 1 選択 / OpenCV solvePnPRansac が fallback**
* 将来 ROS2 ノード化することを前提とした設計 (`docs/ros2_integration.md`)

## インストール

### 軽量インストール (テスト・PnP・JSON I/O だけ使う場合)

`torch` / `hloc` を入れずに済むので、小さい CI 環境などに最適です。
`visual_map_localizer` は PEP 562 lazy import で深層学習依存を遅延ロード
するため、Retrieval / Matching を呼ばない限りこの構成でも動きます。

```bash
pip install -e .
```

### フルインストール (実際に Map を作って Localize する場合)

#### 1. PyTorch を先に CUDA に合わせて入れる

```bash
# 例: CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 2. 本パッケージ + deep extras

```bash
pip install -e .[deep]
```

#### 3. hloc を git からインストール (PyPI 未公開)

```bash
pip install git+https://github.com/cvg/Hierarchical-Localization.git@master
```

#### 4. COLMAP の動作確認

`pycolmap` は wheel 同梱のためバイナリは別途不要です。サンプルが動くか:

```bash
python -c "import pycolmap; print(pycolmap.__version__)"
python -c "from hloc import extract_features; print(list(extract_features.confs)[:5])"
```

## 使い方 (CLI)

### マップ構築

```bash
# 既定 (SuperPoint + LightGlue) — SuperGluePretrainedNetwork submodule が必要
visual-map-localizer build-map \
    --images ./images \
    --output ./map

# Apache-2.0 構成 (third-party submodule 不要、商用利用可)
visual-map-localizer build-map \
    --images ./images \
    --output ./map \
    --local-feature disk \
    --matcher disk+lightglue
```

> 補足: 既定の SuperPoint は hloc が `third_party/SuperGluePretrainedNetwork`
> サブモジュールから読み込むため、hloc を git clone する際は
> `git clone --recursive https://github.com/cvg/Hierarchical-Localization.git`
> としてください。pip 経由で hloc を入れた場合はサブモジュールが含まれないため、
> 上記の DISK + LightGlue 構成のほうが確実に動きます。

主なオプション:

| Option | 既定 | 説明 |
|---|---|---|
| `--local-feature` | `superpoint_aachen` | hloc の local feature config 名 |
| `--global-descriptor` | `netvlad` | hloc の global descriptor config 名 |
| `--matcher` | `superpoint+lightglue` | hloc の matcher config 名 |
| `--num-covisible-pairs` | `None` (exhaustive) | 指定すると retrieval-based pairs (大規模向け) |
| `--overwrite` | off | 中間ファイルを再生成 |

出力ディレクトリ構造:

```
map/
├── sfm/                       # COLMAP sparse model
├── features.h5                # SuperPoint
├── global_descriptors.h5      # NetVLAD
├── pairs-sfm.txt
├── matches-sfm.h5
├── db_images.txt
└── map_meta.json
```

### 1 枚の画像をローカライズ

```bash
visual-map-localizer localize \
    --map ./map \
    --query query.jpg
```

> localize 側は `map_meta.json` を読んで build-map 時と同じ
> `local-feature` / `global-descriptor` / `matcher` を自動採用します。
> 明示指定したい場合のみ CLI オプションで上書きしてください。

主なオプション:

| Option | 既定 | 説明 |
|---|---|---|
| `--top-k` | 10 | retrieval 候補数 |
| `--ransac-max-error` | 12.0 | RANSAC reprojection threshold (px) |
| `--min-inliers` | 12 | これ未満なら success=false |
| `--matcher` | `superpoint+lightglue` | localize 時の matcher |
| `--camera-model` | (推定) | 例: `PINHOLE` (params も必須) |
| `--camera-params` | (推定) | カンマ区切り (例: `fx,fy,cx,cy`) |
| `--image-size` | (画像から取得) | `--camera-model` 指定時に必要 |
| `--output` | stdout | JSON を書き出すパス |

成功時の JSON:

```json
{
  "success": true,
  "pose": {
    "R": [[0.999, 0.034, 0.012], [-0.034, 0.999, 0.005], [-0.012, -0.005, 1.000]],
    "t": [1.234, -0.567, 5.890],
    "qvec": [0.9999, 0.017, 0.006, 0.001]
  },
  "inliers": 128,
  "reproj_error": 0.91,
  "num_matches": 612,
  "retrieval": ["db/0001.jpg", "db/0007.jpg", "db/0042.jpg"],
  "query": "/abs/path/query.jpg",
  "timing": {"total": 1.42, "pnp": 0.03, "matching": 0.21}
}
```

失敗時:

```json
{
  "success": false,
  "error": "PnP failed",
  "retrieval": ["db/0001.jpg", "..."],
  "num_matches": 4,
  "query": "/abs/path/query.jpg",
  "timing": {"total": 0.97}
}
```

CLI の終了コード:
* `0` 成功
* `2` ローカライズ失敗 (画像 / マップは正常に読めた)
* `1` 引数 / IO エラー

### マップの中身を確認

```bash
visual-map-localizer inspect --map ./map
```

## Python API

```python
from visual_map_localizer import VisualMapLocalizer
from visual_map_localizer.config import LocalizeConfig

localizer = VisualMapLocalizer(
    "./map",
    config=LocalizeConfig(top_k=15, ransac_max_error_px=8.0),
)
result = localizer.localize("query.jpg")
print(result.success, result.inliers)
print(result.to_json())
```

### np.ndarray を直接渡す (ROS 統合・常駐サーバ向け)

```python
import numpy as np
from PIL import Image
import pycolmap

from visual_map_localizer import VisualMapLocalizer
from visual_map_localizer.config import LocalizeConfig

localizer = VisualMapLocalizer("./map", config=LocalizeConfig(top_k=10))

# どこかのストリーム / cv_bridge / カメラ SDK から:
rgb = np.asarray(Image.open("query.jpg").convert("RGB"))  # H×W×3 uint8 (RGB)

camera = pycolmap.Camera(model="PINHOLE", width=rgb.shape[1], height=rgb.shape[0],
                         params=[fx, fy, cx, cy])
result = localizer.localize(rgb, camera=camera, name="frame_0123.png")
print(result.inliers, result.pose["t"])
```

`localize()` は `Path` でも `np.ndarray` でも受け取ります。ndarray の場合は
`camera` 必須（EXIF が無いので）。`name` は省略可能で、内部キャッシュキーになります。

### マップ構築 (Python API)

```python
from visual_map_localizer.mapping import build_map
from visual_map_localizer.config import MappingConfig

build_map(
    image_dir="./images",
    output_dir="./map",
    config=MappingConfig(num_covisible_pairs=20),
)
```

## レイテンシ目安

south-building (118 db imgs, 3072×2304 query, GPU, DISK+LightGlue+NetVLAD):

| 起動方法 | 初回 (warmup 込) | 2 回目以降 (steady-state) |
|---|---|---|
| `python3 -m visual_map_localizer.cli.main localize ...` (毎回 subprocess) | — | **8.6 s / query** |
| `VisualMapLocalizer` インスタンス使い回し (path) | 4.1 s | **1.1 s / query** |
| `VisualMapLocalizer` インスタンス使い回し (ndarray) | 4.4 s | **1.4 s / query** |

ndarray 版の +0.3s は PNG エンコードが主因。ROS 系の **1280×720** クラスならエンコードが 0.03s 程度に縮むので、サブセカンドが現実的です。
詳細プロファイルは `scripts/profile_localize.py` を参照。

## サンプル

* `examples/build_map_example.py`
* `examples/localize_example.py`
* `scripts/download_example_data.sh` (placeholder — 推奨データセットの案内のみ)
* `scripts/evaluate_south_building.py` — 公開データセット検証用の Sim(3) 整列 + pose 誤差評価

## 公開データセット検証 (south-building, 128 枚)

COLMAP 公式の `south-building` データセット (Schönberger 氏配布、reference SfM 同梱) で
end-to-end 検証した結果です。128 枚を 118 db / 10 query に分割し、118 枚で
build-map → 10 枚を順次 localize → 同梱 reference SfM との pose 誤差を Sim(3) 整列で評価:

| 指標 | 値 |
|---|---|
| 成功率 | **10 / 10** |
| inlier 数 | 2616〜5229 |
| reprojection error | 1.25〜2.99 px |
| 1 query あたり所要 (CPU/GPU 込) | 約 8.6 s |
| **回転誤差** | median **0.066°**, mean 0.074°, max 0.174° |
| **並進誤差** | median **0.0034**, max 0.0046 (シーン全幅 10.81、つまり 0.03〜0.04%) |
| 自前 SfM と reference SfM の整列残差 | mean 0.0035 (= 0.03% of scene scale) |

つまり、localizer が出す pose は **SfM 自身の数値ノイズと同オーダー** で
reference SfM と一致しています。

再現手順:

```bash
# 1) データ取得
mkdir -p /tmp/vml-public && cd /tmp/vml-public
curl -L -o south-building.zip \
  https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip
unzip -q south-building.zip

# 2) 118 / 10 に分割 (seed 固定)
python3 - <<'PY'
import random, shutil
from pathlib import Path
src = Path('/tmp/vml-public/south-building/images')
imgs = sorted(p.name for p in src.iterdir() if p.suffix.lower() == '.jpg')
qs = sorted(random.Random(42).sample(imgs, 10))
db = [n for n in imgs if n not in qs]
Path('/tmp/vml-public/db_images').mkdir(exist_ok=True)
Path('/tmp/vml-public/query_images').mkdir(exist_ok=True)
for n in db: shutil.copy(src/n, Path('/tmp/vml-public/db_images')/n)
for n in qs: shutil.copy(src/n, Path('/tmp/vml-public/query_images')/n)
PY

# 3) build-map (118 imgs, retrieval-based pairs)
visual-map-localizer build-map \
    --images /tmp/vml-public/db_images \
    --output /tmp/vml-public/map \
    --local-feature disk --matcher disk+lightglue \
    --num-covisible-pairs 20

# 4) 10 queries を localize (intrinsics は reference SfM のものをそのまま指定)
mkdir -p /tmp/vml-public/results
for q in /tmp/vml-public/query_images/*.JPG; do
    visual-map-localizer localize \
        --map /tmp/vml-public/map --query "$q" \
        --camera-model SIMPLE_RADIAL \
        --camera-params 2559.68,1536,1152,-0.0204997 \
        --image-size 3072x2304 \
        --output /tmp/vml-public/results/$(basename "$q" .JPG).json
done

# 5) reference SfM と比較
python3 scripts/evaluate_south_building.py \
    --dataset /tmp/vml-public/south-building \
    --map-dir /tmp/vml-public/map \
    --results-dir /tmp/vml-public/results
```

## 制約 / 注意

詳細は [`docs/limitations.md`](docs/limitations.md) 参照。

* 屋外の **強い照度変化 / 季節変化** には弱い (NetVLAD + SuperPoint の限界)
* `--camera-model` を指定しない場合は EXIF / 60° FoV からの推定にフォールバック
* SfM が成立する程度の画像 overlap が必要 (経験則: 隣接で 60% 以上)
* SuperPoint / SuperGlue は研究用ライセンス。商用は DISK / LightGlue を推奨

## ROS2 統合

ROS2 (Jazzy 想定) ノードは `ros2/visual_map_localizer_ros/` 配下にあります。
セットアップ・パラメータ詳細は [`ros2/visual_map_localizer_ros/README.md`](ros2/visual_map_localizer_ros/README.md) を参照。

```bash
# (build-map で作ったマップに対して)
ros2 launch visual_map_localizer_ros vps.launch.py \
    map_dir:=/abs/path/to/map \
    publish_tf:=true
```

* sub: `/camera/image_raw` (sensor_msgs/Image) + `/camera/camera_info` (CameraInfo)
* pub: `/vps_pose` (geometry_msgs/PoseWithCovarianceStamped) ※ ROS 慣例 world-from-camera
* TF: `frame_id → child_frame_id` (`publish_tf:=true`)
* in-flight 中の画像は drop (1Hz 級の絶対姿勢源として利用)

## アーキテクチャ / 拡張

* [`docs/architecture.md`](docs/architecture.md) — モジュール構成 / データフロー
* [`docs/ros2_integration.md`](docs/ros2_integration.md) — 設計ノート (実装は `ros2/`)
* 大規模地図対応 (sharding / Faiss ANN) はアーキテクチャドキュメントに記載

## 開発

```bash
pip install -e ".[dev]"
pytest -q
```

* `tests/test_pnp.py` は pycolmap / OpenCV があれば実行されます。
* `tests/test_imports.py` は重い依存なしで通る import スモークテストです。

## ライセンス

Apache-2.0 (本リポジトリ)。
内部利用するモデルの重み (SuperPoint / SuperGlue / NetVLAD) は **各々のライセンスに従う必要** があります。
