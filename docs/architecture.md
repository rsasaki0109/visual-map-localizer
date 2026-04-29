# アーキテクチャ概要

## モジュール構成

```
visual_map_localizer/
├── retrieval/        # NetVLAD などの global descriptor (hloc ラッパー)
├── matching/         # SuperPoint / LightGlue / SuperGlue 抽出・対応点マッチング
├── localization/     # PnP+RANSAC・エンドツーエンドパイプライン
│   ├── pnp.py        # pycolmap (primary) と OpenCV (fallback) の二系統
│   └── pipeline.py   # VisualMapLocalizer
├── mapping/          # build-map (SfM 一括実行)
├── io/               # COLMAP マップ・カメラ intrinsics・出力 JSON
├── cli/              # `visual-map-localizer` Click CLI
└── config.py         # 既定値 / dataclass 設定
```

## データフロー

### build-map

```
images/  ──▶ extract_features (SuperPoint)   ──┐
            extract_features (NetVLAD)         │
            pairs_from_exhaustive | retrieval ─┤──▶ match_features ──▶ reconstruction (COLMAP)
                                               │                          │
                                               └──────── h5 / pairs ──────┘
出力:
  map/
    ├── sfm/                 (COLMAP sparse model)
    ├── features.h5          (SuperPoint)
    ├── global_descriptors.h5 (NetVLAD)
    ├── pairs-sfm.txt
    ├── matches-sfm.h5
    ├── db_images.txt
    └── map_meta.json
```

### localize

```
query.jpg ─▶ NetVLAD (query) ─▶ pairs_from_retrieval ──┐
        ─▶ SuperPoint (query)                         ▼
                            ─▶ LightGlue match  ─▶ QueryLocalizer
                                                    │
                                          db keypoints + COLMAP point3D
                                                    ▼
                                           PnP+RANSAC (pycolmap | OpenCV)
                                                    │
                                                    ▼
                                           LocalizationResult (JSON)
```

## クラス設計

| クラス / 関数 | 役割 |
|---|---|
| `VisualMapLocalizer` | 構築済みマップを読込み、`.localize(query)` で 6DoF を返す |
| `ColmapMap` | `pycolmap.Reconstruction` のラッパー (image / camera / point3D 索引) |
| `LocalizationResult` | JSON 互換 dataclass。CLI 出力スキーマ |
| `MappingConfig` / `LocalizeConfig` | パラメータ束 (CLI 引数のミラー) |
| `estimate_pose()` | pycolmap → OpenCV の自動フォールバック付き PnP |

## 拡張の指針

- **新しい local feature**: `matching.AVAILABLE_LOCAL` に追加し、hloc 側に config が存在することを確認。
- **新しい global descriptor**: `retrieval.AVAILABLE_GLOBAL` に追加し、CLI からは `--global-descriptor` で指定可能。
- **新しい matcher**: `matching.matcher.resolve_matcher_conf` の alias テーブルでマップ。
- **大規模化 (sharding / ANN)**: `retrieval/` 配下に Faiss 等のラッパーを追加。`pairs_from_retrieval` は同 API のまま差し替え可能。
- **VIO/IMU 融合**: `localization/pipeline.py` の出力 `LocalizationResult` を観測モデルとして取り、上位の EKF に渡す。
