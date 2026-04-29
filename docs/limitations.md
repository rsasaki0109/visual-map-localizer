# 既知の制約・注意事項

## 屋外 / 屋内 / 季節変化
* COLMAP + SuperPoint + LightGlue という組み合わせは、構築時と問い合わせ時の
  画角・照度差がある程度であれば堅牢。だが:
  * **強い照度変化 (昼/夜)**: 失敗率が顕著に上がる。NetVLAD だと特に夜間 retrieval 弱い。
  * **季節変化 (緑/雪)**: 樹木中心のシーンで失敗が増える。
  * **動的物体 (歩行者・車)**: 多すぎると inlier 不足になりやすい。
* 屋外は GPS / Compass を粗い prior として併用するのが安全。
* 屋内は重複度が高くなるよう、十分なオーバーラップで撮影することが前提。

## カメラ intrinsics
* `--camera-model / --camera-params` を渡さない場合は EXIF + 60° FoV のフォールバックで推定。
* iPhone など EXIF が信頼できる端末ではほぼ問題ないが、トリミング済み画像や
  リサイズ後画像では誤った intrinsics になりやすい。**できるだけ明示する**。

## SfM の頑健性
* `build-map` は COLMAP の incremental mapper をそのまま使う:
  * 画像枚数が極端に少ない (< 10) 場合は 2 view しか復元できないことがある。
  * 画像同士の overlap が小さいと部分集合ごとに分断されたモデルが返る。
  * 大規模 (>1000 枚) では `--num-covisible-pairs` 指定が事実上必須。

## 計算資源
* GPU 必須ではないが、CPU のみだと SuperPoint / LightGlue は重い。
  数百枚規模の地図構築は数十分〜数時間かかる。
* `pycolmap` の PnP は CPU のみ。クエリ単発レイテンシのボトルネックは
  通常 LightGlue マッチング (GPU 数十 ms / CPU 数百 ms)。

## モデルライセンス
* SuperPoint / SuperGlue は **非商用** ライセンス。商用利用するなら
  DISK / LightGlue (Apache-2.0) や ALIKED 等の組み合わせを検討する。
* NetVLAD の事前学習重みも研究用に近い扱い。代替として OpenIBL や EigenPlaces。
