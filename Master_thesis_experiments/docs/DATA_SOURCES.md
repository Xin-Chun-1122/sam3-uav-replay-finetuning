# Official data and model sources

Large raw archives and model weights live under `datasets/raw/` and `models/` but are excluded from Git. Their exact file hashes are recorded after download.

| Resource | Official source | Local target |
|---|---|---|
| VisDrone2019-MOT test-dev | https://github.com/VisDrone/VisDrone-Dataset | `datasets/raw/VisDrone2019-MOT-test-dev/` |
| RefDrone test | https://huggingface.co/datasets/sunzc-sunny/RefDrone | `datasets/raw/RefDrone/` |
| YOLOv8x-WorldV2 | https://docs.ultralytics.com/models/yolo-world/ | `models/yolov8x-worldv2.pt` |

VisDrone-MOT test-dev and RefDrone test include public ground-truth annotations. Do not replace them with challenge splits whose annotations are withheld.

