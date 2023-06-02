# Label Data processing
Label data(annotation file)를 UFO format으로 변환하고, 분리된 UFO format 파일을 하나로 합치는 script를 제공합니다.

* `aihub_movefile.py` : AI Hub 공공행정문서 OCR 데이터셋에서 분리되어있던 label, image 폴더 및 파일들을 하나의 폴더에 포함되도록 만들어줍니다.
  * 실행방법
  ```bash
  python aihub_movefile.py --data_dir "data/New_sample"
  ```
  * `라벨링데이터`, `원천데이터` 폴더가 포함되어있는 디렉토리를 지정해줍니다.
  * 실행 시, labels내에 annotation 파일들을, images내에 image 파일들을 옮깁니다.
* `anno_to_ufo.py` : 다른 형식의 annotation 파일을 UFO 형식으로 변환해줍니다.
  * 실행방법
  ```bash
  python anno_to_ufo.py --data_path "data/New_sample/labels" --target_path "data/New_sample/new_labels/
  ```
  * `data_path`에는 단일 json파일 또는 여러 json 파일들이 포함된 폴더를 지정할 수 있습니다. 폴더 지정 시, 폴더 내의 모든 json파일을 UFO 형식으로 바꿉니다.
  * `target_path`에는 변환된 UFO 파일이 저장될 위치를 지정합니다.
  * 변환 방식은 AI Hub 공공행정문서 OCR만 지원합니다. 다른 변환을 구현하려면 `labelconverter.py`를 참조해주세요.
* `ufo_merge.py` : 여러개의 UFO format annotation 파일을 하나로 만들어줍니다.
  * 실행방법
  ```bash
  python ufo_merge.py data/New_sample/new_labels/ data/medical/ufo/train.json --target_path data/New_sample/ufo
  ```
  * argument로 합칠 대상이 되는 UFO format json 파일 또는 폴더를 지정해줍니다. 폴더 지정 시, 폴더 내의 모든 UFO format json 폴더를 하나로 합칩니다. 여러 개의 파일, 여러 개의 폴더, 또는 여러 개의 파일 및 폴더 모두 지정 가능합니다.
  * `target_path`에는 합쳐진 UFO 파일이 저장될 위치를 지정합니다.
  