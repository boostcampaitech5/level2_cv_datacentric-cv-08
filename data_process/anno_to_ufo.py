import json
import os
from typing import Optional, Union, List
from abc import abstractmethod
from argparse import ArgumentParser
from typing import Any


class BaseLabelConverter():
    """
    annotation 파일을 UFO 형식으로 바꿔주는 base class입니다.
    
    annotation 파일 형식에 따라 `anno_filename`, `anno_img_width`, 
    `anno_img_heigth`, `anno_bboxes`의 구현이 필요합니다.
    
    Attributes:
        json_path: 변환할 annotation 파일 path
        anno_ufo: UFO 형식으로 변환된 annotation 객체
        anno_words: UFO 형식의 word 객체
    """
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.anno_ufo = dict()
        self.anno_words = dict()
        
        with open(self.json_path, 'r') as f:
            self.anno_obj = json.load(f)
    
    @property
    @abstractmethod
    def anno_filename(self) -> str:
        """원본 annotation의 파일명을 반환합니다.
        
        str 형식으로 반환해야 합니다.

        Raises:
            NotImplementedError: 구현되지 않음
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def anno_img_width(self) -> int:
        """원본 annotation 이미지의 넓이를 반환합니다.

        int 형식으로 반환해야 합니다.
        
        Raises:
            NotImplementedError: 구현되지 않음
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def anno_img_height(self) -> int:
        """원본 annotation 이미지의 높이를 반환합니다.

        int 형식으로 반환해야 합니다.
        
        Raises:
            NotImplementedError: 구현되지 않음
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def anno_bboxes(self):
        """원본 annotation으로부터 UFO points(bboxes)를 생성합니다.

        원본 annotation의 모든 bbox로부터 id, transcription, points를
        불러온 뒤, `_construct_word`를 이용해 `self.anno_words`를 만들어줍니다.
        
        `self.anno_words`를 반환하거나, 이와 동일한 형태의 json object를 반환해야 합니다.
        
        Raises:
            NotImplementedError: 구현되지 않음
        """
        raise NotImplementedError
    
    def _construct_word(self, id: Union[int, str], transcription: str, points: List[List[int]]) -> None:
        """id, transcription, points를 받아 UFO 형식에서 사용할 word 객체를 만듭니다.
        
        받은 parameter를 이용해 `self.anno_words`에 id를 키로 가지는 word 객체를 저장합니다.

        Args:
            id (Union[int, str]): bbox id
            transcription (str): bbox에 포함된 글자
            points (List[List[int]]): bbox 좌표. [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] 형태.
        """
        if type(id) is int:
            id = str(id).zfill(4)
            
        self.anno_words[id] = {
            "transcription": transcription,
            "points": points,
            "orientation": "Horizontal",
            "language": [
                "ko"
            ],
            "tags": [
                "Auto"
            ],
            "confidence": None,
            "illegibility": False
        }
    
    def convert(self, anno_holder: Optional[str] = None) -> None:
        """변환된 annotation 객체를 `self.anno_ufo`에 저장합니다.

        Args:
            anno_holder (Optional[str], optional): Annotation holder. Defaults to None.
        """
        self.anno_ufo[self.anno_filename] = {
            "paragraphs": {},
            "words": self.anno_bboxes,
            "chars": {},
            "img_w": self.anno_img_width,
            "img_h": self.anno_img_height,
            "tags": ["autoannotated"],
            "relations": {},
            "annotation_log": {
                "worker": "worker",
                "timestamp": "2023-03-22",
                "tool_version": "",
                "source": None
            },
            "license_tag": {
                "usability": True,
                "public": True,
                "commercial": True,
                "type": None,
                "holder": anno_holder
            }
        }
        
    def save_annotation(self, target_dir="."):
        """변환된 annotation 객체를 저장합니다.

        Args:
            target_dir (str, optional): annotation을 저장할 위치. Defaults to ".".
        """
        save_name = self.json_path.split('/')[-1]
        save_name = os.path.join(target_dir, save_name)
        with open(save_name, 'w') as f:
            json.dump({"images": self.anno_ufo}, f, indent=4, ensure_ascii=False)
    

class AIHubLabelConverter(BaseLabelConverter):
    """AI Hub 공공행정문서 OCR 데이터셋의 annotation파일을 UFO 형식으로 변환합니다.
    """
    @property
    def anno_filename(self):
        return self.anno_obj['images'][0]['image.file.name']
    
    @property
    def anno_img_width(self):
        return self.anno_obj['images'][0]['image.width']
    
    @property
    def anno_img_height(self):
        return self.anno_obj['images'][0]['image.height']
    
    @property
    def anno_bboxes(self):
        anno_words = self.anno_obj['annotations']
        for word in anno_words:
            id = str(word['id'] + 1).zfill(4)
            text = word['annotation.text']
            x, y, w, h = [float(num) for num in word['annotation.bbox']]
            points = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
            self._construct_word(id, text, points)
        return self.anno_words

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--data_path', type=str,
                        default='../../data/New_sample/labels/', help="A json file or directory to convert annotation file to UFO format. If directory is given, converts every json file in given directory.")
    parser.add_argument('--target_path', type=str, default='../../data/New_sample/new_labels/', help="target directory to save converted annotation file.")

    args = parser.parse_args()

    return args

def main(args):
    json_path = args.data_path
    target_dir = args.target_path
    
    if not os.path.isdir(json_path):
        json_path = [json_path]
    else:
        json_path = [json_path + json_file for json_file in os.listdir(json_path)]
    
    for json_file in json_path:
        label_converter = AIHubLabelConverter(json_file)
        label_converter.convert()
        label_converter.save_annotation(target_dir)

if __name__ == '__main__':
    args = parse_args()
    main(args)