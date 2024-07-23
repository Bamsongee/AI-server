import os
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import torch
from yolov5 import detect
import cv2
from PIL import Image

# NotImplementedError: cannot instantiate 'PosixPath' on your system 해결
from pathlib import Path
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)

FoodList = {
    "kochujang": "고추장", "apple": "사과", "avocado": "아보카도", "bacon": "베이컨", "banana": "바나나", 
    "beef": "소고기", "bread": "빵", "burdock": "우엉", "butter": "버터", "cabbage": "양배추", 
    "canned_corn": "옥수수캔", "canned_tuna": "참치캔", "carrot": "당근", "cheese": "치즈", 
    "chicken": "닭고기", "chili_powder": "고춧가루", "chocolate_bread": "초콜릿 빵", "cinnamon": "계피", 
    "cooking_oil": "식용유", "corn": "옥수수", "cornflake": "콘플레이크", "crab_meat": "게살", 
    "cucumber": "오이", "curry_powder": "카레 가루", "dumpling": "만두", "egg": "계란", 
    "fish_cake": "어묵", "french_fries": "감자튀김", "garlic": "마늘", "ginger": "생강", 
    "green_onion": "대파", "ham": "햄", "hash_brown": "해쉬 브라운", "hotdog": "핫도그", 
    "ice": "얼음", "ketchup": "케첩", "kimchi": "김치", "lemon": "레몬", "lemon_juice": "레몬 주스", 
    "mandarin": "귤", "marshmallow": "마시멜로", "mayonnaise": "마요네즈", "milk": "우유", 
    "mozzarella cheese": "모짜렐라 치즈", "mushroom": "버섯", "mustard": "머스타드", 
    "nacho_chips": "나초 칩", "noodle": "국수", "nutella": "누텔라", "olive_oil": "올리브 오일", 
    "onion": "양파", "oreo": "오레오", "parmesan_cheese": "파르메산 치즈", "parsley": "파슬리", 
    "pasta": "파스타", "peanut_butter": "땅콩버터", "pear": "배", "pepper": "후추", 
    "pepper_powder": "고추가루", "pickle": "피클", "pickled_radish": "단무지", "pimento": "피망", 
    "pineapple": "파인애플", "pork": "돼지고기", "potato": "감자", "ramen": "라면", "red_wine": "레드 와인", 
    "rice": "쌀", "salt": "소금", "sausage": "소시지", "seaweed": "김", "sesame": "참깨", 
    "sesame_oil": "참기름", "shrimp_paste": "새우젓", "soy_sauce": "간장", "spam": "스팸", 
    "squid": "오징어", "strawberry": "딸기", "sugar": "설탕", "sweet_potato": "고구마", "tofu": "두부", 
    "tomato": "토마토", "wasabi": "와사비", "watermelon": "수박", "whipping_cream": "휘핑크림"
}

def transform_labels(translated_labels):
    # 중복 제거
    unique_labels = list(set(translated_labels))
    transformed_labels = [FoodList.get(label, label) for label in unique_labels]
    return transformed_labels

def save_results_image(image_path, results, save_path):
    image = cv2.imread(image_path)
    for box in results.pred[0]:
        x1, y1, x2, y2, conf, cls = box
        label = results.names[int(cls)]
        color = (255, 0, 0)  # 파란색으로 인식한 객체 출력
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    cv2.imwrite(save_path, image)

@app.route('/detection', methods=['GET', 'POST'])
def predict():
    output_labels = []
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)

        os.makedirs('./input_dir', exist_ok=True)
        file.save(os.path.join('./input_dir', filename))
        train_img = './input_dir/' + file.filename

        weights = '.\\yolov5\\runs\\train\\exp6\\weights\\best.pt'  # YOLOv5 모델 가중치 경로
        img_size = 640  # 이미지 사이즈

        # YOLOv5 모델 로드
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)
        model.eval()

        # 이미지 예측
        results = model(train_img, size=img_size)

        # 예측된 결과 처리
        if results and results.pred[0] is not None:
            for *box, conf, cls in results.pred[0]:
                output_labels.append(model.names[int(cls)])

            translated_labels = transform_labels(output_labels)

            # 결과 이미지 파일로 저장
            os.makedirs('./output_dir', exist_ok=True)
            save_image_path = os.path.join('./output_dir', filename)
            save_results_image(train_img, results, save_image_path)

            return jsonify(translated_labels), 200
        else:
            return jsonify({"error": "No detections made"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0')
