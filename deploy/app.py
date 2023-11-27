from fastapi import FastAPI, File, UploadFile
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.io import imread
import joblib

app = FastAPI()

# SVM 모델을 불러오거나 학습한 후 사용합니다.
# 여기에서는 모델을 불러오는 것으로 가정합니다.
# model = svm.load_model('svm_model.pkl')

model = joblib.load("rf_stroke_classification")

@app.post("/classify/")
async def classify_image(file: UploadFile):
    # 이미지 업로드 및 처리
    # image = Image.open(file.file)
    # image = image.resize((64, 64))  # 이미지 크기 조정 (모델에 맞게)
    
    img_array = imread(file.file)
    img_gray = rgb2gray(img_array)
    img_resized = resize(img_gray, (150, 150)) 
    flat_data = img_resized.flatten().reshape(1, -1)

    # 이미지를 NumPy 배열로 변환
    #image_array = np.array(image)

    # SVM 모델을 사용하여 이미지 분류
    result = model.predict(flat_data)
    probability = model.predict_proba(flat_data)
    result_list = result.tolist()

    # 분류 결과 반환
    return {
        'prediction': result_list[0],
    'probability_0': probability[0][0],
    'probability_1': probability[0][1]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
