from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from pathlib import Path
import os

app = FastAPI(
    title="Wine Quality Prediction API",
    description="预测葡萄酒品质评分 (3-8分)",
    version="1.0"
)

# ---------- 加载模型和 scaler ----------
# 获取当前文件所在目录
BASE_DIR = Path(__file__).resolve().parent

model = None
scaler = None

try:
    model_path = BASE_DIR / "wine_model.keras"
    scaler_path = BASE_DIR / "scaler.pkl"
    
    if model_path.exists() and scaler_path.exists():
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        print("✅ 模型和 Scaler 加载成功！")
    else:
        print(f"⚠️ 文件不存在: model={model_path.exists()}, scaler={scaler_path.exists()}")
except Exception as e:
    print(f"❌ 加载失败: {e}")

# ---------- 定义请求格式 ----------
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

# ---------- 根路径 ----------
@app.get("/")
def home():
    return {
        "message": "🍷 Wine Quality API is running!",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "docs": "/docs"
    }

# ---------- 预测接口 ----------
@app.post("/predict")
def predict(features: WineFeatures):
    # 检查模型是否加载
    if model is None or scaler is None:
        return {
            "status": "error",
            "message": "模型或 Scaler 未加载，请检查服务器文件"
        }
    
    try:
        # 1. 将输入转为 DataFrame
        input_data = pd.DataFrame([features.dict()])
        
        # 2. 标准化
        input_scaled = scaler.transform(input_data)
        
        # 3. 预测（模型输出 6 个类别的概率）
        probabilities = model.predict(input_scaled, verbose=0)
        
        # 4. 取概率最大的类别
        predicted_class = int(np.argmax(probabilities[0]))
        quality_score = predicted_class + 3  # 类别 0-5 → 分数 3-8
        confidence = float(np.max(probabilities[0]))
        
        return {
            "status": "success",
            "quality_score": quality_score,
            "confidence": round(confidence, 4),
            "class_probabilities": {
                f"{i+3}分": round(float(probabilities[0][i]), 4)
                for i in range(6)
            }
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"预测失败: {str(e)}"
        }

# ---------- 健康检查 ----------
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }