# streamlit_app.py
import streamlit as st
import requests
import json

st.set_page_config(page_title="🍷 AI 葡萄酒品鉴助手", page_icon="🍷", layout="wide")
st.title("🍷 AI 葡萄酒品鉴助手")
st.markdown("输入葡萄酒的理化指标，AI 将调用工具预测品质评分")

# 输入区域
with st.form("wine_form"):
    col1, col2 = st.columns(2)
    with col1:
        fixed_acidity = st.number_input("固定酸度", value=7.4, step=0.1)
        volatile_acidity = st.number_input("挥发性酸度", value=0.7, step=0.01)
        citric_acid = st.number_input("柠檬酸", value=0.0, step=0.01)
        residual_sugar = st.number_input("残糖", value=1.9, step=0.1)
        chlorides = st.number_input("氯化物", value=0.076, step=0.001)
        free_sulfur_dioxide = st.number_input("游离二氧化硫", value=11.0, step=1.0)
    with col2:
        total_sulfur_dioxide = st.number_input("总二氧化硫", value=34.0, step=1.0)
        density = st.number_input("密度", value=0.9978, step=0.0001)
        pH = st.number_input("pH值", value=3.51, step=0.01)
        sulphates = st.number_input("硫酸盐", value=0.56, step=0.01)
        alcohol = st.number_input("酒精含量", value=9.4, step=0.1)
    
    submitted = st.form_submit_button("🔍 预测品质")

if submitted:
    with st.spinner("正在调用 AI 模型预测..."):
        try:
            # 调用本地 API
            response = requests.post(
                "http://localhost:8000/predict",
                json={
                    "fixed_acidity": fixed_acidity,
                    "volatile_acidity": volatile_acidity,
                    "citric_acid": citric_acid,
                    "residual_sugar": residual_sugar,
                    "chlorides": chlorides,
                    "free_sulfur_dioxide": free_sulfur_dioxide,
                    "total_sulfur_dioxide": total_sulfur_dioxide,
                    "density": density,
                    "pH": pH,
                    "sulphates": sulphates,
                    "alcohol": alcohol
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    score = result.get("quality_score")
                    confidence = result.get("confidence", 0)
                    st.success(f"🍷 **预测品质评分: {score} / 8 分**")
                    st.metric("置信度", f"{confidence:.2%}")
                else:
                    st.error(f"预测失败: {result.get('message', '未知错误')}")
            else:
                st.error(f"API 请求失败，状态码: {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("❌ 无法连接到 API 服务！请确保已运行 `uvicorn main:app --reload`")
        except Exception as e:
            st.error(f"发生错误: {str(e)}")