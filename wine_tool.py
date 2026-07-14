# wine_tool.py
import requests
import json
from langchain.tools import tool

@tool
def predict_wine_quality(input_data: str) -> str:
    """
    根据葡萄酒的11项理化指标预测其品质评分。
    当用户询问葡萄酒品质、质量评估、红酒或白酒好不好喝、评分多少时，应该使用此工具。
    输入必须是一个包含所有11个键值对的JSON字符串。
    返回值为3-8分的质量评分及置信度。
    """
    # 1. 尝试解析输入（可能是JSON字符串，也可能是已被解析的字典）
    try:
        # 如果输入已经是字典，直接使用
        if isinstance(input_data, dict):
            data = input_data
        else:
            # 否则尝试解析为JSON
            data = json.loads(input_data)
    except json.JSONDecodeError:
        return f"错误：无法解析输入数据，请确保传入的是有效的JSON格式。收到内容：{input_data}"

    # 2. 检查是否包含所有必需的字段
    required_keys = [
        "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
        "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
        "pH", "sulphates", "alcohol"
    ]
    
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        return f"错误：缺少以下必需字段：{', '.join(missing_keys)}"

    # 3. 构建请求
    api_url = "http://localhost:8000/predict"  # 本地测试就用这个
    
    # 确保所有数值都是float类型
    payload = {key: float(data[key]) for key in required_keys}
    
    try:
        response = requests.post(api_url, json=payload, timeout=10)
        result = response.json()
        
        if result.get("status") == "success":
            score = result.get("quality_score")
            confidence = result.get("confidence", "N/A")
            return f"预测品质评分: {score}分，置信度: {confidence}"
        else:
            return f"预测服务返回错误: {result.get('message', '未知错误')}"
            
    except requests.exceptions.Timeout:
        return "调用预测服务超时，请稍后重试"
    except Exception as e:
        return f"调用预测服务失败: {str(e)}"

# 测试代码（运行前请确保密钥已设置）
if __name__ == "__main__":
    test_input = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }
    # 测试时传入JSON字符串
    print(predict_wine_quality.invoke(json.dumps(test_input)))