from fastmcp import FastMCP
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from typing import List, Dict, Any
import json
import io

# สร้าง MCP server
mcp = FastMCP("House Price Prediction", host="0.0.0.0", port=8080)


# กำหนด data models
class HouseData(BaseModel):
    area: float = Field(description="พื้นที่ใช้สอยในหน่วยตารางเมตร")
    bedrooms: int = Field(description="จำนวนห้องนอน")
    price: float = Field(description="ราคาบ้านในหน่วยบาท")


class PredictionInput(BaseModel):
    area: float = Field(description="พื้นที่ใช้สอยในหน่วยตารางเมตร")
    bedrooms: int = Field(description="จำนวนห้องนอน")


class TrainedModel(BaseModel):
    intercept: float
    coefficients: List[float]
    feature_names: List[str]
    mse: float
    r2_score: float


# เก็บโมเดลที่ได้รับการฝึก
trained_models = {}


@mcp.tool()
def train_model(
    csv_data: str = Field(description="ข้อมูล CSV ที่มี columns: area, bedrooms, price"),
    model_name: str = Field(default="default", description="ชื่อโมเดล"),
) -> Dict[str, Any]:
    """
    ฝึกโมเดล Linear Regression จากข้อมูล CSV
    """
    try:
        # อ่านข้อมูล CSV
        df = pd.read_csv(io.StringIO(csv_data))

        # ตรวจสอบ columns ที่จำเป็น
        required_columns = ["area", "bedrooms", "price"]
        if not all(col in df.columns for col in required_columns):
            return {
                "success": False,
                "error": f"ต้องมี columns: {required_columns}. พบ: {list(df.columns)}",
            }

        # เตรียมข้อมูล
        X = df[["area", "bedrooms"]]
        y = df["price"]

        # สร้างและฝึกโมเดล
        model = LinearRegression()
        model.fit(X, y)

        # ทำนายและประเมิน
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # เก็บโมเดล
        trained_models[model_name] = {
            "model": model,
            "feature_names": ["area", "bedrooms"],
            "training_data": df,
        }

        # สร้างผลลัพธ์
        result = {
            "success": True,
            "model_name": model_name,
            "data_points": len(df),
            "intercept": float(model.intercept_),
            "coefficients": {
                "area": float(model.coef_[0]),
                "bedrooms": float(model.coef_[1]),
            },
            "metrics": {
                "mse": float(mse),
                "rmse": float(np.sqrt(mse)),
                "r2_score": float(r2),
            },
            "equation": f"ราคา = {model.intercept_:.0f} + ({model.coef_[0]:.0f} × พื้นที่) + ({model.coef_[1]:.0f} × จำนวนห้องนอน)",
            "training_summary": df.describe().to_dict(),
        }

        return result

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def predict_price(
    area: float = Field(description="พื้นที่ใช้สอยในหน่วยตารางเมตร"),
    bedrooms: int = Field(description="จำนวนห้องนอน"),
    model_name: str = Field(default="default", description="ชื่อโมเดลที่ต้องการใช้"),
) -> Dict[str, Any]:
    """
    ทำนายราคาบ้านจากพื้นที่และจำนวนห้องนอน
    """
    try:
        if model_name not in trained_models:
            return {
                "success": False,
                "error": f"ไม่พบโมเดล '{model_name}'. กรุณาฝึกโมเดลก่อน",
            }

        model_data = trained_models[model_name]
        model = model_data["model"]

        # ทำนาย
        prediction = model.predict([[area, bedrooms]])[0]

        # คำนวณช่วงความเชื่อมั่น (ประมาณการง่ายๆ)
        training_prices = model_data["training_data"]["price"]
        std_price = training_prices.std()
        confidence_interval = (prediction - std_price, prediction + std_price)

        return {
            "success": True,
            "model_name": model_name,
            "input": {"area": area, "bedrooms": bedrooms},
            "predicted_price": float(prediction),
            "predicted_price_formatted": f"{prediction:,.0f} บาท",
            "confidence_interval": {
                "lower": float(confidence_interval[0]),
                "upper": float(confidence_interval[1]),
                "formatted": f"{confidence_interval[0]:,.0f} - {confidence_interval[1]:,.0f} บาท",
            },
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def batch_predict(
    predictions: str = Field(
        description="ข้อมูล CSV ที่มี columns: area, bedrooms สำหรับทำนายหลายรายการ"
    ),
    model_name: str = Field(default="default", description="ชื่อโมเดลที่ต้องการใช้"),
) -> Dict[str, Any]:
    """
    ทำนายราคาบ้านหลายรายการพร้อมกัน
    """
    try:
        if model_name not in trained_models:
            return {
                "success": False,
                "error": f"ไม่พบโมเดล '{model_name}'. กรุณาฝึกโมเดลก่อน",
            }

        # อ่านข้อมูล
        df = pd.read_csv(io.StringIO(predictions))

        # ตรวจสอบ columns
        required_columns = ["area", "bedrooms"]
        if not all(col in df.columns for col in required_columns):
            return {
                "success": False,
                "error": f"ต้องมี columns: {required_columns}. พบ: {list(df.columns)}",
            }

        model_data = trained_models[model_name]
        model = model_data["model"]

        # ทำนายทั้งหมด
        X = df[["area", "bedrooms"]]
        predictions = model.predict(X)

        # สร้างผลลัพธ์
        results = []
        for i, (_, row) in enumerate(df.iterrows()):
            results.append(
                {
                    "area": float(row["area"]),
                    "bedrooms": int(row["bedrooms"]),
                    "predicted_price": float(predictions[i]),
                    "predicted_price_formatted": f"{predictions[i]:,.0f} บาท",
                }
            )

        return {
            "success": True,
            "model_name": model_name,
            "predictions": results,
            "total_predictions": len(results),
            "summary": {
                "min_price": float(predictions.min()),
                "max_price": float(predictions.max()),
                "avg_price": float(predictions.mean()),
            },
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def get_model_info(
    model_name: str = Field(default="default", description="ชื่อโมเดลที่ต้องการดูข้อมูล")
) -> Dict[str, Any]:
    """
    ดูข้อมูลรายละเอียดของโมเดล
    """
    try:
        if model_name not in trained_models:
            return {"success": False, "error": f"ไม่พบโมเดล '{model_name}'"}

        model_data = trained_models[model_name]
        model = model_data["model"]
        training_data = model_data["training_data"]

        # คำนวณเมตริกซ์
        X = training_data[["area", "bedrooms"]]
        y = training_data["price"]
        y_pred = model.predict(X)

        return {
            "success": True,
            "model_name": model_name,
            "intercept": float(model.intercept_),
            "coefficients": {
                "area": float(model.coef_[0]),
                "bedrooms": float(model.coef_[1]),
            },
            "metrics": {
                "mse": float(mean_squared_error(y, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
                "r2_score": float(r2_score(y, y_pred)),
            },
            "training_data_info": {
                "data_points": len(training_data),
                "area_range": f"{training_data['area'].min():.0f} - {training_data['area'].max():.0f} ตร.ม.",
                "bedrooms_range": f"{training_data['bedrooms'].min()} - {training_data['bedrooms'].max()} ห้อง",
                "price_range": f"{training_data['price'].min():,.0f} - {training_data['price'].max():,.0f} บาท",
            },
            "equation": f"ราคา = {model.intercept_:.0f} + ({model.coef_[0]:.0f} × พื้นที่) + ({model.coef_[1]:.0f} × จำนวนห้องนอน)",
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def list_models() -> Dict[str, Any]:
    """
    แสดงรายการโมเดลทั้งหมดที่มีอยู่
    """
    models = []
    for name, data in trained_models.items():
        training_data = data["training_data"]
        models.append(
            {
                "name": name,
                "data_points": len(training_data),
                "created": "recently",  # ใน production ควรเก็บ timestamp
            }
        )

    return {"success": True, "models": models, "total_models": len(models)}


if __name__ == "__main__":
    # ตัวอย่างการทดสอบ
    sample_data = """area,bedrooms,price
50,1,1100000
60,2,1250000
70,2,1400000
80,3,1600000
90,3,1750000"""

    print("เริ่มต้น MCP Server สำหรับทำนายราคาบ้าน")
    print("Tools ที่ใช้ได้:")
    print("1. train_model - ฝึกโมเดลจากข้อมูล CSV")
    print("2. predict_price - ทำนายราคาบ้าน")
    print("3. batch_predict - ทำนายหลายรายการ")
    print("4. get_model_info - ดูข้อมูลโมเดล")
    print("5. list_models - แสดงรายการโมเดล")
    print("\nตัวอย่างข้อมูล CSV:")
    print(sample_data)

    # รันเซิร์ฟเวอร์
    mcp.run(transport="sse")
