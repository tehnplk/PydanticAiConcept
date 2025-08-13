import pandas as pd
from sklearn.linear_model import LinearRegression

# ข้อมูลตัวอย่าง
data = {
    "พื้นที่ใช้สอย": [50, 60, 70, 80, 90],  # X1
    "จำนวนห้องนอน": [1, 2, 2, 3, 3],  # X2
    "ราคาบ้าน": [1100000, 1250000, 1400000, 1600000, 1750000],  # Y
}

df = pd.DataFrame(data)

# กำหนด X และ Y
X = df[["พื้นที่ใช้สอย","จำนวนห้องนอน"]]
y = df["ราคาบ้าน"]

# สร้างโมเดล
model = LinearRegression()
model.fit(X, y)

# ทำนายราคาบ้านเมื่อพื้นที่ 85 ตร.ม. และมี 3 ห้องนอน
predicted_price = model.predict([[110,4]])[0]
print(f"ทำนายราคา: {predicted_price:.0f}")
