import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# สร้างข้อมูล
data = {
    'พื้นที่ใช้สอย (ตร.ม.)': [50, 60, 70, 80, 90],
    'จำนวนห้องนอน': [1, 2, 2, 3, 3],
    'ราคาบ้าน (บาท)': [1100000, 1250000, 1400000, 1600000, 1750000]
}

# สร้าง DataFrame
df = pd.DataFrame(data)
print("ข้อมูลที่ใช้ในการฝึกโมเดล:")
print(df)
print("\n")

# เตรียมข้อมูล Features (X) และ Target (y)
X = df[['พื้นที่ใช้สอย (ตร.ม.)', 'จำนวนห้องนอน']]
y = df['ราคาบ้าน (บาท)']

# สร้างโมเดล Linear Regression
model = LinearRegression()

# ฝึกโมเดลด้วยข้อมูลทั้งหมด (เนื่องจากข้อมูลน้อย)
model.fit(X, y)

# ทำนายราคาด้วยข้อมูลเดิม
y_pred = model.predict(X)

# คำนวณค่าความแม่นยำ
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("ผลลัพธ์ของโมเดล:")
print(f"Mean Squared Error: {mse:,.2f}")
print(f"R-squared Score: {r2:.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mse):,.2f}")
print("\n")

# แสดงค่าสัมประสิทธิ์
print("สัมประสิทธิ์ของโมเดล:")
print(f"Intercept (ค่าคงที่): {model.intercept_:,.2f}")
print(f"Coefficient สำหรับพื้นที่: {model.coef_[0]:.2f} บาท/ตร.ม.")
print(f"Coefficient สำหรับจำนวนห้องนอน: {model.coef_[1]:,.2f} บาท/ห้อง")
print("\n")

# สมการการทำนาย
print("สมการการทำนายราคาบ้าน:")
print(f"ราคา = {model.intercept_:,.0f} + ({model.coef_[0]:.0f} × พื้นที่) + ({model.coef_[1]:,.0f} × จำนวนห้องนอน)")
print("\n")

# เปรียบเทียบราคาจริงกับราคาที่ทำนาย
comparison = pd.DataFrame({
    'พื้นที่': X['พื้นที่ใช้สอย (ตร.ม.)'],
    'ห้องนอน': X['จำนวนห้องนอน'],
    'ราคาจริง': y,
    'ราคาทำนาย': y_pred.round(0),
    'ความต่าง': (y - y_pred).round(0)
})
print("เปรียบเทียบราคาจริงกับราคาที่ทำนาย:")
print(comparison)
print("\n")

# ตัวอย่างการทำนายราคาบ้านใหม่
print("=== ตัวอย่างการทำนายราคาบ้านใหม่ ===")

# ตัวอย่าง: บ้านพื้นที่ 85 ตร.ม. 3 ห้องนอน
new_house = [[85, 3]]
predicted_price = model.predict(new_house)[0]
print(f"บ้านพื้นที่ 85 ตร.ม., 3 ห้องนอน")
print(f"ราคาที่ทำนายได้: {predicted_price:,.0f} บาท")
print()