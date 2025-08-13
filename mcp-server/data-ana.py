from mcp.server.fastmcp import FastMCP
import pandas as pd
import numpy as np
from io import StringIO
from typing import Dict, Any
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

mcp = FastMCP("Data Analysis MCP Server", host="0.0.0.0", port=8080)


@mcp.tool()
def simple_statistics(data: str) -> Dict[str, Any]:
    """
    Analyze data and return standard deviation analysis

    Args:
        data: Comma-separated numbers or CSV data to analyze

    Returns:
        Result of the standard deviation analysis
    """
    try:
        # Try to parse as comma-separated numbers first
        if ',' in data and '\n' not in data:
            # Handle comma-separated numbers like "10, 30, 60, 30, 100"
            numbers = [float(x.strip()) for x in data.split(',') if x.strip()]
            
            if not numbers:
                return {
                    "success": False,
                    "error": "No valid numbers found",
                    "message": "ไม่พบตัวเลขที่ถูกต้อง"
                }
            
            # Calculate statistics
            np_array = np.array(numbers)
            
            result = {
                "success": True,
                "data": numbers,
                "count": len(numbers),
                "mean": float(np.mean(np_array)),
                "median": float(np.median(np_array)),
                "std_deviation": float(np.std(np_array, ddof=1)),  # Sample standard deviation
                "population_std": float(np.std(np_array, ddof=0)),  # Population standard deviation
                "variance": float(np.var(np_array, ddof=1)),
                "min": float(np.min(np_array)),
                "max": float(np.max(np_array)),
                "range": float(np.max(np_array) - np.min(np_array)),
                "sum": float(np.sum(np_array)),
                "message": f"วิเคราะห์ข้อมูล {len(numbers)} ตัวเลขสำเร็จ"
            }
            
            return result
            
        else:
            # Try to parse as CSV data
            csv_buffer = StringIO(data.strip())
            df = pd.read_csv(csv_buffer)
            
            # Check if data is valid
            if df.empty:
                return {
                    "success": False,
                    "error": "Data is empty",
                    "message": "ข้อมูลว่างเปล่า"
                }
            
            # Get numeric columns for analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                return {
                    "success": False,
                    "error": "No numeric columns found",
                    "message": "ไม่พบคอลัมน์ตัวเลข"
                }
            
            # Perform statistical analysis for each numeric column
            analysis_results = {}
            
            for col in numeric_cols:
                col_data = df[col].dropna()  # Remove NaN values
                
                if len(col_data) > 0:
                    analysis_results[col] = {
                        "count": len(col_data),
                        "mean": float(col_data.mean()),
                        "median": float(col_data.median()),
                        "std_deviation": float(col_data.std()),
                        "variance": float(col_data.var()),
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "range": float(col_data.max() - col_data.min()),
                        "sum": float(col_data.sum())
                    }
            
            # Create summary
            result = {
                "success": True,
                "data_shape": list(df.shape),
                "columns": df.columns.tolist(),
                "numeric_columns": numeric_cols,
                "statistical_analysis": analysis_results,
                "message": "วิเคราะห์ข้อมูล CSV สำเร็จ"
            }
            
            return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "เกิดข้อผิดพลาดในการวิเคราะห์ข้อมูล"
        }


@mcp.tool()
def linear_regression(data: str) -> Dict[str, Any]:
    """
    Perform  linear regression analysis on CSV data using scikit-learn
    
    Args:
        data: dictionary data such as {"x1": [1, 2, 3], "x2": [4, 5, 6], "y": [7, 8, 9]}
        
    Returns:
         linear regression analysis results including coefficients, intercept, R-squared, predictions
    """

    try:
        # Handle markdown table format first
        if '|' in data:
            lines = [line.strip() for line in data.strip().split('\n') if line.strip()]
            
            # Find header line
            header_line = None
            for line in lines:
                if '|' in line and not line.replace('|', '').replace('-', '').replace(' ', ''):
                    continue  # Skip separator lines
                if '|' in line:
                    header_line = line
                    break
            
            if not header_line:
                return {
                    "success": False,
                    "error": "No valid header found",
                    "message": "ไม่พบหัวตารางที่ถูกต้อง"
                }
            
            # Parse headers and clean them
            headers = [col.strip() for col in header_line.split('|') if col.strip()]
            clean_headers = []
            for header in headers:
                clean_header = header.split('—')[0].strip()
                clean_header = clean_header.split('(')[0].strip()
                clean_headers.append(clean_header)
            
            # Parse data rows
            data_rows = []
            for line in lines:
                if '|' in line and not line.replace('|', '').replace('-', '').replace(' ', ''):
                    continue
                if '|' in line and line != header_line:
                    row_data = [col.strip() for col in line.split('|') if col.strip()]
                    if len(row_data) == len(clean_headers):
                        processed_row = []
                        for value in row_data:
                            clean_value = value.replace(',', '')
                            try:
                                processed_row.append(float(clean_value))
                            except ValueError:
                                processed_row.append(value)
                        data_rows.append(processed_row)
            
            # Convert to dictionary format like your example
            data_dict = {}
            for i, header in enumerate(clean_headers):
                data_dict[header] = [row[i] for row in data_rows if i < len(row)]
        
        else:
            # Handle CSV format
            csv_buffer = StringIO(data.strip())
            df_temp = pd.read_csv(csv_buffer)
            data_dict = {col: df_temp[col].tolist() for col in df_temp.columns}
        
        # Create DataFrame from dictionary (like your example)
        df = pd.DataFrame(data_dict)
        
        # Check if data is valid
        if df.empty:
            return {
                "success": False,
                "error": "Data is empty",
                "message": "ข้อมูลว่างเปล่า"
            }
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return {
                "success": False,
                "error": "Need at least 2 numeric columns for regression",
                "message": "ต้องการอย่างน้อย 2 คอลัมน์ตัวเลขสำหรับการถดถอย"
            }
        
        # Use last column as target (Y) and others as features (X)
        feature_cols = numeric_cols[:-1]
        target_col = numeric_cols[-1]
        
        # กำหนด X และ Y (like your example)
        X = df[feature_cols]
        y = df[target_col]
        
        # สร้างโมเดล (like your example)
        model = LinearRegression()
        model.fit(X, y)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        
        # Create equation string
        equation_parts = []
        for i, (coef, col) in enumerate(zip(model.coef_, feature_cols)):
            if i == 0:
                equation_parts.append(f"{coef:.4f}*{col}")
            else:
                sign = "+" if coef >= 0 else "-"
                equation_parts.append(f" {sign} {abs(coef):.4f}*{col}")
        
        if model.intercept_ >= 0:
            equation = f"{target_col} = " + "".join(equation_parts) + f" + {model.intercept_:.4f}"
        else:
            equation = f"{target_col} = " + "".join(equation_parts) + f" - {abs(model.intercept_):.4f}"
        
        # Extract prediction values from user context if available
        prediction_example = None
        if len(feature_cols) >= 1:
            # Try to extract numbers from the original data string for prediction
            import re
            
            # Look for prediction request patterns
            prediction_patterns = [
                r'ทำนาย.*?(\d+).*?(\d+)',  # "ทำนายราคาบ้านที่มีพื้นที่ใช้สอย 85 ตร.ม. และมี 3 ห้องนอน"
                r'พื้นที่.*?(\d+).*?ห้องนอน.*?(\d+)',
                r'(\d+).*?ตร\.ม\..*?(\d+).*?ห้องนอน'
            ]
            
            prediction_values = None
            for pattern in prediction_patterns:
                matches = re.findall(pattern, data)
                if matches:
                    try:
                        prediction_values = [float(matches[-1][0]), float(matches[-1][1])]
                        break
                    except (ValueError, IndexError):
                        continue
            
            # If no specific prediction request found, use mean values
            if prediction_values is None:
                prediction_values = [float(df[col].mean()) for col in feature_cols[:2]]
            
            # Make prediction
            if len(feature_cols) >= 2:
                predicted_price = model.predict([prediction_values[:2]])[0]
                prediction_example = {
                    "input_values": prediction_values[:2],
                    "input_labels": feature_cols[:2],
                    "predicted_value": float(predicted_price),
                    "prediction_context": "จากบริบทของผู้ใช้" if prediction_values != [float(df[col].mean()) for col in feature_cols[:2]] else "จากค่าเฉลี่ย"
                }
            elif len(feature_cols) == 1:
                predicted_price = model.predict([prediction_values[:1]])[0]
                prediction_example = {
                    "input_values": prediction_values[:1],
                    "input_labels": feature_cols[:1],
                    "predicted_value": float(predicted_price),
                    "prediction_context": "จากบริบทของผู้ใช้"
                }
        
        result = {
            "success": True,
            "data_dictionary": data_dict,  # Dictionary format like your example
            "regression_analysis": {
                "equation": equation,
                "intercept_a": float(model.intercept_),  # Intercept (a)
                "coefficients_b": {col: float(coef) for col, coef in zip(feature_cols, model.coef_)},  # Coefficients (b1, b2)
                "r_squared": float(r2),
                "rmse": float(rmse),
                "data_points": int(len(df))
            },
            "model_info": {
                "feature_variables": feature_cols,
                "target_variable": target_col,
                "model_performance": "ดีมาก" if r2 > 0.8 else "ดี" if r2 > 0.6 else "ปานกลาง"
            },
            "prediction_example": prediction_example,
            "message": f"วิเคราะห์การถดถอยเชิงเส้นสำเร็จ (R² = {r2:.4f})"
        }
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "เกิดข้อผิดพลาดในการวิเคราะห์การถดถอย"
        }

if __name__ == "__main__":
    mcp.run(transport="sse")
