from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Any, Optional
import pandas as pd
from io import StringIO
import matplotlib
matplotlib.use("Agg")  # ใช้ backend ที่ไม่ต้องการ GUI
import matplotlib.pyplot as plt
import base64
import io

mcp = FastMCP("Chart Generator Server", host="0.0.0.0", port=1224)


@mcp.tool()
def generate_bar_chart(data: str) -> Dict[str, Any]:
    """
    Generate bar chart from CSV data

     Args:
        data: CSV string with first column as labels (x-axis) and second column as values (y-axis)
        
    Returns:
        Dictionary containing bar chart image and metadata
    """
    try:
        # Parse CSV string
        csv_buffer = StringIO(data.strip())
        df = pd.read_csv(csv_buffer)
        
        # Assume first column is labels (x-axis) and second column is values (y-axis)
        if len(df.columns) < 2:
            return {
                "success": False,
                "error": "CSV must have at least 2 columns",
                "message": "ข้อมูล CSV ต้องมีอย่างน้อย 2 คอลัมน์"
            }
        
        # Extract data
        x_labels = df.iloc[:, 0].astype(str).tolist()
        y_values = pd.to_numeric(df.iloc[:, 1], errors='coerce').tolist()
        
        # Create bar chart data format
        chart_data = {
            "success": True,
            "data": {
                "x": x_labels,
                "y": y_values,
                "label": "Data Series"
            },
            "config": {
                "title": "Chart from CSV Data",
                "xlabel": df.columns[0] if len(df.columns) > 0 else "Categories",
                "ylabel": df.columns[1] if len(df.columns) > 1 else "Values",
                "width": 10,
                "height": 6,
                "grid": True,
                "style": "default"
            },
            "message": "แปลงข้อมูล CSV สำเร็จ"
        }
        
        # generate chart using matplotlib
        
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(chart_data["config"]["width"], chart_data["config"]["height"]))
            
            # Create bar chart
            bars = ax.bar(chart_data["data"]["x"], chart_data["data"]["y"], 
                         label=chart_data["data"]["label"], color='skyblue', edgecolor='navy')
            
            # Set chart properties
            ax.set_title(chart_data["config"]["title"], fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel(chart_data["config"]["xlabel"], fontsize=12)
            ax.set_ylabel(chart_data["config"]["ylabel"], fontsize=12)
            
            # Add grid
            if chart_data["config"]["grid"]:
                ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
            
            # Rotate x-axis labels if needed
            if len(chart_data["data"]["x"]) > 5:
                plt.xticks(rotation=45, ha='right')
            
            # Add legend
            ax.legend()
            
            # Convert to base64
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            buffer.close()
            plt.close(fig)
            
            # Return complete response with chart
            return {
                "success": True,
                "chart_type": "bar",
                "image": f"data:image/png;base64,{image_base64}",
                "data": chart_data["data"],
                "config": chart_data["config"],
                "message": "สร้างแผนภูมิแท่งสำเร็จ"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "เกิดข้อผิดพลาดในการสร้างแผนภูมิ"
            }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "เกิดข้อผิดพลาดในการแปลงข้อมูล CSV"
        }

@mcp.tool()
def generate_line_chart(data: str) -> Dict[str, Any]:
    """
    Generate line chart from CSV data
    
    Args:
        data: CSV string with first column as x-values and second column as y-values
        
    Returns:
        Dictionary containing line chart image and metadata
    """
    try:
        # Parse CSV string
        csv_buffer = StringIO(data.strip())
        df = pd.read_csv(csv_buffer)
        
        # Assume first column is x-axis and second column is y-axis
        if len(df.columns) < 2:
            return {
                "success": False,
                "error": "CSV must have at least 2 columns",
                "message": "ข้อมูล CSV ต้องมีอย่างน้อย 2 คอลัมน์"
            }
        
        # Extract data
        x_data = df.iloc[:, 0].tolist()
        y_data = pd.to_numeric(df.iloc[:, 1], errors='coerce').tolist()
        
        # Create chart data
        chart_data = {
            "success": True,
            "data": {
                "x": x_data,
                "y": y_data,
                "label": "Data Series"
            },
            "config": {
                "title": "Line Chart from CSV Data",
                "xlabel": df.columns[0] if len(df.columns) > 0 else "X-axis",
                "ylabel": df.columns[1] if len(df.columns) > 1 else "Y-axis",
                "width": 10,
                "height": 6,
                "grid": True,
                "style": "default"
            },
            "message": "แปลงข้อมูล CSV สำเร็จ"
        }
        
        # generate chart using matplotlib
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(chart_data["config"]["width"], chart_data["config"]["height"]))
            
            # Create line chart
            line = ax.plot(chart_data["data"]["x"], chart_data["data"]["y"], 
                          label=chart_data["data"]["label"], color='blue', 
                          marker='o', markersize=6, linewidth=2)
            
            # Set chart properties
            ax.set_title(chart_data["config"]["title"], fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel(chart_data["config"]["xlabel"], fontsize=12)
            ax.set_ylabel(chart_data["config"]["ylabel"], fontsize=12)
            
            # Add grid
            if chart_data["config"]["grid"]:
                ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add value labels on points
            for i, (x, y) in enumerate(zip(chart_data["data"]["x"], chart_data["data"]["y"])):
                ax.annotate(f'{int(y)}', (x, y), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=9)
            
            # Rotate x-axis labels if needed
            if len(chart_data["data"]["x"]) > 5:
                plt.xticks(rotation=45, ha='right')
            
            # Add legend
            ax.legend()
            
            # Convert to base64
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            buffer.close()
            plt.close(fig)
            
            # Return complete response with chart
            return {
                "success": True,
                "chart_type": "line",
                "image": f"data:image/png;base64,{image_base64}",
                "data": chart_data["data"],
                "config": chart_data["config"],
                "message": "สร้างแผนภูมิเส้นสำเร็จ"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "เกิดข้อผิดพลาดในการสร้างแผนภูมิ"
            }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "เกิดข้อผิดพลาดในการแปลงข้อมูล CSV"
        }

@mcp.tool()
def generate_pie_chart(data: str) -> Dict[str, Any]:
    """
    Generate pie chart from CSV data
    
    Args:
        data: CSV string with first column as labels and second column as values
        
    Returns:
        Dictionary containing pie chart image and metadata
    """
    try:
        # Parse CSV string
        csv_buffer = StringIO(data.strip())
        df = pd.read_csv(csv_buffer)
        
        # Assume first column is labels and second column is values
        if len(df.columns) < 2:
            return {
                "success": False,
                "error": "CSV must have at least 2 columns",
                "message": "ข้อมูล CSV ต้องมีอย่างน้อย 2 คอลัมน์"
            }
        
        # Extract data
        labels = df.iloc[:, 0].astype(str).tolist()
        values = pd.to_numeric(df.iloc[:, 1], errors='coerce').tolist()
        
        # Create chart data
        chart_data = {
            "success": True,
            "data": {
                "labels": labels,
                "values": values
            },
            "config": {
                "title": "Pie Chart from CSV Data",
                "width": 10,
                "height": 8,
                "style": "default"
            },
            "message": "แปลงข้อมูล CSV สำเร็จ"
        }
        
        # generate chart using matplotlib
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(chart_data["config"]["width"], chart_data["config"]["height"]))
            
            # Create pie chart with colors
            colors = plt.cm.Set3(range(len(labels)))
            wedges, texts, autotexts = ax.pie(
                values, 
                labels=labels, 
                autopct='%1.1f%%',
                colors=colors,
                startangle=90,
                explode=[0.05] * len(labels)  # slight separation for all slices
            )
            
            # Set chart properties
            ax.set_title(chart_data["config"]["title"], fontsize=16, fontweight="bold", pad=20)
            
            # Style the text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
            
            for text in texts:
                text.set_fontsize(11)
            
            # Convert to base64
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            buffer.close()
            plt.close(fig)
            
            # Return complete response with chart
            return {
                "success": True,
                "chart_type": "pie",
                "image": f"data:image/png;base64,{image_base64}",
                "data": chart_data["data"],
                "config": chart_data["config"],
                "message": "สร้างแผนภูมิวงกลมสำเร็จ"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "เกิดข้อผิดพลาดในการสร้างแผนภูมิ"
            }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "เกิดข้อผิดพลาดในการแปลงข้อมูล CSV"
        }


if __name__ == "__main__":
    mcp.run(transport="sse")
