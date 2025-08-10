import sys
import asyncio
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton, QLabel
from PyQt6.QtCore import QThread, pyqtSignal, QObject
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.mcp import MCPServerSSE
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

import logfire

logfire.configure()
logfire.instrument_pydantic_ai()

mcp_server = MCPServerSSE(url="http://localhost:8081/sse")




class AgentWorker(QObject):
    signal_finished = pyqtSignal(str, str)
    signal_error = pyqtSignal(str)
    signal_message_history = pyqtSignal(list)
    
    class Result(BaseModel):
        query: str = Field(description="SQL query")
        result: str = Field(description="result of the SQL query display in tabular format")


    def __init__(self, prompt,message_history):
        super().__init__()
        self.prompt = prompt
        self.message_history = message_history
        self.new_message_history = []

        self.sys_prompt = open("sys_prompt.txt", "r", encoding="utf-8").read()
        self.agent = Agent(
            model="google-gla:gemini-2.5-flash",
            toolsets=[mcp_server],
            system_prompt=self.sys_prompt,
            output_type=self.Result,
        )


    def run(self):
        asyncio.run(self.chat())

    async def chat(self):
        try:
            async with self.agent:
                result = await self.agent.run(self.prompt, message_history=self.message_history)
            self.new_message_history = result.all_messages()
            self.signal_message_history.emit(self.new_message_history)  
            self.signal_finished.emit(result.output.query, result.output.result)
        except Exception as e:
            self.signal_error.emit(str(e))

class ChatWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MCP SSE Chat")
        self.resize(600, 400)
        layout = QVBoxLayout()
        self.chatbox = QLineEdit()
        self.send_btn = QPushButton("Send")
        self.result_label = QLabel("Result:")
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(QLabel("Enter your prompt:"))
        layout.addWidget(self.chatbox)
        layout.addWidget(self.send_btn)
        layout.addWidget(self.result_label)
        layout.addWidget(self.result_text)
        self.setLayout(layout)
        self.send_btn.clicked.connect(self.handle_send)
        self.massage_history = []

    def handle_send(self):
        prompt = self.chatbox.text()
        if not prompt:
            return
        self.send_btn.setEnabled(False)
        self.result_text.setText("Waiting for response...")
        self.thread = QThread()
        self.worker = AgentWorker(prompt, self.massage_history)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.signal_message_history.connect(self.manage_history)
        self.worker.signal_finished.connect(self.display_result)
        self.worker.signal_error.connect(self.display_error)
        self.worker.signal_finished.connect(self.thread.quit)
        self.worker.signal_finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
    
    def manage_history(self, new_message_history):
        self.massage_history = new_message_history
        # Here you can handle the message history as needed, e.g., display it in the UI or log it.

    def display_result(self, query, result):
        self.result_text.setText(f"Query:\n{query}\n\nResult:\n{result}")
        self.send_btn.setEnabled(True)

    def display_error(self, error):
        self.result_text.setText(f"Error: {error}")
        self.send_btn.setEnabled(True)

def main():
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
