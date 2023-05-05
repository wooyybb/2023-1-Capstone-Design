import sys
import requests
import openai
import os
import pandas as pd
import numpy as np
import utils
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QTextEdit, QComboBox, QHBoxLayout, QVBoxLayout, QFileDialog

class ChatGPT_GUI(QWidget):
    def __init__(self):
        super().__init__()

        self.prob_type = None #'Classification' or 'Regression'
        self.query_list = None
        self.csv_path = None
        self.target_cols = None
        self.useless_cols = None
        self.util = utils.utils()

        self.initUI()

    def initUI(self):
        # Create labels for input fields
        csv_path_label = QLabel('Path of CSV file:')
        model_type_label = QLabel('Model type (Classification/Regression):')
        target_cols_label = QLabel('Target columns (separated by //):')
        useless_cols_label = QLabel('Useless-feature columns (separated by //):')

        # Create input fields
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(['Classification', 'Regression'])
        self.target_cols_edit = QTextEdit()
        self.useless_cols_edit = QTextEdit()

        # Adjust the size of target_cols_edit and useless_cols_edit to be the same
        self.target_cols_edit.setMinimumHeight(50)
        self.useless_cols_edit.setMinimumHeight(50)

        # Create select CSV file button
        self.csv_path_btn = QPushButton('Select CSV File', self)
        self.csv_path_btn.clicked.connect(self.select_csv_file)

        # Create send button
        send_btn = QPushButton('Send', self)
        send_btn.clicked.connect(self.send_request)

        # Create output field
        self.output_field = QTextEdit()

        # Create layout for input fields
        input_layout = QVBoxLayout()
        input_layout.addWidget(csv_path_label)
        input_layout.addWidget(self.csv_path_btn)
        input_layout.addWidget(model_type_label)
        input_layout.addWidget(self.model_type_combo)
        input_layout.addWidget(target_cols_label)
        input_layout.addWidget(self.target_cols_edit)
        input_layout.addWidget(useless_cols_label)
        input_layout.addWidget(self.useless_cols_edit)

        # Create layout for send button and output field
        output_layout = QVBoxLayout()
        output_layout.addWidget(send_btn)
        output_layout.addWidget(self.output_field)

        # Create main layout
        main_layout = QHBoxLayout()
        main_layout.addLayout(input_layout)
        main_layout.addLayout(output_layout)

        # Set main layout
        self.setLayout(main_layout)

        # Set window properties - 1, 2는 생성위치 3, 4는 창 크기
        self.setWindowTitle('A.C.G.C')
        self.setGeometry(1000, 500, 1000, 900)
    
    def select_csv_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)", options=options)
        if filename:
            with open('input_values.txt', 'w') as f:
                f.write(f"CSV Path: {filename}\n")
            self.csv_path_btn.setText(filename)

    def save_input_values(self, csv_path, model_type, target_cols, useless_cols):
        with open('input_values.txt', 'w') as f:
            f.write(f"CSV Path: {csv_path}\n")
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Target Columns: {', '.join(target_cols)}\n")
            f.write(f"Useless-Feature Columns: {', '.join(useless_cols)}\n")
            
            
    def send_request(self):
        # Get input values
        csv_path = self.csv_path_btn.text() 
        model_type = self.model_type_combo.currentText()
        target_cols = self.target_cols_edit.toPlainText().split('//')
        useless_cols = self.useless_cols_edit.toPlainText().split('//')

        # Save input values to file
        self.save_input_values(csv_path, model_type, target_cols, useless_cols)

        openai.api_key = "<api key>"

        if model_type == "Classification":
            query_list = self.util.classification_process(csv_path, useless_cols, target_cols)

        elif model_type == "Regression":
            query_list = self.util.regression_process(csv_path, useless_cols, target_cols)

        # Set user_content with the content from query_list
        user_content = query_list[0]

        messages = []
        messages.append({"role": "user", "content": f"{user_content}"})
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

        assistant_content = completion.choices[0].message["content"].strip()
        messages.append({"role": "assistant", "content": f"{assistant_content}"})

        print(f"GPT: {assistant_content}")
    

        # Update output field with response text
        self.output_field.setText(assistant_content)


    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    chatgpt_gui = ChatGPT_GUI()
    chatgpt_gui.show()
    sys.exit(app.exec_())
