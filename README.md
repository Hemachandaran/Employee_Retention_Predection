# Employee Retention Prediction

## Overview
This repository contains a Streamlit application designed for classifying employee retention based on various features. The application utilizes a LightGBM model for manual classification and a simple neural network for automatic text classification.

## Features
- **Automatic Classification**: Users can input text data, and the model will predict the classification automatically.
- **Manual Classification**: Users can enter specific features regarding an employee, and the model will provide predictions based on these inputs.

## Requirements
To run this application, you need to install the following libraries:
- `streamlit`
- `torch`
- `pandas`
- `pickle`
- `lightgbm`

You can install these libraries using pip:

```bash
pip install streamlit torch pandas lightgbm
```

## Files
- `trained_pipeline.pkl`: A pickled LightGBM model for manual classification.
- `text_model.pkl`: A pickled neural network model for automatic text classification.
- `vectorizer_text.pkl`: A pickled TF-IDF vectorizer used for text feature extraction.

## Usage
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Access the application in your web browser at `http://localhost:8501`.

## Application Layout
The application consists of two main sections:
- **Automatic Classification**: Enter text in the provided text area and click "Classify" to receive a prediction.
- **Manual Classification**: Fill in all required fields regarding employee details, then click "Classify" to obtain predictions based on the provided features.

## Contributions
Contributions are welcome! Please fork the repository and submit a pull request with your improvements or bug fixes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

