# LeafLine: Plant Disease Classification Web App

LeafLine is an AI-powered web application for plant disease detection. Upload a plant leaf image and instantly receive a prediction of the plant type and its health status using a deep learning model. Built with a modern HTML/JS frontend and a Python Flask backend.

## Features
- Upload a plant leaf image and get a prediction (plant type and health status)
- Beautiful, responsive frontend
- Python Flask backend with TensorFlow/Keras model
- Logging of predictions for analysis
- Easy to run locally

## Project Structure
```
├── backend/
│   └── api_backend.py         # Flask backend API
│   └── requirements.txt      # Backend dependencies
├── frontend/
│   └── index.html            # Main UI
│   └── about.html            # About page
├── models/
│   └── plant_disease_model.keras  # Trained Keras model
│   └── best_plant_disease_model.h5
├── assets/
│   └── logo.png              # App logo
├── prediction_log.txt        # Log of predictions
├── Sample Images/            # Example images for testing
├── LICENSE                   # MIT License
├── README.md                 # Project documentation
```

## Setup & Usage

1. **Clone the repository**
2. **Install dependencies**
   ```powershell
   cd backend
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   pip install -r requirements.txt
   ```
   *(A `requirements.txt` is provided in the `backend/` folder with Flask, flask-cors, tensorflow, pillow, numpy)*

3. **Run the backend**
   ```powershell
   python api_backend.py
   ```
   This will start the Flask server and open the frontend in your browser.

4. **Use the app**
   - Go to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) *or your localhost domain if changed*
   - Upload a plant leaf image and view the prediction.

## Model Training (Google Colab)

The model used in this project was developed and trained in Google Colab. You can view the original Colab notebook here (accessible only to Batangas State University email accounts):

[LeafLine Model Training Notebook (Google Colab)](https://colab.research.google.com/drive/1oAFY0ma_gCPOwuRBal_9PVO8iIjE2uFm#scrollTo=MMg00V7LMeuO)

## Dataset Acknowledgement

This project uses the PlantVillage dataset:
- Original paper: [PlantVillage: A Public Dataset for Plant Disease Detection](https://arxiv.org/abs/1511.08060)
- Dataset URL: [Mendeley Data](https://data.mendeley.com/datasets/tywbtsjrjv/1)

The PlantVillage dataset is © the original authors and is distributed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license. We do not claim ownership of the dataset.

## License

The code in this repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.
