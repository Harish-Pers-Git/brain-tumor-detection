
🧠 Brain Tumor Detection App
This project is a deep learning-based web application that detects and classifies brain tumors from MRI images using Convolutional Neural Networks (CNN). The app supports four tumor types: Glioma, Meningioma, Pituitary, and No Tumor.

🚀 Features
🔍 Upload MRI images and detect tumor type.

📊 Display prediction confidence levels as animated bar graphs.

💡 Suggest diagnosis and treatment based on tumor class.

📄 Generate and download diagnosis reports as PDFs.

🔐 Secure login/logout functionality.

🌐 Flask-powered web interface with Chart.js animations.

🖼️ Demo
![image](https://github.com/user-attachments/assets/de07b699-baa0-4a08-b256-4b703e96f88e)
![image](https://github.com/user-attachments/assets/c71a31fc-ba50-4c22-979e-9b2fda078ff2)
![image](https://github.com/user-attachments/assets/5344d460-9620-4a77-a3bf-5d3939cb354d)
![image](https://github.com/user-attachments/assets/dfbcb7c2-2563-4f64-9936-33c65d7b8b4c)

📁 Project Structure
brain_tumor_app/
│
├── app.py                    # Flask backend
├── model/brain_tumor_model.h5  # Pre-trained CNN model
├── static/
│   ├── uploads/             # Uploaded MRI images
│   ├── graphs/              # Prediction confidence graphs
│   └── style.css            # Custom styles
├── templates/
│   ├── login.html
│   ├── input.html
│   ├── process.html
│   ├── detection.html
│   └── report.html
├── utils/
│   └── graph_generator.py   # Graph generation using matplotlib
├── requirements.txt         # Dependencies
└── README.md
🧪 Model Details
Architecture: Custom CNN

Dataset: Labeled MRI images (4 classes)

Format: .h5 Keras model

Accuracy: ~98% on validation set

💻 Installation & Usage
1. Clone the repository
git clone https://github.com/Harish-Pers-Git/brain-tumor-detection.git
cd brain-tumor-detection
2. Create a virtual environment and activate it
python -m venv venv
venv\Scripts\activate  # On Windows
# OR
source venv/bin/activate  # On Linux/Mac
3. Install dependencies
pip install -r requirements.txt
4. Run the app
python app.py
Then open your browser and go to:
http://127.0.0.1:5000

📋 Technologies Used
Technology	Role
Python	Core language
Flask	Web backend
Keras + TensorFlow	CNN Model
Chart.js	Confidence graph animations
HTML/CSS	Frontend UI
Matplotlib	Bar graph generation
ReportLab	PDF generation

🛡️ Suggested Treatments
Based on the tumor type detected, the app provides general recommendations for:

Imaging & follow-up

Biopsy

Chemotherapy

Radiation Therapy

Surgical options

⚠️ This tool is for educational/research purposes only and should not replace medical advice.

🙋‍♂️ Author
Harish – GitHub
