from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
import os
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Spacer


# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)
app.secret_key = "your_secret_key"
users = {
    "patient1": "password123",
    "patient2": "securepassword"
   
}

# Define folders
UPLOAD_FOLDER = os.path.join('static', 'uploads')
GRAPH_FOLDER = os.path.join('static', 'graphs')
PDF_FOLDER = os.path.join('static', 'pdfs')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRAPH_FOLDER, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GRAPH_FOLDER'] = GRAPH_FOLDER
app.config['PDF_FOLDER'] = PDF_FOLDER

# Load the trained model
MODEL_PATH = os.path.join('models', 'best_brain_tumor_model_v2.keras')
model = load_model(MODEL_PATH)

# Define the classes
CLASSES = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

def preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  
    return image

def classify_tumor(image_path):
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)[0]
    predicted_class_index = np.argmax(predictions)
    predicted_class = CLASSES[predicted_class_index]
    confidence_level = round(float(predictions[predicted_class_index] * 100), 2)  # Convert to Python float
    confidence_scores = [float(score * 100) for score in predictions]  # Convert all to Python float
    return predicted_class, confidence_level, confidence_scores

def generate_confidence_graph(confidence_scores):
    plt.figure(figsize=(6, 4))
    plt.bar(CLASSES, confidence_scores, color=['red', 'blue', 'green', 'purple'], edgecolor='black')
    plt.ylim(0, 100)
    plt.ylabel('Confidence (%)', fontsize=12)
    plt.title('Tumor Classification Confidence', fontsize=14, fontweight='bold')

    for i, value in enumerate(confidence_scores):
        plt.text(i, value + 2, f'{value:.2f}%', ha='center', fontsize=10)

    graph_path = os.path.join(GRAPH_FOLDER, 'confidence_graph.png')
    plt.savefig(graph_path, bbox_inches='tight', transparent=True)
    plt.close()
    return graph_path

def get_treatment_and_explanation(tumor_type, confidence_level):
    treatments = {
        "Glioma": ("A glioma brain tumor is a type of cancerous brain  tumor that originates from glial cells which are the supporting cells of the brain and spinal cord meaning it grows from the cells that surround and protect nerve cells rather than the nerve cells themselves; gliomas can vary in aggressiveness and can be either benign (non-cancerous) or malignant (cancerous) with the most common type being glioblastoma, a highly aggressive form of glioma. ", "Treatment includes surgery, radiation, and chemotherapy."),
        "Pituitary": ("Pituitary tumors are unusual growths that develop in the pituitary gland. This gland is an organ about the size of a pea. It's located behind the nose at the base of the brain. Some of these tumors cause the pituitary gland to make too much of certain hormones that control important body functions.", "Common treatments are surgery, radiation, and hormone therapy."),
        "Meningioma": ("Meningiomas form in the meninges.", "Treatment involves surgery and possibly radiation therapy."),
        "No Tumor": ("No tumor detected.", "No treatment required, but regular checkups are advised.")
    }
    explanation, treatment = treatments[tumor_type]
    diagnosis_time = "Immediate diagnosis recommended." if confidence_level >= 80 else "Diagnosis within a few days."
    return explanation, treatment, diagnosis_time


from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from textwrap import wrap
import os

def generate_pdf(patient_name, patient_age, tumor_type, confidence_level, explanation, treatment, diagnosis_time, uploaded_image_filename):
    pdf_filename = f"{patient_name}_tumor_report.pdf"
    pdf_path = os.path.join(PDF_FOLDER, pdf_filename)

    c = canvas.Canvas(pdf_path, pagesize=letter)

    # Header Title - White Background and Black Text
    c.setFont("Helvetica-Bold", 16)
    c.setFillColorRGB(0, 0, 0)  # Black text
    c.drawCentredString(300, 770, "BRAIN TUMOR DETECTION REPORT")

    # Patient Details
    c.setFont("Helvetica", 12)
    c.drawString(50, 720, f"Patient Name: {patient_name}")
    c.drawString(50, 700, f"Age: {patient_age}")
    c.drawString(50, 680, f"Tumor Type: {tumor_type}")

    # Confidence Level - Bold
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 660, f"Confidence Level: {confidence_level}%")

    # Diagnosis Time
    c.setFont("Helvetica", 12)
    c.drawString(50, 640, f"Diagnosis Time: {diagnosis_time}")

    # Observations
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 610, "Observations:")
    c.setFont("Helvetica", 12)

    # Wrap explanation text
    wrapped_explanation = wrap(explanation, width=80)
    y_position = 590
    for line in wrapped_explanation:
        c.drawString(50, y_position, line)
        y_position -= 20

    # Suggested Treatment
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position - 10, "Suggested Treatment:")
    c.setFont("Helvetica", 12)
    c.drawString(50, y_position - 30, treatment)

    # Add space before the image
    y_position -= 80

    # Add uploaded image
    uploaded_image_path = os.path.join(UPLOAD_FOLDER, uploaded_image_filename)
    if os.path.exists(uploaded_image_path):
        c.drawImage(ImageReader(uploaded_image_path), 150, y_position - 200, width=200, height=200)

    # Footer
    c.setFont("Helvetica", 10)
    c.drawString(50, 100, "Report generated by Brain Tumor Detection System")

    c.save()
    return pdf_path


@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['user'] = username
            return redirect(url_for('input_module'))
        else:
            flash("Invalid username or password.", "danger")
    return render_template('login.html')

@app.route('/input', methods=['GET', 'POST'])
def input_module():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        patient_name = request.form.get('patient_name', 'Unknown')
        patient_age = request.form.get('patient_age', 'Unknown')
        uploaded_image = request.files['image']

        if uploaded_image:
            filename = secure_filename(uploaded_image.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            uploaded_image.save(file_path)

            session['patient_name'] = patient_name
            session['patient_age'] = patient_age
            session['uploaded_image_filename'] = filename

            return redirect(url_for('detection_module'))
    return render_template('input.html')

@app.route('/detection')
def detection_module():
    if 'user' not in session:
        return redirect(url_for('login'))

    patient_name = session.get('patient_name', 'Unknown')
    patient_age = session.get('patient_age', 'Unknown')
    uploaded_image_filename = session.get('uploaded_image_filename', '')

    tumor_type, confidence_level, confidence_scores = classify_tumor(os.path.join(UPLOAD_FOLDER, uploaded_image_filename))
    graph_path = generate_confidence_graph(confidence_scores)

    explanation, treatment, diagnosis_time = get_treatment_and_explanation(tumor_type, confidence_level)

    session['detection_result'] = {
        'tumor_type': tumor_type,
        'confidence_level': confidence_level,
        'patient_name': patient_name,
        'patient_age': patient_age,
        'uploaded_image_filename': uploaded_image_filename
    }

    return render_template('detection.html', patient_name=patient_name, patient_age=patient_age, tumor_type=tumor_type, confidence_level=confidence_level, uploaded_image_filename=uploaded_image_filename, graph_path=graph_path, treatment=treatment)

@app.route('/generate_pdf_report')
def generate_pdf_report():
    result = session['detection_result']
    pdf_path = generate_pdf(result['patient_name'], result['patient_age'], result['tumor_type'], result['confidence_level'], *get_treatment_and_explanation(result['tumor_type'], result['confidence_level']), result['uploaded_image_filename'])
    return send_file(pdf_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
