import os
import warnings
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import pdfplumber
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import streamlit as st
import numpy as np
import io
import soundfile as sf

# Suppress warnings globally
warnings.filterwarnings("ignore")

# Setup models
device = "cuda:0" if torch.cuda.is_available() else "cpu"
whisper_model_id = "openai/whisper-medium"

# Load Whisper model and processor
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(whisper_model_id)
whisper_processor = AutoProcessor.from_pretrained(whisper_model_id)

# Create Whisper pipeline
whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=whisper_processor.tokenizer,
    feature_extractor=whisper_processor.feature_extractor,
    device=device
)

# Setup FLAN-T5 model and tokenizer
flan_t5_model_id = "google/flan-t5-large"
flan_t5_tokenizer = T5Tokenizer.from_pretrained(flan_t5_model_id)
flan_t5_model = T5ForConditionalGeneration.from_pretrained(flan_t5_model_id)

# Function to transcribe audio files
def transcribe_audio(audio_file):
    try:
        with io.BytesIO(audio_file.read()) as file:
            audio_data, sample_rate = sf.read(file, format='mp3') if audio_file.type == 'audio/mpeg' else sf.read(file)
        inputs = whisper_processor(audio_data, sampling_rate=sample_rate, return_tensors="pt")
        result = whisper_pipe(inputs)
        return result['text']
    except Exception as e:
        st.error(f"Error in audio transcription: {e}")
        return "Error during transcription"

# Function to extract text and questions from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    questions = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
                lines = page_text.split("\n")
                for line in lines:
                    if line.strip() and line.strip()[0].isdigit():
                        questions.append(line.strip())
    return text, questions

# Function to generate form data with FLAN-T5
def generate_form_data(text, questions):
    responses = []
    for question in questions:
        input_text = f"""The following text is a transcript from an audio recording. Read the text and answer the following question in a complete sentence.\n\nText: {text}\n\nQuestion: {question}\n\nAnswer:"""

        inputs = flan_t5_tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True)
        with torch.no_grad():
            outputs = flan_t5_model.generate(**inputs, max_length=100)

        generated_text = flan_t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

        if not generated_text.strip():
            generated_text = "The answer to this question is not present in the script."
        elif len(generated_text.strip()) < 10:
            input_text = f"""Based on the following transcript, provide a more detailed answer to the question.\n\nText: {text}\n\nQuestion: {question}\n\nAnswer:"""
            inputs = flan_t5_tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True)
            outputs = flan_t5_model.generate(**inputs, max_length=100)
            generated_text = flan_t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

        responses.append(f"Question: {question}\nAnswer: {generated_text.strip()}")

    return "\n\n".join(responses)

# Function to save responses to PDF
def save_responses_to_pdf(responses, output_pdf_path):
    document = SimpleDocTemplate(output_pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()

    response_style = ParagraphStyle(
        name='ResponseStyle',
        parent=styles['BodyText'],
        fontSize=10,
        spaceAfter=12
    )

    content = []
    for index, response in enumerate(responses, start=1):
        heading = Paragraph(f"<b>File {index}:</b>", styles['Heading2'])
        response_text = Paragraph(response.replace("\n", "<br/>"), response_style)

        content.append(heading)
        content.append(Spacer(1, 6))
        content.append(response_text)
        content.append(Spacer(1, 18))

    document.build(content)

# Streamlit UI
st.title("FillUp by Umar Majeed")

# Upload audio files
audio_files = st.file_uploader("Upload multiple audio files", type=["wav", "mp3"], accept_multiple_files=True)

# Upload PDF file
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

if st.button("Process"):
    if audio_files and pdf_file:
        responses = []
        pdf_text, pdf_questions = extract_text_from_pdf(pdf_file)

        for audio_file in audio_files:
            transcribed_text = transcribe_audio(audio_file)
            form_data = generate_form_data(transcribed_text, pdf_questions)
            responses.append(form_data)
            st.write(f"File {len(responses)}:\n{form_data}\n")

        output_pdf_path = "/tmp/response_output.pdf"
        save_responses_to_pdf(responses, output_pdf_path)
        st.write("Responses have been generated. You can download the result below.")

        with open(output_pdf_path, "rb") as file:
            st.download_button(
                label="Download PDF",
                data=file,
                file_name="response_output.pdf",
                mime="application/pdf"
            )
    else:
        st.error("Please upload both audio files and a PDF file.")
