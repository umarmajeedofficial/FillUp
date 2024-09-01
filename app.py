import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForSpeechSeq2Seq, AutoProcessor
import pdfplumber
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io

# Function to load models
def load_models():
    try:
        flan_t5_model_id = "google/flan-t5-large"
        whisper_model_id = "openai/whisper-medium"

        flan_t5_tokenizer = T5Tokenizer.from_pretrained(flan_t5_model_id)
        flan_t5_model = T5ForConditionalGeneration.from_pretrained(flan_t5_model_id)
        
        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(whisper_model_id)
        whisper_processor = AutoProcessor.from_pretrained(whisper_model_id)
        
        return flan_t5_tokenizer, flan_t5_model, whisper_model, whisper_processor
    except ImportError as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# Load models
flan_t5_tokenizer, flan_t5_model, whisper_model, whisper_processor = load_models()

# Function to transcribe audio files
def transcribe_audio(file):
    audio_file = io.BytesIO(file.read())
    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=whisper_processor.tokenizer,
        feature_extractor=whisper_processor.feature_extractor,
        device=device
    )
    result = whisper_pipe(audio_file)
    return result['text']

# Function to extract text and questions from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    questions = []
    pdf = pdfplumber.open(pdf_file)
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
            # Extract questions based on numbering
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

        # Tokenize the input text
        inputs = flan_t5_tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True)

        # Generate the answer using the model
        with torch.no_grad():
            outputs = flan_t5_model.generate(**inputs, max_length=100)  # Adjust max_length as needed

        # Decode the generated text
        generated_text = flan_t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Handle incomplete or missing answers
        if not generated_text.strip():
            generated_text = "The answer to this question is not present in the script."
        elif len(generated_text.strip()) < 10:  # Arbitrary threshold for short/incomplete answers
            input_text = f"""Based on the following transcript, provide a more detailed answer to the question.\n\nText: {text}\n\nQuestion: {question}\n\nAnswer:"""
            inputs = flan_t5_tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True)
            outputs = flan_t5_model.generate(**inputs, max_length=100)
            generated_text = flan_t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Append question and response
        responses.append(f"Question: {question}\nAnswer: {generated_text.strip()}")

    return "\n\n".join(responses)

# Function to save responses to PDF
def save_responses_to_pdf(responses, output_pdf):
    document = SimpleDocTemplate(output_pdf, pagesize=letter)
    styles = getSampleStyleSheet()

    # Custom style for responses
    response_style = ParagraphStyle(
        name='ResponseStyle',
        parent=styles['BodyText'],
        fontSize=10,
        spaceAfter=12
    )

    content = []
    for index, response in enumerate(responses, start=1):
        # Add the response number and content
        heading = Paragraph(f"<b>File {index}:</b>", styles['Heading2'])
        response_text = Paragraph(response.replace("\n", "<br/>"), response_style)

        content.append(heading)
        content.append(Spacer(1, 6))  # Space between heading and response
        content.append(response_text)
        content.append(Spacer(1, 18))  # Space between responses

    document.build(content)

# Streamlit UI
st.title("FillUp by Umar Majeed")

# Upload multiple audio files
audio_files = st.file_uploader("Upload multiple audio files", type=["wav", "mp3"], accept_multiple_files=True)

# Upload PDF file
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

if st.button("Process"):
    if not audio_files or not pdf_file:
        st.error("Please upload both audio and PDF files.")
    else:
        # Process PDF
        pdf_text, pdf_questions = extract_text_from_pdf(pdf_file)

        # Process audio files
        responses = []
        for audio_file in audio_files:
            # Transcribe audio
            transcribed_text = transcribe_audio(audio_file)

            # Generate form data
            form_data = generate_form_data(transcribed_text, pdf_questions)
            responses.append(form_data)

        # Show results
        st.write("### Results")
        st.write("\n\n".join(responses))

        # Save results to PDF and offer download
        output_pdf = io.BytesIO()
        save_responses_to_pdf(responses, output_pdf)
        output_pdf.seek(0)
        st.download_button("Download Results as PDF", output_pdf, file_name="responses.pdf")
