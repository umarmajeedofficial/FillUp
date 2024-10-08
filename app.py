import os
import warnings
import torch
import gradio as gr
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import pdfplumber
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

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
def transcribe_audio(file):
    result = whisper_pipe(file)
    return result['text']

# Function to extract text and questions from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    questions = []
    with pdfplumber.open(pdf_file) as pdf:
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
            outputs = flan_t5_model.generate(**inputs, max_length=100)

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
def save_responses_to_pdf(responses, output_pdf_path):
    document = SimpleDocTemplate(output_pdf_path, pagesize=letter)
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

# Gradio interface function
def process_files(audio_files, pdf_file):
    responses = []
    for audio_file in audio_files:
        # Transcribe audio
        transcribed_text = transcribe_audio(audio_file.name)
        # Extract text and form fields from PDF
        pdf_text, pdf_questions = extract_text_from_pdf(pdf_file.name)
        # Generate form data
        form_data = generate_form_data(transcribed_text, pdf_questions)
        responses.append(form_data)
    
    # Save all responses to a PDF
    output_pdf_path = "output.pdf"
    save_responses_to_pdf(responses, output_pdf_path)
    
    # Return the PDF path and the generated responses
    return output_pdf_path, "\n\n".join(responses)

# Gradio interface definition
interface = gr.Interface(
    fn=process_files,
    inputs=[
        gr.Files(label="Upload Audio Dataset"),
        gr.File(label="Upload PDF File with Questions")
    ],
    outputs=[
        gr.File(label="Download Output PDF"),
        gr.Textbox(label="Generated Responses", lines=20, placeholder="The responses will be shown here...")
    ],
    title="FillUp by Umar Majeed",
    description="""This is a beta version of FillUp, an application designed to auto-fill predefined forms using call data. 
    Upload the audio files from which you want to extract text and a PDF form that contains the questions to be answered. 
    At the end, you will receive a PDF file with the responses.
    For reference, you can download a sample form from [https://drive.google.com/drive/folders/13LolIqxufzysqNoGMfuCAvpA9AkbRfL7?usp=drive_link]. Use this dummy data to understand how the model works."""
)

# Launch the interface
interface.launch()
