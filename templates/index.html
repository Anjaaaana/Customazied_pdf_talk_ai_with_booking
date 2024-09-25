from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import threading
from pdfminer.high_level import extract_text
import transformers
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from datetime import datetime, timedelta
import re

app = Flask(__name__)

UPLOAD_FOLDER = '/data/ai/gaurav/flask/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB limit

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"Created upload folder: {UPLOAD_FOLDER}")

model_id = '/data/ai/gaurav/llama-2-7b-chat-hf'
hf_auth = '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'
device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'

model = None
tokenizer = None
llm = None
vectorstore = None
chain = None

# Global variable to store processing status
processing_status = {}

# Global variables for user info and appointments
user_info = {}
appointments = {}

def load_model():
    global model, tokenizer, llm
    if model is None or tokenizer is None:
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model_config = transformers.AutoConfig.from_pretrained(
            model_id,
            token=hf_auth
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            token=hf_auth
        )
        model.eval()
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id,
            token=hf_auth
        )
        print(f"Model loaded on {device}")

        stop_list = ['\nHuman:', '\n```\n']
        stop_token_ids = [torch.LongTensor(tokenizer(x)['input_ids']).to(device) for x in stop_list]

        class StopOnTokens(StoppingCriteria):
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                return any(torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all() for stop_ids in stop_token_ids)

        stopping_criteria = StoppingCriteriaList([StopOnTokens()])

        generate_text = transformers.pipeline(
            model=model,
            tokenizer=tokenizer,
            return_full_text=True,
            task='text-generation',
            stopping_criteria=stopping_criteria,
            temperature=0.7,
            max_new_tokens=256,
            repetition_penalty=1.0
        )

        llm = HuggingFacePipeline(pipeline=generate_text)

def extract_text_from_pdf(filepath):
    return extract_text(filepath)

def process_pdf(filepath, filename):
    global vectorstore, chain, processing_status
    
    processing_status[filename] = "Extracting text from PDF"
    text = extract_text_from_pdf(filepath)
    
    processing_status[filename] = "Splitting text"
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    splits = text_splitter.split_text(text)
    documents = [Document(page_content=split, metadata={}) for split in splits]

    processing_status[filename] = "Creating embeddings"
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    processing_status[filename] = "Creating vector store"
    if vectorstore is None:
        vectorstore = FAISS.from_documents(documents, embeddings)
    else:
        vectorstore.add_documents(documents)

    processing_status[filename] = "Loading model"
    load_model()

    processing_status[filename] = "Creating conversation chain"
    chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)

    processing_status[filename] = "Completed"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    try:
        if 'pdfUpload' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['pdfUpload']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Start processing in a separate thread
            threading.Thread(target=process_pdf, args=(filepath, filename)).start()
            
            return jsonify({'message': 'File uploaded successfully, processing started', 'filename': filename}), 202
        else:
            return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        app.logger.error(f"Error uploading PDF: {str(e)}")
        return jsonify({'error': f'Error uploading PDF: {str(e)}'}), 500

@app.route('/processing_status/<filename>')
def get_processing_status(filename):
    status = processing_status.get(filename, "Not found")
    return jsonify({'status': status})

def extract_date(query: str) -> str:
    today = datetime.now()
    if "next monday" in query.lower():
        next_monday = today + timedelta(days=(7 - today.weekday()) % 7)
        return next_monday.strftime("%Y-%m-%d")
    # Add more date extraction logic as needed
    return None

def validate_email(email: str) -> bool:
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def validate_phone(phone: str) -> bool:
    pattern = r'^\+?1?\d{9,15}$'
    return re.match(pattern, phone) is not None

@app.route('/ask', methods=['POST'])
def ask_question():
    global vectorstore, chain, llm, user_info, appointments
    
    query = request.form.get('question')
    filename = request.form.get('filename')
    print("Input question:", query)
    
    if not filename:
        return jsonify({'error': 'No PDF has been uploaded yet. Please upload a PDF first.'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'PDF file not found. Please upload the PDF again.'}), 400

    if vectorstore is None or chain is None:
        return jsonify({'error': 'PDF is still being processed. Please wait and try again.'}), 400

    if "call me" in query.lower():
        # Collect user information
        if "name" not in user_info:
            return jsonify({'answer': "Sure, I'd be happy to call you. First, could you please tell me your name?"})
        elif "phone" not in user_info:
            user_info["name"] = query
            return jsonify({'answer': f"Thank you, {user_info['name']}. What's your phone number?"})
        elif "email" not in user_info:
            if validate_phone(query):
                user_info["phone"] = query
                return jsonify({'answer': "Great, and what's your email address?"})
            else:
                return jsonify({'answer': "That doesn't seem to be a valid phone number. Could you please provide a valid phone number?"})
        else:
            if validate_email(query):
                user_info["email"] = query
                return jsonify({'answer': f"Thank you, {user_info['name']}. We'll call you at {user_info['phone']} or email you at {user_info['email']}. How else can I assist you?"})
            else:
                return jsonify({'answer': "That doesn't seem to be a valid email address. Could you please provide a valid email?"})

    elif "book appointment" in query.lower():
        date = extract_date(query)
        if date:
            appointments[user_info.get("name", "User")] = date
            return jsonify({'answer': f"I've booked your appointment for {date}. Is there anything else I can help you with?"})
        else:
            return jsonify({'answer': "I couldn't understand the date for the appointment. Could you please provide a specific date or day (e.g., 'next Monday')?"})

    else:
        prompt_template = """You are a highly accurate AI assistant tasked with answering questions based on the given context. Use the following pieces of context to answer the question at the end. If the answer is explicitly stated in the context, cite the relevant part. If the answer is not explicitly stated, use your knowledge to provide a relevant response, but make it clear that you're inferring or providing general information. If you absolutely cannot provide an answer based on the context or your knowledge, then say "I don't have enough information to answer this question." Always give amounts in Nepali Rupees (Rs) when applicable. Provide a concise and direct answer without repeating the question or mentioning the context.

Context: {context}

Question: {question}

Answer: """

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )

        try:
            docs = compression_retriever.get_relevant_documents(query)
            docs = docs[:5]  # Use top 5 documents
            
            context = "\n".join([doc.page_content for doc in docs])
            
            result = qa_chain({"input_documents": docs, "question": query}, return_only_outputs=True)
            
            response = result['output_text'].strip()
            if response.lower().startswith("answer:"):
                response = response[7:].strip()
            
            print("Generated response:", response)
            print("Used context:", context)  # Log the used context
            return jsonify({'answer': response})
        except Exception as e:
            print(f"Error during question answering: {str(e)}")
            return jsonify({'error': 'An error occurred while processing your question. Please try again.'}), 500

@app.route('/pdf_status', methods=['GET'])
def pdf_status():
    return jsonify({'pdfs_uploaded': vectorstore is not None, 'pdf_count': len(vectorstore) if vectorstore else 0})

@app.route('/book_appointment', methods=['POST'])
def book_appointment():
    try:
        date = request.form.get('date')
        if not date:
            return jsonify({'error': 'No date provided'}), 400
        
        # You might want to add more validation here
        appointment_date = datetime.strptime(date, "%Y-%m-%d")
        
        # Store the appointment (for now, we'll just print it)
        print(f"Appointment booked for {appointment_date}")
        
        return jsonify({'message': 'Appointment booked successfully'}), 200
    except Exception as e:
        print(f"Error booking appointment: {str(e)}")
        return jsonify({'error': 'Error booking appointment'}), 500

@app.route('/submit_user_info', methods=['POST'])
def submit_user_info():
    try:
        name = request.form.get('name')
        phone = request.form.get('phone')
        email = request.form.get('email')
        
        if not all([name, phone, email]):
            return jsonify({'error': 'Missing required information'}), 400
        
        # You might want to add more validation here
        
        # Store the user information (for now, we'll just print it)
        print(f"User info submitted: Name: {name}, Phone: {phone}, Email: {email}")
        
        return jsonify({'message': 'User information submitted successfully'}), 200
    except Exception as e:
        print(f"Error submitting user information: {str(e)}")
        return jsonify({'error': 'Error submitting user information'}), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Test endpoint is working!'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5028)
