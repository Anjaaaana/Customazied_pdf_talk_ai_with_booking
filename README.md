# Flask PDF Processing and Question-Answering Application

This is a Flask-based web application that enables users to upload PDF documents, processes them by extracting text, and allows users to ask questions based on the content of the PDFs using a conversational retrieval chain. The backend utilizes a pre-trained language model (`llama-2-7b-chat-hf`) for generating answers, combined with FAISS for efficient document search.

## Features

- **Upload PDF**: Users can upload PDFs for text extraction.
- **Document Splitting**: The text is split into chunks to optimize question-answering performance.
- **Embedding Creation**: Uses Hugging Face models to create document embeddings for efficient retrieval.
- **Conversational Retrieval Chain**: Provides a question-answer interface to get answers based on the uploaded documents.
- **Callback and Appointment System**: Users can book appointments or request callbacks with email and phone validation.
- **PDF Status**: Tracks how many PDFs have been uploaded and processed.

## Requirements

Before running the application, ensure you have the following installed:

- **Python 3.8+**
- **Pip** (for managing Python packages)
- **A compatible GPU** (if using CUDA for model inference)
- **Hugging Face Access** (for accessing `llama-2-7b-chat-hf` or similar models)
- **CUDA** (optional, for GPU-accelerated processing)

![image](https://github.com/user-attachments/assets/2a8aa3e9-08ac-4714-9f34-81f9c2380eb3)

## Booking/Appoinments
![image](https://github.com/user-attachments/assets/a1c255d4-401e-4fbb-be9b-bf37182e260a)
![image](https://github.com/user-attachments/assets/27ef78a5-e5a9-44ab-8ed0-2a080608e194)


# output
![image](https://github.com/user-attachments/assets/8c261eef-8e5c-4ba1-94a7-0e61f6a514b2)


