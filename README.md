## 🚀 Catalyzing Content Generation and Automation: Empowering Diverse Written Content Creation through Generative AI 

This repository hosts a comprehensive research project and practical implementation exploring how Generative AI can empower the creation of diverse written content. The project systematically evaluates powerful language models like GPT-2, GPT-Neo, and OPT-125m, focusing on automated creativity assessment and practical deployment through an intuitive web application.

### 🌟 Project Overview

Title: Catalyzing Content Generation and Automation: Empowering Diverse Written Content Creation through Generative AI

Conducted by: Lam Hao Fah

Supervisor: Dr. Owen Noel Newton Fernando

Institution: School of Computing and Data Science, Nanyang Technological University, Singapore

This research addresses key limitations of traditional content creation by harnessing the capabilities of large language models (LLMs), evaluating their creative potential systematically, and demonstrating real-world applications through an interactive web prototype.

### 📂 Project Structure

```
.
├── web_app/                        # Web application prototype
│   ├── backend/                    # Flask backend service
│   │   ├── __pycache__/
│   │   ├── .env                    # Environment variables
│   │   ├── .gitignore
│   │   ├── app.py                 # Backend API endpoints
│   │   ├── functions.py           # Utility functions for backend
│   │   └── requirements.txt       # Python dependencies
│   ├── frontend/                   # React frontend interface
│   │   ├── node_modules/
│   │   ├── public/
│   │   ├── src/
│   │   ├── .gitignore
│   │   ├── package-lock.json
│   │   └── package.json
│   └── README.md                   # README for the web_app module
├── .gitattributes
├── .gitignore
├── README.md                       # Main project documentation
├── gpt-2-model.ipynb               # GPT-2 training notebook
├── gpt-neo-model.ipynb             # GPT-Neo training notebook
├── my_dataset.csv                  # Dataset used for training
└── opt-model.ipynb                 # OPT-125 training notebook
```

### 🛠 Model Weights

Due to size limitations, the model weights are hosted on Hugging Face.
```
https://huggingface.co/haofah14/models
```