# Portfolio

<center><img src="images/photo-1499951360447-b19be8fe80f5.png"/></center>

---

## Natural language processing (NLP)

### Multimodal QnA System for Personal Injury Case Data (AWS Bedrock + Claude Sonnet 3)

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/RahulChatterjee07/legalmed-qa-system.git)
<div style="text-align: justify"> This project simulates a multimodal Question Answering system for legal and medical documents — designed for workflows such as personal injury cases, disability assessments, and insurance claims. It demonstrates: Claude Sonnet QnA via AWS Bedrock, Retrieval-Augmented Generation (RAG), Prompt templating and structured outputs, Multimodal inputs (text + image). </div>
  
<br>
<center><img src="images/legalmed_qa_output_example_4.png"/><img src="images/legalmed_qa_output_example_6.png"/></center>
<br>

--- 
### Fine-tuning LLMs with AWS Claude Haiku

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/RahulChatterjee07/haiku-finetune-lab.git)
<div style="text-align: justify"> This project simulates the fine-tuning of AWS Claude Haiku for domain-specific tasks like structured information extraction and Q&A.

🔧 Features
📦 Converts training data into Claude-compatible JSONL format
🎯 Simulates fine-tuning on custom tasks
📊 Evaluates performance with basic metrics
🧠 Shows output differences before and after fine-tuning. </div>
  
<br>
<center><img src="images/haiku_token_cost_comparison.png"/><img src="images/haiku_finetune_comparison.png"/></center>
<br>

--- 
### Insurance Claim Auditor – Medical Timeline & Fraud Detection with LLMs

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/RahulChatterjee07/haiku-finetune-lab.git)
<div style="text-align: justify"> This project simulates an advanced LLM-based pipeline for insurance claim analysis, specifically targeting Ontario Claim Forms (OCFs), medical invoices, and duplicate billing detection. The system combines fast inference with Hugging Face Phi-3.5 and structured JSON generation using fine-tuned LLaMA, and is deployable to Amazon SageMaker.
  
🔧 Features: Extracts chronological medical events from unstructured documents
Parses OCF-21, OCF-18 and OCF-23 forms to extract claimed amounts
Identifies invoices and maps them to treatment categories
Detects duplicate or overlapping billings for fraud prevention
Outputs standardized JSON for downstream audit tools </div>
  
<br>
<center><img src="images/insurance_claim_visual_output.png"/></center>
<br>

--- 
### Intelligent Prompt Routing Engine using AWS Textract Layout for LLM Preprocessing

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/RahulChatterjee07/prompt-routing-engine.git)
<div style="text-align: justify"> This project simulates an intelligent routing engine that automatically directs various document types (e.g., forms, reports, assessments, correspondence) to appropriate LLM prompt templates. Inspired by real-world enterprise applications, it uses mocked Textract outputs and a simple rule engine to demonstrate the concept. </div>
  
<br>
<center><img src="images/assessment_output_view_full.png"/><img src="images/correspondence_output_view_full.png"/></center>
<br>

--- 

### Tweet Emotion Recognition with TensorFlow

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://github.com/RahulChatterjee07/Tweet-Emotion-Recognition-with-TensorFlow)
<div style="text-align: justify"> This is a multi class classification problem using a tweet emotion dataset to learn to recognize 6 different emotions ('sadness, 'surprise', 'love', 'anger', 'fear', 'joy) in tweets. A tokenizer is implemented in Tensorflow to perform padding and truncating sequences from tweets. The deep learning model includes Bidirectional long-short term memory(Bidirectional LSTM) and is also implemented in TensorFlow. </div>
  
<br>
<center><img src="images/confusion matrix.png"/></center>
<br>

--- 
## Big Data & Distributed Systems

### Scalable Text & Transaction Intelligence System (spaCy + AWS Quicksight)

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/RahulChatterjee07/Transaction_Insights_Dashboard.git)
<div style="text-align: justify"> In this project, I built a non-LLM analytics pipeline that transforms raw documents and transactional records into structured insights, enabling fraud detection, forecasting, classification, and real-time monitoring. Designed for scalable environments like rideshare, e-commerce, and financial operations. 🔧 Features: Extracts key fields (dates, prices, vendors) from receipts, logs, contracts
📍 Named entity recognition (NER) for org names, locations, references
📂 ML-based document classification: SVM, logistic regression
📈 Statistical forecasting (ARIMA, Holt-Winters) for tickets & transactions.
<br>
<center><img src="images/transaction_dashboard_mock.png"/></center>
<br>
  
---
## Exploratory Data Analysis

### Geocoding and Analyzing San Francisco Building Permit Data (Jan 2021)

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/RahulChatterjee07/Exploratory-data-analysis-using-R-and-Tableau)

<div style="text-align: justify"> In this project, I explored and analyzed more than seven years of the City of San Francisco's building permit data and used the API OpenStreetMap to find the geo coordinates of buildings. After creating a new clean dataset that includes geo coordinates, I used Tableau to visualize and analyze the dataset. </div>
<br>
<center><img src="images/R 1.png"/><img src="images/retl.png"/></center>
<br>

---
### Hyper Parameter Optimization in Artificial Neural Network (ANN) using MNIST Data from sklearn (Jan 2021)

[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1G8tDUuHyqtIXt-AGDxIZmXF7QlgCfJ18#scrollTo=3FQzo9EoObww)

<div style="text-align: justify"> In this project, I implemented different hyperparameter tuning methods (e.g. Grid search, Random search, Hyperband, Bayesian Optimization with Gaussian Processes (BO-GP)) to achieve the optimized set of hyperparameters for the ANN architecture. </div>

  
---
## Time Series Analysis

### Stock Price Prediction of Apple Inc. Using Recurrent Neural Network (Dec 2021)

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://github.com/RahulChatterjee07/LSTM-Time-Series-Analysis/blob/main/StockPricePrediction.py)
<div style="text-align: justify"> The project is about prediction of stock price using deep learning. The dataset consists of Open, High, Low and Closing Prices of Apple Inc. stocks from 3rd january 2011 to 13th August 2021. Two sequential LSTM layers have been combined together and a dense layer is used to build the RNN model using Keras deep learning library. Since this is a regression task, 'linear' activation has been used in final layer. </div>
<br>
<center><img src="images/stock_result.png"/></center>
<br>
  
---
### Deep Convolutional Model to Study Brain-stimulation Induced Features from Human EEG Data (May - Aug 2019)

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/RahulChatterjee07/MITACS_PROJECT-CNN-Classifier)

<div style="text-align: justify"> In this project, I built a Convolutional Neural network-based framework in PyTorch to identify brain stimulation-induced EEG features. EEG data were collected from 30 subjects with 5 different conditions: sham/20Hz/70Hz/individual β/individual γ tACS brain stimulation. [Ref: "Deep Semantic Architecture with discriminative feature visualization for neuroimage analysis", Ghosh A. et al. 2018.</div>
<br>
<center><img src="images/CNN architecture.jpg"/></center>
<br>


---
### Software-hardware Interface for Synchronization of EEG and Force sensor Devices (May - Aug 2019)

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/RahulChatterjee07/MITACS_Project-Real-time-data-visualization)

<div style="text-align: justify">In this project, I designed the software-hardware interface to synchronize the EEG device with the force sensor and also developed experimental tasks to study motor learning characteristics.</div>
<br>
<center><img src="images/Motor task display.jpg"/></center>
<br>
  

---
### Predict Transient Spikes in Time Series Data using Time-delay Embedded (TDE) Hidden Markov Model (Sep 2020 - Present)

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/RahulChatterjee07/Transient-burst-detection)
[![Open Research Poster](https://img.shields.io/badge/PDF-Open_Research_Poster-blue?logo=adobe-acrobat-reader&logoColor=white)](pdf/REPAR_Poster_Rahul.pdf)

<div style="text-align: justify">As part of my Master’s project, I designed a Time-Delay Embedded (TDE) Hidden Markov model to detect transient bursts from the beta frequency range (13 - 30 Hz) of MEG signal. In this study, I have used morlet wavelet transform to extract beta oscillatory envelopes from the raw MEG signal. Also, I am currently developing a Machine learning (ML) pipeline to classify MEG/EEG signal into ’burst’ states in real-time and the results will be further used to design a closed-loop neurofeedback system.</div>
<br>
<center><img src="images/download.png"/></center>
<br>
<center><img src="images/2.png"/></center>
<br>
  
  
---
### Supervised-learning-based Classification of EEG Signals to Predict Mental States (May - Jun 2018)

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/RahulChatterjee07/Emotion_state_classifier)
<div style="text-align: justify"> This project is about mental state classification of human subjects using single channel EEG data. EEG data from 5 subjects were collected during a horror movie, a comedy movie clip and during a mental task. In this study, I have used different supervised machine learning algorithms (such as KNN, SVM, LDA) and studied their performance (precision, recall and F1 score) in classifying different mental states.</div>
<br>
<center><img src="images/Capture 8.PNG"/></center>
<br>

  
---
### Removing Artificial Noise from Acoustic Cardiac Signal (Dec - Jan 2021)

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/RahulChatterjee07/Artifact-remove-from-cardiac-signal)

<div style="text-align: justify">In this project, I used digital signal processing methods to remove artificially induced artifacts from simulated cardiac rhythms.</div>
<br>
<center><img src="images/Signal_1.png"/></center>
<br>

  
---
## Network Analysis

### Community Ranking using Social Network Analysis (Sep 2017 - Apr 2018)

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/RahulChatterjee07/Social-Network-Analysis)

<div style="text-align: justify">In this project, I used the k-clique algorithm to partition the network into communities. The network was formed using Facebook data, collected from survey participants using the Facebook app. The dataset includes node features (profiles), circles, and ego networks. Community ranking was obtained based on the following methods: Method-1: No of social active nodes present in the community. Method-2: On the basis of the value of the function W(m) of nodes present in the community. W(m)=0.5A1(m)+0.5A2(m), where A1(m) is the average number of posts posted per week by user m and A2(m) is the average number of shares plus the number of comments plus the number of likes for each of his posts. </div>
<br>
<center><img src="images/community_3.png"/></center>
<br>

---



