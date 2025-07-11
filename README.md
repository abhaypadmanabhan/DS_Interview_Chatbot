# DS Interview Chatbot: Fine-Tuning DialoGPT for Data Science Q&A

## Project Type: Predictive Modeling / End-to-End Data Pipeline (with Interactive Demo)

This project showcases the development of a conversational AI chatbot specialized in answering Data Science interview questions. By fine-tuning a powerful pre-trained language model (DialoGPT) on custom interview data, this project demonstrates end-to-end capabilities in natural language processing, model training, evaluation, and deploying an interactive web interface.

## The Business Problem

Aspiring data scientists and machine learning engineers often face a significant challenge in preparing for technical interviews, which require a broad and deep understanding of concepts, algorithms, and practical applications. Traditional study methods can be inefficient.

This project addresses the problem: **"How can we provide an accessible, interactive, and intelligent tool to help individuals practice and master Data Science interview questions through a conversational interface?"**

## Your Approach and Process

My approach involved building a full pipeline for training, evaluating, and deploying a specialized chatbot.

1.  **Data Collection & Cleaning**:
    * **Source**: A raw text file (`ds_interview_data.txt`) containing Data Science Q&A pairs was used as the primary training data.
    * **Preprocessing**: The data underwent a cleaning process to remove excessive newlines and strip whitespace, ensuring a clean and consistent input format for model training. This step mimics handling real-world, often messy, text data.
    * *(Optional: If real custom data was collected, mention: "Custom interview Q&A pairs were collected, along with mock multi-turn interview transcripts, to create a highly relevant and contextual training dataset.")*

2.  **Model Fine-Tuning**:
    * **Base Model**: I utilized `microsoft/DialoGPT-medium`, a pre-trained transformer model optimized for conversational AI, as the foundation.
    * **Training Framework**: The model was fine-tuned using the `transformers` library's `Trainer` API, which simplifies the training loop management.
    * **Hyperparameters**:
        * **Optimizer**: AdamW was chosen for its effectiveness in transformer models, handling weight decay efficiently.
        * **Learning Rate**: Set to a range of `1e-5` to `5e-5` to ensure stable and effective learning from the fine-tuning data without disrupting the pre-trained weights too much.
        * **Batch Size**: Adjusted to manage GPU memory efficiently (e.g., `1` per device with `gradient_accumulation_steps=2`).
        * **Number of Epochs**: Trained for `5` epochs to allow sufficient learning while mitigating overfitting.
        * **Stopping Criteria (Implicit)**: While not explicitly coded for early stopping, the notebook demonstrates a complete training run. For production, early stopping based on validation loss would be implemented.

3.  **Conversational Interface Development**:
    * **Libraries**: `Gradio` was used to rapidly build an interactive web interface for the chatbot, making it easily accessible for testing and demonstration.
    * **Features**: The interface supports both single-turn and multi-turn conversations, allowing users to ask follow-up questions and experience a more natural dialogue flow. Parameters like `max_length` and `max_new_tokens` are exposed for user control over response length.

4.  **Model Evaluation**:
    * **Metrics**: The chatbot's performance was quantitatively assessed using:
        * **BERTScore**: A metric that measures semantic similarity between generated and reference sentences, capturing the meaning rather than just exact word overlap.
        * **ROUGE Score**: A set of metrics (e.g., ROUGE-1, ROUGE-2, ROUGE-L) that evaluate the overlap of n-grams and longest common subsequences between generated and reference summaries/answers.
    * **Test Data**: A dedicated set of `test_samples` (question-reference answer pairs) was used to systematically evaluate the model's ability to generate relevant and accurate responses.

## Key Findings and Insights

* **Semantic Understanding**: BERTScore results indicate the model's ability to grasp the semantic content of data science questions and generate responses that are meaningfully similar to reference answers, even if not phrased identically.
* **Coherence in Conversation**: The multi-turn capability, combined with techniques like `no_repeat_ngram_size` and `top_k`/`top_p` sampling, allowed the chatbot to maintain context and generate coherent follow-up responses, demonstrating its conversational prowess.
* **Interactive Learning Potential**: The Gradio interface proved highly effective in creating an accessible tool for iterative testing and user interaction, highlighting its potential as a self-study aid.
* **Performance Metrics**:
    * **BERTScore F1**: [e.g., `0.336`, `0.471`] (Specific values from the output for illustration)
    * **ROUGE-L**: [e.g., `0.090`] (Specific values from the output for illustration)
    These scores, while not indicative of human-level performance, provide a baseline for future improvements and demonstrate the model's ability to generate relevant text.

## Business Impact and Recommendations

This project delivers tangible business value by:

* **Streamlining Interview Preparation**: Offers an on-demand, interactive study companion, reducing the need for human tutors and providing immediate feedback.
* **Improving Candidate Readiness**: Helps aspiring data scientists strengthen their understanding of core concepts, leading to better interview performance and higher quality hires.
* **Scalable Knowledge Access**: Can be extended to serve as an internal knowledge base for teams, providing quick answers to technical questions.
* **Foundation for Advanced AI Tutors**: The established fine-tuning and evaluation pipeline can be adapted to build more sophisticated AI tutors capable of personalized learning paths.
* **Cost Efficiency**: Automates a part of the interview preparation process, potentially reducing training costs for educational institutions or recruitment firms.

## Technical Implementation

The project is implemented as a Jupyter Notebook, detailing each step from data loading to model deployment and evaluation.

### Repository Structure
.
├── ML_HW6_abhay.ipynb   # The main Jupyter Notebook
├── ds_interview_data.txt # Example raw data (or placeholder for actual data)
└── README.md            # This README file
### Setup and Installation

To run this project, you will need Python and the following libraries. It is recommended to use a virtual environment.

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/your-username/DS_Interview_Chatbot.git](https://github.com/your-username/DS_Interview_Chatbot.git)
    cd DS_Interview_Chatbot
    ```
2.  **Install Dependencies**:
    ```bash
    pip install transformers torch sentencepiece nltk sacrebleu pandas numpy evaluate gradio bert_score rouge_score
    ```
3.  **Data File**: Ensure you have a `ds_interview_data.txt` file in the `/content/` directory, or modify the `input_file_path` in the notebook to point to your data. This file should contain raw text data for fine-tuning.
4.  **Run the Notebook**: Open `ML_HW6_abhay.ipynb` with Jupyter Notebook or JupyterLab:
    ```bash
    jupyter notebook ML_HW6_abhay.ipynb
    ```
    Execute the cells sequentially to perform data cleaning, model fine-tuning, testing, and launch the Gradio interface.

### Code Quality & Reproducibility

The notebook is designed for clarity, with steps logically organized and comments provided where necessary. The `Trainer` API ensures a standard and reproducible training process.

---
