# AI-Virtual-Career-Counsellor

An NLP-powered AI chatbot built using `SentenceTransformers` that provides smart, context-aware **career guidance** based on user queries. Perfect for students and job seekers looking for personalized suggestions on their career path.

---

## 💡 Features

- 🔍 Query understanding with Sentence-BERT
- 💬 Suggests career answers from a pre-curated dataset
- 📈 Cosine similarity matching
- 🌐 Beautiful **Streamlit** web interface
- ⚡ Fast and scalable using precomputed embeddings

---

## 📁 Dataset

The dataset includes questions and answers related to career options and paths. It is stored in:

- `career_df.pkl` – Pandas DataFrame with `question` and `answer` columns.
- `career_embeddings.npy` – NumPy array of precomputed embeddings.
- `career_embedding_model/` – Trained `SentenceTransformer` model (e.g., `paraphrase-MiniLM-L6-v2`).

You can further expand it using your own FAQ or scrape public Q&A sites.

---

## 🧠 Technologies Used

| Component | Description |
|----------|-------------|
| **NLP** | Sentence-BERT (via `sentence-transformers`) |
| **Vector Similarity** | Cosine Similarity using `scikit-learn` |
| **Frontend** | Streamlit |
| **Data Storage** | Pickle + NumPy |

---
## visualizations

![image](https://github.com/user-attachments/assets/f79b325d-e297-4bed-969b-a0c6eb33238f)
![image](https://github.com/user-attachments/assets/843ac13d-9c0f-45a0-b03e-614602d019c3)




### 🔘 Homepage

![image](https://github.com/user-attachments/assets/c91f0f9a-0fc9-432f-8451-dbf88629f09b)


### 🔍 User Enters a Query

![image](https://github.com/user-attachments/assets/725db355-1e6d-415a-8ea4-87cedc2bd3df)

### 🤖 AI Returns Suggestions

![image](https://github.com/user-attachments/assets/d3a5e1d5-0056-4864-be5f-f066dac589ef)




---

## 🚀 Run the App

```bash
streamlit run "D:\ML PROJECTS\AI Virtual Career Counsellor\career_counsellor_app.py"
📊 Evaluation
Although this is a retrieval-based system (not classification), you can evaluate semantic match quality using:

python
Copy
Edit
from sklearn.metrics import precision_score, recall_score, f1_score

# Assuming you have a test set with (true_question, expected_answer)
# and you're retrieving top_k answers:
# Compute precision@k, recall@k or use ranking metrics like MRR, nDCG
✅ Example Use Cases
🤖 College students unsure about courses

👨‍💻 Working professionals exploring new roles

🎓 School counselors assisting students

📌 To-Do / Improvements
 Add resume parsing to suggest roles based on CV

 Add field/category filtering (Engineering, Medical, etc.)

 Add voice-based input (SpeechRecognition + TTS)

🙌 Author
V Yuvan Krishnan

Made with ❤️ using Python & Streamlit.
