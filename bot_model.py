# bot_model.py
# TF-IDF Machine Learning Model for Coconut Farming Bot
# Train once, use forever - No API, No internet needed

import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from coconut_data import QA_DATA

MODEL_PATH = "coconut_bot_model.pkl"

def train_model():
    """Train TF-IDF model on coconut Q&A data and save it"""
    print("Training Coconut Expert Bot...")

    all_questions = []
    all_answers = []
    answer_indices = []

    for idx, item in enumerate(QA_DATA):
        for question in item["questions"]:
            all_questions.append(question.lower())
            all_answers.append(item["answer"])
            answer_indices.append(idx)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        stop_words=None,
        analyzer='word'
    )

    question_vectors = vectorizer.fit_transform(all_questions)

    model_data = {
        "vectorizer": vectorizer,
        "question_vectors": question_vectors,
        "all_questions": all_questions,
        "all_answers": all_answers,
        "answer_indices": answer_indices,
        "qa_data": QA_DATA
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)

    print(f"Model trained successfully!")
    print(f"Total questions trained: {len(all_questions)}")
    print(f"Total answer categories: {len(QA_DATA)}")
    print(f"Model saved to: {MODEL_PATH}")

    return model_data

def load_model():
    """Load trained model from file"""
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Training new model...")
        return train_model()

    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)

    print("Coconut Bot model loaded successfully!")
    return model_data

def get_answer(user_message, model_data, threshold=0.15):
    """Get answer for user question using TF-IDF cosine similarity"""
    vectorizer = model_data["vectorizer"]
    question_vectors = model_data["question_vectors"]
    all_answers = model_data["all_answers"]

    user_vector = vectorizer.transform([user_message.lower()])
    similarities = cosine_similarity(user_vector, question_vectors)[0]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    print(f"DEBUG - Query: '{user_message}' | Best score: {best_score:.3f}")

    if best_score >= threshold:
        return all_answers[best_idx]
    else:
        return get_fallback_response(user_message)

def get_fallback_response(user_message):
    """Friendly fallback when question not understood"""
    msg = user_message.lower()

    if any(word in msg for word in ["disease", "problem", "sick", "dying", "rot", "spot", "pest"]):
        return """I noticed you have a disease or pest concern. Please describe more specifically:

- Which part of tree is affected? (leaves, stem, crown, roots, nuts)
- What does it look like? (color, texture, smell)
- When did you first notice it?

Common diseases I can help with:
- Bud Rot - Ask: "what is bud rot treatment"
- Stem Bleeding - Ask: "stem bleeding treatment"  
- Leaf Blight - Ask: "brown spots on leaves treatment"
- Rhinoceros Beetle - Ask: "how to control rhinoceros beetle"
- Root Wilt - Ask: "root wilt treatment" """

    elif any(word in msg for word in ["fertilizer", "nutrition", "manure", "feed", "npk"]):
        return """For fertilizer questions, please mention:
- Tree age (young 1-3 years / bearing 8+ years)
- Current problem (low yield, yellow leaves, etc.)

Ask specifically:
- "fertilizer dose for 10 year old coconut tree"
- "organic fertilizer for coconut"
- "potassium deficiency coconut treatment" """

    elif any(word in msg for word in ["price", "sell", "market", "money", "income"]):
        return """For market and pricing information, ask:
- "coconut price today Tamil Nadu"
- "where to sell coconut for best price"
- "coconut market rate Pollachi"
- "government scheme coconut farmer subsidy" """

    else:
        return """I am your Coconut Farming Expert for Tamil Nadu. I can help with:

DISEASES: Bud rot, Stem bleeding, Leaf blight, Beetle, Mite
FARMING: Fertilizer, Irrigation, Planting, Harvesting
BUSINESS: Market prices, Government schemes, Intercropping

Try asking:
- "my coconut leaves are turning yellow"
- "how to increase coconut yield"
- "what is the coconut price today"
- "fertilizer schedule for coconut tree"
- "government subsidy for coconut farming" """

_model_data = None

def get_bot_reply(user_message):
    """Main function called by Flask app"""
    global _model_data

    if _model_data is None:
        _model_data = load_model()

    if not user_message or len(user_message.strip()) < 2:
        return "Please ask a question about coconut farming!"

    return get_answer(user_message.strip(), _model_data)

if __name__ == "__main__":
    model = train_model()
    print("\n--- Testing Bot ---")
    test_questions = [
        "my coconut leaves are turning yellow",
        "how to treat bud rot",
        "how much fertilizer for coconut",
        "what is coconut price today",
        "how to control rhinoceros beetle",
        "government scheme for coconut farmer"
    ]
    for q in test_questions:
        print(f"\nQ: {q}")
        answer = get_answer(q, model)
        print(f"A: {answer[:100]}...")
