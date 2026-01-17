# 🎬 MovieMind: Project Technical Report

**Date:** Jan 16, 2026
**Status:** Operational (v1.4)

## 1. Project Overview
MovieMind is an AI-powered movie recommendation system that combines **Reinforcement Learning (RL)** with **Large Language Models (LLM)** to provide personalized, mood-based movie suggestions. Unlike traditional collaborative filtering, it understands natural language user inputs and dynamically adapts to the user's current emotional state.

## 2. Technical Architecture

### 2.1 Technology Stack
*   **Backend:** Python 3.12+, FastAPI
*   **Database:** MongoDB (via Motor & Beanie ODM)
*   **AI/ML:**
    *   **Agent:** Deep Q-Network (PyTorch) for policy learning.
    *   **NLP:** Google Gemini/Gemma API (`google/gemma-3-27b-it`) for mood extraction.
    *   **Data:** MovieLens 1M Dataset.
*   **Frontend:** HTML5, CSS3 (Custom Design), Vanilla JavaScript.
*   **Serving:** Uvicorn (ASGI Server).

### 2.2 Core Components

#### A. The Recommendation Engine (`src/services.py`)
The heart of the system uses a hybrid approach:
1.  **State Representation:** Combines the user's interaction history (last 5 movies) + Current Mood Vector.
2.  **DQN Agent:** Takes the state vector and predicts Q-values (expected reward) for all movies.
3.  **Strict Filtering Pipeline:**
    *   **Exclusion:** Fetches user's **entire** rating history to ensure zero repeats.
    *   **Relevance:** detailed dot-product matching between movie genres and mood vector.
4.  **Fallback Mechanism:** If the AI agent fails to propose valid candidates (e.g. strict filtering removes all), the system automatically falls back to a direct database search for the requested genre, ensuring the user always gets 5 recommendations.

#### B. The "Brain": Detailed Model Architecture (`src/agent/dqn.py`)
The AI agent is a **Deep Q-Network (DQN)** built with PyTorch. It does not just classify movies; it learns a "policy" to maximize user satisfaction.

*   **Structure**: 3-Layer Feed-Forward Neural Network.
    *   **Input Layer**: Size depends on state (History Length × 18 Genres + 18 Mood Genres). Typically `(5 * 18) + 18 = 108` neurons.
    *   **Hidden Layers**: Two dense layers of **128 neurons** each with ReLU activation.
    *   **Output Layer**: One neuron per movie in the database (approx 3,700 neurons), representing the Q-value (expected reward) for recommending that specific movie.
*   **Training Methodology**:
    *   **Experience Replay**: Stores past interactions `(state, action, reward, next_state)` in a buffer (size 10,000) to break data correlation.
    *   **Target Network**: Uses a separate, slowly updating network to calculate target values, stabilizing training.
    *   **Loss Function**: Mean Squared Error (MSE) between predicted Q and target Q.
    *   **Optimizer**: Adam (`lr=1e-3`).

#### C. Mood Understanding (`src/utils/llm_helper.py`)
*   Uses `google/gemma-3-27b-it` (or `gemini-1.5-flash`) to analyze text (e.g., "I want a space adventure").
*   Maps natural language to a multi-hot vector of 18 specific genres (Action, Sci-Fi, etc.).
*   **Robustness:** Includes a keyword-matching fallback system if the LLM API is unavailable.

#### D. Database Schema (`src/models.py`)
*   **Users:** Stores username, password hash (Argon2), and metadata.
*   **Interactions:** Stores `user_id`, `movie_id`, `rating` (1-5), and `timestamp`.

## 3. Key Features

### ✅ AI-Driven Personalization
Recommendations are not static. The same user asking "I'm sad" vs "I'm happy" gets completely different results, dynamically generated in real-time.

### ✅ Visual Feedback System
*   Interactive Star Ratings with instant visual updates.
*   Custom CSS implementation (`★` text based) to ensure cross-browser/OS compatibility (fixing iOS emoji issues).
*   Gold highlighting for active ratings, dimmed for others.

### ✅ User Experience
*   Clean, "Glassmorphism" inspired dark UI.
*   SPA (Single Page Application) feel with tabbed auth and seamless dashboard transitions.
*   **Zero-Dead-End:** Fallback logic guarantees users always see recommendations.

## 4. Recent Improvements & Fixes
1.  **CORS & Networking:** Fixed communication issues between frontend (`localhost`) and backend (`127.0.0.1`).
2.  **No Repeats:** Implemented strict exclusion logic fetching full user history.
3.  **Visual Polish:** Fixed star rating display issues and removed broken metadata (IMDb links).
4.  **Model Upgrade:** Upgraded LLM pipeline to use `google/gemma-3-27b-it`.

## 5. Future Roadmap
*   **Model Training:** Periodically retrain the DQN agent on new user interaction data.
*   **Rich Media:** Integrate TMDB API for real movie posters (currently text-only cards).
*   **Advanced Moods:** Support complex compound moods (e.g., "Funny but also scary").
