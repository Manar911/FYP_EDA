# EDA – Emergency Diversion Assistant

## AI-Powered Emergency Airport Recommendation System

EDA (Emergency Diversion Assistant) is an embedded AI decision-support system designed to assist pilots during in-flight emergencies by recommending and ranking suitable diversion airports in real time.

The system combines deterministic aviation safety logic, machine learning ranking, and explainable AI to provide safer and more transparent diversion recommendations.

EDA is designed for offline edge deployment on a Raspberry Pi 5 with a touchscreen interface.

---

# Project Overview

During flight emergencies, pilots must rapidly evaluate diversion options under time pressure. EDA helps simplify this process by analysing nearby airports and recommending the safest available alternatives based on operational and emergency-specific constraints.

The system evaluates factors such as:

- Airport distance
- Runway suitability
- Aircraft diversion capability
- Emergency type
- Medical availability
- Rescue services
- Airport operational restrictions
- Safety feasibility constraints

EDA acts as an advisory decision-support system and does not replace pilot judgement.

---

# Key Features

- Offline embedded deployment
- Raspberry Pi 5 compatible
- Touchscreen-based interface
- Deterministic feasibility filtering
- Machine learning airport ranking
- Explainable AI recommendations
- Safety-focused fallback mechanism
- Emergency-specific airport evaluation
- Pilot confirmation logging
- Real-time recommendation generation

---

# System Architecture

EDA follows a layered safety-focused architecture:

## 1. Safety & Validation Layer

- Validates pilot emergency scenarios
- Removes unsafe or infeasible airports
- Enforces aviation safety constraints

## 2. AI Inference Layer

- Uses a trained LightGBM model
- Ranks feasible diversion airports
- Generates confidence-based recommendations

## 3. Presentation Layer

- Displays ranked airport recommendations
- Shows explanation points for transparency
- Supports touchscreen interaction

If the machine learning model becomes unavailable, the system automatically falls back to deterministic ranking.

---

# Core Pipeline

Scenario Input
      ↓
Validation
      ↓
Feature Engineering
      ↓
Feasibility Filtering
      ↓
Airport Ranking
      ↓
Explainable Recommendations
      ↓
Pilot Confirmation

Technologies Used
Software
Python
LightGBM
scikit-learn
pandas
NumPy
PySide6
Hardware
Raspberry Pi 5
Raspberry Pi OS
7-inch touchscreen display
Machine Learning Approach

EDA uses supervised machine learning trained on synthetic aviation emergency scenarios.

The model learns to imitate an expert deterministic ranking system and prioritise the safest diversion airports based on structured aviation-related features.

The final deployed model uses LightGBM due to:

High ranking performance
Efficient inference speed
Suitability for embedded systems
Strong performance on structured tabular data

Evaluation metrics include:

Top-1 Accuracy
Mean Reciprocal Rank (MRR)
Explainable AI

EDA provides human-readable explanations for recommendations to improve transparency and pilot trust.

Example explanation factors include:

Shorter diversion distance
Suitable runway margin
Medical facility availability
Rescue service availability
Emergency-specific suitability
Operational safety status

The project also uses SHAP analysis during development for offline model interpretability.

Embedded Deployment

EDA is designed for fully offline deployment using:

Raspberry Pi 5
Local airport database
Local ML model storage
Touchscreen cockpit-style interface

The system does not require internet connectivity during operation.

Safety and Integrity

EDA includes multiple safety-focused mechanisms:

Feasibility filtering always executes before ranking
Unsafe airports are rejected before recommendation
SHA-256 integrity verification for critical files
Deterministic fallback ranking if ML fails
Pilot decision logging for traceability
Repository Structure
FYP_EDA/
│
├── EDA_CoreEngine/
│   ├── scenario.py
│   ├── validation.py
│   ├── features.py
│   ├── filter.py
│   ├── ranking.py
│   ├── explanation.py
│   ├── pipeline.py
│   ├── airport_db.py
│   └── integrity.py
│
├── .gitignore
└── README.md
Project Status

This project was developed as a Final Year Project prototype demonstrating the integration of:

Embedded Systems
Machine Learning
Explainable AI
Aviation Decision Support
Disclaimer

EDA is an academic prototype developed for educational and research purposes only.

It is not certified for real-world aviation operation. Final flight decisions must always remain with qualified aviation professionals.

Author
Manar Almosawi

BSc Software Engineering
British University of Bahrain
University of Salford
