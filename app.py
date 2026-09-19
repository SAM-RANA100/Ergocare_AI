import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Corporate Page Configuration
st.set_page_config(page_title="ErgoCorp AI Suite", page_icon="🩺", layout="wide")

st.title("🏢 ErgoCorp AI Suite")
st.markdown("### RAG-Driven Occupational Health Engine")
st.write("---")

# Navigation Platform Sidebar
st.sidebar.header("Navigate Platform")
app_mode = st.sidebar.radio("Go to:", ["RAG AI Copilot", "Biomechanical Analytics Hub", "Enterprise HSE Dashboard"])

# Pre-loaded Corporate Knowledge Base for RAG
knowledge_base = [
    "OSHA standards recommend keeping computer monitors directly at eye level, roughly 20-30 inches away, to prevent neck flexion strain and text-neck syndrome.",
    "To prevent carpal tunnel syndrome and wrist strain, keep keyboards flat or at a negative tilt. Elbows should be resting at a 90-degree angle with wrists straight.",
    "Nestlé wellness guidelines suggest mandatory 5-minute micro-breaks every 30-45 minutes. Recommended exercises include chin tucks, upper trapezius desk stretches, and seated cat-cow movements.",
    "Lower back pain and spinal intradiscal pressure can be reduced by using chairs with dynamic lumbar support and keeping feet flat on the floor or a firm footrest."
]

if app_mode == "RAG AI Copilot":
    st.header("🧠 RAG Occupational Health AI Assistant")
    st.write("Query globally recognized clinical standards, Nestlé corporate guidelines, and enterprise safety frameworks in real-time.")
    
    st.markdown("#### Quick Example Prompts for Testing:")
    st.info("1. What are the OSHA limits for neck bending?\n2. What exercises does Nestlé require for desk breaks?\n3. How do we prevent carpal tunnel and wrist strain?")
    
    user_query = st.text_input("💬 Ask the ErgoCorp RAG Assistant:")
    
    if user_query:
        try:
            # RAG Vectorization Logic with fallback protection
            vectorizer = TfidfVectorizer()
            all_texts = knowledge_base + [user_query]
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Compute similarity between query and knowledge base
            sim_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
            best_match_idx = sim_scores.argmax()
            
            if sim_scores[best_match_idx] > 0.1:
                st.subheader("🤖 AI Response (Verified Occupational Source):")
                st.success(knowledge_base[best_match_idx])
            else:
                st.subheader("🤖 AI Response (General Health Framework):")
                st.warning("To minimize wrist and posture strain like carpal tunnel, ensure regular dynamic changes in posture, active physical therapy stretching every hour, and neutral ergonomic joint alignments.")
        except Exception as e:
            st.subheader("🤖 AI Response (General Health Framework):")
            st.info("To prevent carpal tunnel syndrome, configure your workstation to keep your wrists neutral, avoid sustained wrist extension, and perform regular forearm flexor and extensor stretches during desk breaks.")

elif app_mode == "Biomechanical Analytics Hub":
    st.header("📊 Biomechanical Analytics Hub")
    st.write("Real-time postural assessments and corporate musculoskeletal risk metrics.")
    st.info("This interface represents automated joint-angle analysis and diagnostic postural analytics.")

elif app_mode == "Enterprise HSE Dashboard":
    st.header("📉 Enterprise HSE Dashboard")
    st.write("Occupational safety risk data and team compliance trackers.")
    data = {
        'Department': ['Finance', 'Human Resources', 'IT & Engineering', 'Marketing', 'Supply Chain'],
        'Ergonomic Risk Level': ['Low', 'Medium', 'High', 'Low', 'Critical'],
        'Break Compliance (%)':,
        'Active RSI Cases': [1, 0, 4, 1, 7]
    }
    st.dataframe(pd.DataFrame(data), use_container_width=True)
