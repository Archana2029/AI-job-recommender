import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎓 AI-Based Job Recommendation System")
st.write("Select your degree and enter your skills to get job recommendations")

degree = st.selectbox(
    "Select Your Degree",
    ["Select Your Degree","BBA", "BCA", "BA", "BCom"]
)

# Degree based datasets
if degree == "BBA":
    data = {
        "Job Title": [
            "Marketing Executive",
            "HR Executive",
            "Financial Analyst",
            "Business Development Executive"
        ],
        "Skills": [
            "marketing sales communication branding",
            "recruitment hr management leadership",
            "finance accounting excel analysis",
            "business strategy negotiation communication"
        ]
    }

elif degree == "BCA":
    data = {
        "Job Title": [
            "Web Developer",
            "Software Developer",
            "System Administrator",
            "Data Analyst"
        ],
        "Skills": [
            "html css javascript react",
            "java python c++ coding",
            "linux networking troubleshooting",
            "python sql excel data analysis"
        ]
    }

elif degree == "BA":
    data = {
        "Job Title": [
            "Content Writer",
            "Social Media Manager",
            "Public Relations Officer",
            "Journalist"
        ],
        "Skills": [
            "writing communication creativity research",
            "social media marketing communication",
            "public speaking communication management",
            "reporting writing research"
        ]
    }

else:  # BCom
    data = {
        "Job Title": [
            "Accountant",
            "Banking Associate",
            "Tax Consultant",
            "Auditor"
        ],
        "Skills": [
            "accounting tally gst finance",
            "banking finance customer service",
            "taxation gst filing accounting",
            "auditing accounting compliance"
        ]
    }

df = pd.DataFrame(data)

user_input = st.text_input("Enter your skills (space separated):")

if st.button("Get Recommendations"):

    if user_input:

        skills_data = df["Skills"].tolist()
        skills_data.append(user_input)

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(skills_data)

        similarity = cosine_similarity(vectors[-1], vectors[:-1])
        scores = similarity.flatten()

        df["Similarity"] = scores
        recommended_jobs = df.sort_values(by="Similarity", ascending=False)

        st.subheader("🎯 Top Recommended Jobs For You:")
        st.write(recommended_jobs[["Job Title"]].head(3))

    else:
        st.warning("Please enter your skills")