import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Student Attendance Dashboard", layout="wide")
st.title("🎓 Student Attendance Analysis & Prediction System")

# -----------------------------
# LOAD MODEL & DATA
# -----------------------------
try:
    model = joblib.load("model.pkl")
    data = pd.read_csv("global_university_students_performance_habits_10000.csv")
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# -----------------------------
# CREATE STATUS
# -----------------------------
if "status" not in data.columns:
    data["status"] = data["class_attendance_percent"].apply(
        lambda x: 1 if x > 75 else 0
    )

# -----------------------------
# AGGREGATION
# -----------------------------
data["attendance_group"] = pd.cut(data["class_attendance_percent"],
                                  bins=[40, 60, 70, 80, 90, 100])

score_group = data.groupby("attendance_group")["final_exam_score"].mean()

data["study_group"] = pd.cut(data["study_hours_per_day"],
                            bins=[0, 2, 4, 6, 8, 12])

study_attendance = data.groupby("study_group")["class_attendance_percent"].mean()

stress_group = data.groupby("mental_stress_level")["study_hours_per_day"].mean()

data["screen_group"] = pd.cut(data["screen_time_hours"],
                             bins=[0, 2, 4, 6, 8, 12])

screen_attendance = data.groupby("screen_group")["class_attendance_percent"].mean()

# -----------------------------
# SIDEBAR INPUT
# -----------------------------
st.sidebar.header("📥 Enter Student Details")

study = st.sidebar.slider("Study Hours per Day", 0, 12, 5)
sleep = st.sidebar.slider("Sleep Hours", 3, 12, 7)
stress = st.sidebar.selectbox("Stress Level (1–10)", list(range(1, 11)), index=4)
screen = st.sidebar.slider("Screen Time Hours", 0, 12, 4)
prep = st.sidebar.slider("Exam Preparation Days", 0, 30, 10)

# -----------------------------
# PREDICTION
# -----------------------------
st.header("🤖 Prediction Result")

input_data = pd.DataFrame(
    [[study, sleep, stress, screen, prep]],
    columns=[
        "study_hours_per_day",
        "sleep_hours",
        "mental_stress_level",
        "screen_time_hours",
        "exam_preparation_days"
    ]
)

if st.sidebar.button("Predict"):

    if study < 2:
        st.warning("⚠️ Irregular Student (Very Low Study Hours)")
    else:
        result = model.predict(input_data)

        if result[0] == 1:
            st.success("✅ Regular Student")
        else:
            st.warning("⚠️ Irregular Student")

# -----------------------------
# DYNAMIC GRAPH
# -----------------------------
st.header("📊 Your Performance vs Average")

avg_values = data[[
    "study_hours_per_day",
    "sleep_hours",
    "mental_stress_level",
    "screen_time_hours",
    "exam_preparation_days"
]].mean()

compare_df = pd.DataFrame({
    "Feature": avg_values.index,
    "Your Value": input_data.iloc[0].values,
    "Average": avg_values.values
})

fig_dyn, ax_dyn = plt.subplots()
compare_df.set_index("Feature").plot(kind="bar", ax=ax_dyn)

ax_dyn.set_title("Your Input vs Average Student")

st.pyplot(fig_dyn)

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Basic Analysis",
    "📈 Relationship Graphs",
    "🧠 Advanced Insights"
])

# -----------------------------
# TAB 1
# -----------------------------
with tab1:
    st.subheader("Attendance Distribution")

    fig1, ax1 = plt.subplots()
    sns.histplot(data["class_attendance_percent"], bins=20, kde=True, ax=ax1)

    ax1.set_title("Attendance Distribution")
    st.pyplot(fig1)

    st.subheader("Regular vs Irregular")

    fig1b, ax1b = plt.subplots()
    counts = data["status"].value_counts()

    counts.plot(kind="bar", ax=ax1b)
    ax1b.set_xticklabels(["Irregular", "Regular"])

    for i, v in enumerate(counts):
        ax1b.text(i, v + 5, str(v), ha='center')

    st.pyplot(fig1b)

# -----------------------------
# TAB 2
# -----------------------------
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Study vs Attendance")

        fig2, ax2 = plt.subplots()
        study_attendance.plot(kind="line", marker='o', ax=ax2)

        st.pyplot(fig2)

    with col2:
        st.subheader("Attendance vs Score")

        fig3, ax3 = plt.subplots()
        score_group.plot(kind="bar", ax=ax3)

        for i, v in enumerate(score_group):
            ax3.text(i, v + 1, f"{v:.1f}", ha='center')

        st.pyplot(fig3)

# -----------------------------
# TAB 3
# -----------------------------
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Stress vs Study")

        fig4, ax4 = plt.subplots()
        stress_group.plot(kind="line", marker='o', ax=ax4)

        st.pyplot(fig4)

    with col2:
        st.subheader("Screen vs Attendance")

        fig5, ax5 = plt.subplots()
        screen_attendance.plot(kind="bar", ax=ax5)

        for i, v in enumerate(screen_attendance):
            ax5.text(i, v + 1, f"{v:.1f}", ha='center')

        st.pyplot(fig5)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("A1 Level Dashboard | Aggregation + Prediction + Clear Insights")