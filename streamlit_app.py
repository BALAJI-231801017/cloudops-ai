import streamlit as st
import pandas as pd
import ollama

# --------------------------------------------------
# Page setup
# --------------------------------------------------

st.set_page_config(
    page_title="CloudOps AI",
    page_icon="☁️",
    layout="wide"
)

st.title("☁️ CloudOps AI")
st.subheader("AI-Powered Cloud Application Health Assistant")


# --------------------------------------------------
# Load health data
# --------------------------------------------------

data = pd.read_csv("health_data.csv")

# Latest health record
latest = data.iloc[-1]

# Recent monitoring history
recent_data = data.tail(6).to_string(index=False)


# --------------------------------------------------
# Detect problems
# --------------------------------------------------

issues = []

if latest["cpu_usage"] > 80:
    issues.append("CPU usage is very high.")

if latest["memory_usage"] > 80:
    issues.append("Memory usage is very high.")

if latest["error_rate"] > 10:
    issues.append("Error rate is very high.")

if latest["response_time"] > 3:
    issues.append("Response time is very high.")


# --------------------------------------------------
# Calculate health score
# --------------------------------------------------

health_score = 100

if latest["cpu_usage"] > 80:
    health_score -= 25

if latest["memory_usage"] > 80:
    health_score -= 25

if latest["error_rate"] > 10:
    health_score -= 25

if latest["response_time"] > 3:
    health_score -= 25


# --------------------------------------------------
# Application Health
# --------------------------------------------------

st.header("Application Health")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "CPU Usage",
    f'{latest["cpu_usage"]}%'
)

col2.metric(
    "Memory Usage",
    f'{latest["memory_usage"]}%'
)

col3.metric(
    "Error Rate",
    f'{latest["error_rate"]}%'
)

col4.metric(
    "Response Time",
    f'{latest["response_time"]} sec'
)

col5.metric(
    "Health Score",
    f"{health_score}/100"
)


# --------------------------------------------------
# Application Status
# --------------------------------------------------

if latest["status"] == "Critical":
    st.error(f'Application Status: {latest["status"]}')

elif latest["status"] == "Warning":
    st.warning(f'Application Status: {latest["status"]}')

else:
    st.success(f'Application Status: {latest["status"]}')


# --------------------------------------------------
# Detected Problems
# --------------------------------------------------

st.header("Detected Problems")

if issues:

    for issue in issues:
        st.warning(issue)

else:

    st.success("No major problems detected.")


# --------------------------------------------------
# Health Trends
# --------------------------------------------------

st.header("📈 Health Trends")

st.line_chart(
    data.set_index("timestamp")[
        [
            "cpu_usage",
            "memory_usage",
            "error_rate"
        ]
    ]
)


# --------------------------------------------------
# AI Diagnosis
# --------------------------------------------------

st.header("🤖 AI Diagnosis")

if st.button("Analyze Application"):

    prompt = f"""
You are a CloudOps application health assistant.

The Python monitoring system detected these problems:

{issues}

Current application health:

CPU: {latest["cpu_usage"]}%
Memory: {latest["memory_usage"]}%
Error rate: {latest["error_rate"]}%
Response time: {latest["response_time"]} seconds
Status: {latest["status"]}

Recent monitoring history:

{recent_data}

Explain:

1. What is happening?
2. What are the likely causes?
3. What should the user do?

Use simple and concise language.

Do not invent monitoring data.

Base your explanation on the provided metrics,
detected problems, and monitoring history.
"""

    with st.spinner("AI is analyzing the application..."):

        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

    st.write(response["message"]["content"])


# --------------------------------------------------
# CloudOps AI Assistant
# --------------------------------------------------

st.header("💬 CloudOps AI Assistant")

question = st.text_input(
    "Ask about your application",
    placeholder="Example: Why is my application slow?"
)

if st.button("Ask AI"):

    if not question:

        st.warning("Please enter a question first.")

    else:

        prompt = f"""
You are a CloudOps application health assistant.

User question:
{question}

Current application health:

CPU: {latest["cpu_usage"]}%
Memory: {latest["memory_usage"]}%
Error rate: {latest["error_rate"]}%
Response time: {latest["response_time"]} seconds
Status: {latest["status"]}

Detected problems:

{issues}

Recent monitoring history:

{recent_data}

Answer the user's question using the provided
monitoring data.

Rules:

- Keep the answer simple and concise.
- Explain the likely problem clearly.
- Give practical recommendations.
- Do not invent monitoring data.
- Use the historical data when relevant.
"""

        with st.spinner("AI is analyzing..."):

            response = ollama.chat(
                model="llama3.2:3b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

        st.write(response["message"]["content"])