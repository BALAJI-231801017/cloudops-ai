import pandas as pd
import ollama

# Read health data
data = pd.read_csv("health_data.csv")

# Get the latest health record
latest = data.iloc[-1]

print("Latest Application Status")
print("-------------------------")

print("CPU Usage:", latest["cpu_usage"], "%")
print("Memory Usage:", latest["memory_usage"], "%")
print("Error Rate:", latest["error_rate"], "%")
print("Response Time:", latest["response_time"], "seconds")
print("Status:", latest["status"])


# Detect problems using Python
issues = []

if latest["cpu_usage"] > 80:
    issues.append("CPU usage is very high.")

if latest["memory_usage"] > 80:
    issues.append("Memory usage is very high.")

if latest["error_rate"] > 10:
    issues.append("Error rate is very high.")

if latest["response_time"] > 3:
    issues.append("Response time is very high.")


# Display detected problems
print("\nDetected Problems")
print("-----------------")

for issue in issues:
    print("-", issue)


# Ask AI to explain the detected problems
prompt = f"""
You are a CloudOps application health assistant.

The Python monitoring system detected these problems:

{issues}

Current application status:
CPU: {latest["cpu_usage"]}%
Memory: {latest["memory_usage"]}%
Error rate: {latest["error_rate"]}%
Response time: {latest["response_time"]} seconds
Status: {latest["status"]}

Explain:
1. What is happening?
2. What are the likely causes?
3. What should the user do?

Use simple, concise language.
Do not invent monitoring data.
Base your explanation on the detected problems and the provided metrics.
"""

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nAI Diagnosis")
print("------------")
print(response["message"]["content"])