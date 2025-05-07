# # Install OpenAI SDK
# # pip install openai

# from openai import OpenAI

# # Initialize client with DeepSeek API key and base URL
# client = OpenAI(
#     api_key="sk-6389b5217c4a4a3699716af25942516a", 
#     base_url="https://api.deepseek.com"
# )

# # Make API call
# response = client.chat.completions.create(
#     model="deepseek-chat",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Write a short Python function to calculate the factorial of a number."}
#     ],
#     stream=False
# )

# # Print response
# print(response.choices[0].message.content)



import os
from openai import OpenAI
from openai import APIError, APIStatusError

# Load API key from environment variable for security (recommended)
# Alternatively, replace with your API key directly (less secure)
API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-6389b5217c4a4a3699716af25942516a")  # Replace with your actual API key if not using env

# Initialize client with DeepSeek API key and base URL
client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com"
)

try:
    # Make API call
    response = client.chat.completions.create(
        model="deepseek-chat",  # Use 'deepseek-reasoner' for R1 or 'deepseek-chat' for V3
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a short Python function to calculate the factorial of a number."}
        ],
        stream=False
    )

    # Print response
    print(response.choices[0].message.content)

except APIStatusError as e:
    # Handle specific API status errors
    if e.status_code == 402:
        print("Error: Insufficient balance in your DeepSeek account. Please top up your account at https://platform.deepseek.com/.")
    else:
        print(f"API Status Error: {e.status_code} - {e.message}")
except APIError as e:
    # Handle other API-related errors
    print(f"API Error: {e.message}")
except Exception as e:
    # Catch any other unexpected errors
    print(f"Unexpected Error: {str(e)}")