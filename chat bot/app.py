from flask import Flask, render_template, request, jsonify
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

template = """
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer: 
"""

model = OllamaLLM(model="llama3.2")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Store conversation context
conversation_context = ""


@app.route("/")
def home():
    return render_template("chat.html")


@app.route("/get_response", methods=["POST"])
def get_response():
    global conversation_context
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Process with LangChain
    result = chain.invoke({"context": conversation_context, "question": user_input})

    # Update context
    conversation_context += f'\nuser: {user_input}\nAi: {result}'

    return jsonify({"response": result})


if __name__ == "__main__":
    app.run(debug=True)
