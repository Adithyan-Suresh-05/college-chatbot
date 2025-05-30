import tkinter as tk
from tkinter import scrolledtext
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

faq = {
    "what courses are offered": "We offer B.Tech, M.Tech, MBA, and MCA programs across various disciplines.",
    "what is the college timing": "Our college operates from 9 AM to 4 PM, Monday to Friday.",
    "where is the college located": "The college is located in Trivandrum, Kerala.",
    "how can i contact the college": "You can contact the college office at +91-9876543210 or email info@college.edu.",
    "tell me about the admission process": "Admission starts in June every year. Visit the admission portal on our website to apply.",
    "hello": "Hello! I'm here to help you with college-related queries.",
    "hi": "Hi there! How can I help you today?",
    "good morning": "Good morning! How can I assist you today?",
    "good evening": "Good evening! How can I assist you?",
    "i want to enroll in the college": "Great! You can apply through our online admission portal starting in June."
}

faq_questions = list(faq.keys())
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)

def get_smart_response(user_input):
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    similarities = util.cos_sim(user_embedding, faq_embeddings)[0]
    best_match_idx = int(similarities.argmax())
    confidence = float(similarities[best_match_idx])
    if confidence > 0.5:
        matched_question = faq_questions[best_match_idx]
        return faq[matched_question]
    else:
        return "I'm not sure I understood that. Could you rephrase?"

def send_message(event=None):
    user_msg = user_input.get().strip()
    if not user_msg:
        return
    chat_area.config(state=tk.NORMAL)
    chat_area.insert(tk.END, "You: " + user_msg + "\n", "user")
    response = get_smart_response(user_msg)
    chat_area.insert(tk.END, "CollegeBot: " + response + "\n\n", "bot")
    chat_area.config(state=tk.DISABLED)
    chat_area.yview(tk.END)
    user_input.delete(0, tk.END)
    user_input.focus()

# Main window
root = tk.Tk()
root.title("CollegeBot - Smart Assistant")
root.geometry("600x500")
root.configure(bg="#f0f4f7")


root.rowconfigure(0, weight=1)  
root.rowconfigure(1, weight=0)  

# Chat area frame
chat_frame = tk.Frame(root)
chat_frame.grid(row=0, column=0, sticky="nsew")

chat_area = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, font=("Segoe UI", 11), bg="white", fg="#333")
chat_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
chat_area.tag_config("user", foreground="blue")
chat_area.tag_config("bot", foreground="green")
chat_area.config(state=tk.NORMAL)
chat_area.insert(tk.END, "🎓 CollegeBot: Welcome! Ask me anything about the college.\n\n", "bot")
chat_area.config(state=tk.DISABLED)

# Input area frame
input_frame = tk.Frame(root, bg="#f0f4f7")
input_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
input_frame.columnconfigure(0, weight=1)

user_input = tk.Entry(input_frame, font=("Segoe UI", 12))
user_input.grid(row=0, column=0, sticky="ew", padx=(0, 10))
user_input.focus()

send_button = tk.Button(input_frame, text="Send", command=send_message, font=("Segoe UI", 11), bg="#4CAF50", fg="white")
send_button.grid(row=0, column=1)


root.bind('<Return>', send_message)


root.mainloop()
