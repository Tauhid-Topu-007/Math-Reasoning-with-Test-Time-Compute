import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from collections import Counter
import re
import time
from datetime import datetime
from dotenv import load_dotenv 

load_dotenv() 

# ================= Page Config =================
st.set_page_config(
    page_title="Math Reasoning with Test-Time Compute",
    page_icon="🧮",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ================= Custom CSS =================
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.3rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .answer-box {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        font-size: 1.4rem;
        font-weight: 600;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
    .example-btn {
        margin-bottom: 0.4rem;
    }
    .history-card {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ================= Title =================
st.markdown('<p class="main-title">🧮 Math Reasoning Demo</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Qwen2.5-1.5B + LoRA + Self-Consistency (Test-Time Compute)</p>', unsafe_allow_html=True)

# ================= Session State =================
if "history" not in st.session_state:
    st.session_state.history = []

# ================= Sidebar =================
with st.sidebar:
    st.header("⚙️ Settings")
    
    use_self_consistency = st.checkbox("Enable Self-Consistency", value=True)
    num_samples = st.slider("Number of samples", 4, 16, 8, 2)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
    max_tokens = st.slider("Max new tokens", 256, 1024, 512, 64)
    
    st.markdown("---")
    st.markdown("**Model Info**")
    st.caption("Qwen2.5-1.5B-Instruct + LoRA\nFine-tuned on GSM8K")
    
    st.markdown("---")
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ================= Load Model =================
@st.cache_resource
def load_model():
    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    lora_path = "./checkpoints/qwen-1.5b-gsm8k-lora"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # CPU তে চলার জন্য
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float32,
        device_map="cpu",               # GPU নাই বলে cpu
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    return model, tokenizer

with st.spinner("Loading model..."):
    model, tokenizer = load_model()

# ================= Helper Functions =================
def extract_answer(text):
    match = re.search(r"####\s*(-?\d+\.?\d*)", text)
    if match:
        return match.group(1).strip()
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1] if numbers else "Not found"

def generate_one(prompt, do_sample=True, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 0.0,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated

# ================= Example Questions =================
st.markdown("#### 📌 Example Questions")
examples = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    "Josh decides to try running a marathon. He runs 5 miles a day for 3 weeks, then increases to 8 miles a day for the next 2 weeks. How many total miles does he run?",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"
]

cols = st.columns(2)
for i, ex in enumerate(examples):
    with cols[i % 2]:
        if st.button(f"Example {i+1}", key=f"ex_{i}", use_container_width=True):
            st.session_state.user_question = ex

# ================= Input Area =================
st.markdown("#### ✍️ Enter Your Math Problem")
user_question = st.text_area(
    "Question",
    value=st.session_state.get("user_question", ""),
    height=130,
    placeholder="Type or select an example question...",
    label_visibility="collapsed"
)

col1, col2 = st.columns([2, 1])
with col1:
    generate_btn = st.button("🚀 Generate Solution", type="primary", use_container_width=True)
with col2:
    if st.button("Clear Input", use_container_width=True):
        st.session_state.user_question = ""
        st.rerun()

# ================= Generation Logic =================
if generate_btn and user_question.strip():
    prompt = f"""Solve the following math problem step by step. Put the final answer after the step-by-step solution.

Problem: {user_question.strip()}

Solution:"""

    start_time = time.time()
    
    if use_self_consistency:
        st.info(f"🔄 Running Self-Consistency with **{num_samples}** samples...")
        
        all_answers = []
        all_solutions = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(num_samples):
            status_text.text(f"Generating sample {i+1}/{num_samples}...")
            solution = generate_one(prompt, do_sample=True, temperature=temperature)
            ans = extract_answer(solution)
            all_solutions.append(solution)
            if ans != "Not found":
                all_answers.append(ans)
            progress_bar.progress((i + 1) / num_samples)
        
        status_text.empty()
        progress_bar.empty()
        
        if all_answers:
            majority_answer = Counter(all_answers).most_common(1)[0][0]
            vote_count = Counter(all_answers).most_common(1)[0][1]
        else:
            majority_answer = "Could not extract"
            vote_count = 0
        
        # Beautiful Answer Box
        st.markdown(f'<div class="answer-box">Final Answer: {majority_answer}</div>', unsafe_allow_html=True)
        st.caption(f"✅ Agreed by **{vote_count}/{len(all_answers)}** samples &nbsp;|&nbsp; ⏱️ {time.time()-start_time:.1f}s")
        
        with st.expander("📄 View all generated solutions"):
            for idx, sol in enumerate(all_solutions, 1):
                st.markdown(f"**Sample {idx}:**")
                st.code(sol, language="text")
                st.markdown("---")
        
        # Save to history
        st.session_state.history.insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            "question": user_question.strip()[:80] + ("..." if len(user_question) > 80 else ""),
            "answer": majority_answer,
            "method": f"Self-Consistency ({num_samples})"
        })
    
    else:
        with st.spinner("Generating solution..."):
            solution = generate_one(prompt, do_sample=False)
            final_ans = extract_answer(solution)
        
        st.markdown(f'<div class="answer-box">Final Answer: {final_ans}</div>', unsafe_allow_html=True)
        st.caption(f"⏱️ {time.time()-start_time:.1f}s")
        
        st.markdown("### Step-by-step Solution")
        st.code(solution, language="text")
        
        st.session_state.history.insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            "question": user_question.strip()[:80] + ("..." if len(user_question) > 80 else ""),
            "answer": final_ans,
            "method": "Greedy"
        })

elif generate_btn:
    st.warning("⚠️ Please enter a math problem first.")

# ================= History Section =================
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📜 Recent History")
    
    for item in st.session_state.history[:8]:  # শুধু সর্বশেষ ৮টা দেখাবে
        st.markdown(f"""
        <div class="history-card">
            <small style="color:#6b7280;">{item['time']} • {item['method']}</small><br>
            <b>Q:</b> {item['question']}<br>
            <b>A:</b> <span style="color:#059669; font-weight:600;">{item['answer']}</span>
        </div>
        """, unsafe_allow_html=True)

# ================= Footer =================
st.markdown("---")
st.caption("Built with Qwen2.5-1.5B + LoRA fine-tuned on GSM8K | Test-Time Compute (Self-Consistency)")