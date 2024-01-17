import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

st.set_page_config(page_title="My Llama2 Chatbot")

st.title("My Llama2 Chatbot")

# Use st.cache_resource for caching the model as it's a global resource
@st.cache_resource
def load_model(base_model_path, peft_model_path):
    # 기본 모델 로드
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    # PEFT 모델로 변환
    model = PeftModel.from_pretrained(base_model, peft_model_path)    
    return model

# Use st.cache_data for caching the tokenizer as it returns data
@st.cache_data
def load_tokenizer(base_model_path):
    # Tokenizer 로드
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    return tokenizer

base_model_path = "TinyPixel/CodeLlama-7B-Python-bf16-sharded"
peft_model_path = "hariqueen/myllama2"
model = load_model(base_model_path, peft_model_path)
tokenizer = load_tokenizer(base_model_path)

# 응답 생성 함수
def get_response(input_text):
    if not input_text:
        return "Please enter some text to get a response."
    try:
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        output_ids = model.generate(input_ids=input_ids, max_length=200, num_return_sequences=1)
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    except Exception as e:
        response = f"An error occurred: {e}"
    return response


# 기존의 prompt 및 gen 함수 정의 유지
prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: %s ### Response: "

def gen(x):
    q = prompt % (x,)
    gened = model.generate(
        **tokenizer(
            q,
            return_tensors='pt',
            return_token_type_ids=False
        ), 
        max_new_tokens=400,
        early_stopping=True,
        do_sample=False,
    )
    return tokenizer.decode(gened[0]).replace(q, "")

# 사용자 입력 받기
user_input = st.chat_input("질문을 입력하세요:")

# 사용자가 입력한 경우 응답 생성
if user_input:
    with st.spinner("Generating response..."):
        response = gen(user_input)

        # 챗봇의 응답 메시지 생성
        with st.chat_message("Llama2 Chatbot", avatar="🤖"):
            st.write(response)