from flask import Flask, request, jsonify
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

app = Flask(_name_)

# Model and tokenizer setup with LangChain
def create_model_and_tokenizer():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_safetensors=True,
        trust_remote_code=True,
        device_map="auto",
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    generation_config = GenerationConfig.from_pretrained(model_name)
    generation_config.max_new_tokens = 512
    generation_config.temperature = 0.0001
    generation_config.top_p = 0.95
    generation_config.do_sample = True
    generation_config.repetition_penalty = 1.15

    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        generation_config=generation_config,
    )
    llm = HuggingFacePipeline(pipeline=text_pipeline)

    return llm

llm = create_model_and_tokenizer()

# Define the API endpoint for asking questions
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        question = data.get('question')
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        result = llm(question)
        answer = result
        print(answer)

        return jsonify({'answer': answer}), 200
    except Exception as e:
        print("Unexpected server error:", e)
        traceback.print_exc()
        return jsonify({'error': 'Server encountered an unexpected error'}), 500

if _name_ == '_main_':
    app.run(host='0.0.0.0', port=5000, workers=4)