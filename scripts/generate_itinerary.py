import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER = "backend/models/mistral7b_lora_itinerary"

print(" Loading fine-tuned Mistral model")

tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.eos_token  

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE,
    quantization_config=bnb_config,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, ADAPTER)

model.config.pad_token_id = tok.eos_token_id
model.config.use_cache = True
model.config.do_sample = True
model.config.temperature = 0.7
model.config.top_p = 0.9

def generate_itinerary(query):
    """Generate a travel itinerary from a user query."""
    system = "You are a helpful travel planner that creates structured, day-by-day itineraries."
    prompt = f"<s>[SYSTEM] {system}\n[USER] {query}\n[ASSISTANT]"
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=500
        )

    return tok.decode(output[0], skip_special_tokens=True).split("[ASSISTANT]")[-1].strip()

if __name__ == "__main__":
    query = "Plan a 4-day luxury trip to Paris for a couple interested in art and fine dining."
    print("\nUser Query:", query)
    print("\nGenerated Itinerary:\n")
    print(generate_itinerary(query))
