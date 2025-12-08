from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

from google_places import search_places


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print(" Loading BERT preference extractor")

bert_path = "models/bert_pref_extractor"
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModelForSequenceClassification.from_pretrained(bert_path).to("cuda")

LABELS = ["destination", "days", "budget", "style"]


def extract_preferences(query: str) -> str:
    inputs = bert_tokenizer(
        query, return_tensors="pt", truncation=True, padding=True
    ).to("cuda")
    with torch.no_grad():
        logits = bert_model(**inputs).logits
        print("DEBUG logits =", logits)
        pred = torch.argmax(logits, dim=1).item()
        print("DEBUG predicted index:", pred)

    return LABELS[pred]

print(" Loading Mistral-7B LoRA itinerary model")

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_PATH = "models/mistral7b_lora_itinerary"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

mistral_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
mistral_tokenizer.pad_token = mistral_tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

mistral_model = PeftModel.from_pretrained(base_model, LORA_PATH)


def build_places_block(places):
    if not places:
        return "No specific Google Places results were found for this query."

    lines = []
    for i, p in enumerate(places, start=1):
        name = p.get("name", "Unknown place")
        addr = p.get("address", "")
        rating = p.get("rating")
        rating_str = f" (rating {rating})" if rating is not None else ""
        lines.append(f"{i}. {name} – {addr}{rating_str}")
    return "\n".join(lines)


def generate_itinerary(query: str, places):
    places_block = build_places_block(places)

    system_msg = (
        "You are a travel assistant. Create a structured, day-by-day itinerary "
        "based on the user request.\n\n"
        "Below is a list of real places returned by Google Places. "
        "Each entry has NAME and ADDRESS fields.\n\n"
        f"GOOGLE PLACES RESULTS:\n{places_block}\n\n"
        "When you recommend a place in the itinerary, you MUST always include both "
        "its NAME and its full ADDRESS exactly as written above. For example:\n"
        "  Breakfast: NAME (ADDRESS)\n"
        "  Lunch: NAME (ADDRESS)\n"
        "Do not invent new place names or addresses; only use places from the list.\n\n"
        "Write headings as plain text like 'Day 1', 'Day 2', etc. "
        "Do not use markdown formatting such as *, **, or bullet points in the headings. "
        "Label days as Day 1, Day 2, Day 3, etc., unless the user explicitly provides "
        "specific calendar dates. If the user gives dates, you may reuse those dates, "
        "but do not invent additional calendar dates that the user did not mention. "
        "Do not add extra notes, explanations, disclaimers, or assumptions at the end. "
        "Only return the itinerary itself."
    )

    prompt = f"<s>[SYSTEM] {system_msg}\n[USER] {query}\n[ASSISTANT]"

    inputs = mistral_tokenizer(prompt, return_tensors="pt").to(base_model.device)

    output = mistral_model.generate(
        **inputs,
        max_new_tokens=2000,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
    )

    text = mistral_tokenizer.decode(output[0], skip_special_tokens=True)
    return text.split("[ASSISTANT]")[-1].strip()



class QueryInput(BaseModel):
    text: str

@app.post("/generate")
def generate(data: QueryInput):
    query = data.text

    print(f"\n Incoming Query: {query}")

    # BERT preference
    preference = extract_preferences(query)
    print(f" Extracted BERT Preference → {preference}")

    # Google Places
    places = search_places(query, limit=10)
    print(" Google Places results:", len(places))

    # Mistral itinerary which is grounded on Places
    itinerary = generate_itinerary(query, places)
    print(f" Generated Itinerary Length → {len(itinerary)} chars")

    return {
        "preference": preference,
        "places": places,  
        "itinerary": itinerary,
    }


@app.get("/places")
def get_places(q: str, limit: int = 5):
    print(f" Google Places search: {q}")
    places = search_places(q, limit=limit)
    return {"query": q, "places": places}


@app.get("/")
def home():
    return {"status": "API is running with Google Places!"}