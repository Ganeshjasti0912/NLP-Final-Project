import os
import sys
import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
BACKEND_PATH = os.path.join(PROJECT_ROOT, "backend")

if BACKEND_PATH not in sys.path:
    sys.path.insert(0, BACKEND_PATH)

from google_places import search_places 

BASE = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER = "backend/models/mistral7b_lora_itinerary"

SUBSET_SIZE = 50

MAX_NEW_TOKENS = 2000

print("Loading fine-tuned model")
tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base_model, ADAPTER)

def build_places_block(places):
    if not places:
        return "No specific Google Places results were found for this destination."

    lines = []
    for i, p in enumerate(places, start=1):
        name = p.get("name", "Unknown place")
        addr = p.get("address", "")
        rating = p.get("rating")
        rating_str = f" (rating {rating})" if rating is not None else ""
        lines.append(f"{i}. {name} – {addr}{rating_str}")
    return "\n".join(lines)

def generate_itinerary(query: str, places) -> str:
    places_block = build_places_block(places)

    system = (
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

    prompt = f"<s>[SYSTEM] {system}\n[USER] {query}\n[ASSISTANT]"
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.9,
            do_sample=False,  
        )

    text = tok.decode(output[0], skip_special_tokens=True)
    return text.split("[ASSISTANT]")[-1].strip()

def is_delivered(plan: str) -> bool:
    words = plan.split()
    if len(words) < 30:
        return False

    for line in plan.splitlines():
        s = line.strip().lstrip("*-#> \t").lower()
        if re.search(r"\bday\s+\d+", s):
            return True

    return False


def satisfies_constraints(example, plan: str) -> bool:
    text_lower = plan.lower()

    dest = example.get("dest")
    if dest:
        dest_str = str(dest).lower()
        if dest_str not in text_lower:
            return False

    days = example.get("days")
    if days is not None and str(days) != "":
        if str(days) not in plan:
            return False

    budget = example.get("budget")
    if budget:
        budget_str = str(budget).lower()
        if budget_str not in ["", "none", "unknown"]:
            if budget_str not in text_lower:
                return False

    style = example.get("style")
    if style:
        style_str = str(style).lower()
        if style_str not in ["", "none", "unknown"]:
            if style_str not in text_lower:
                return False

    return True

# Loading Test data
print("Loading TravelPlanner test split...")
test_ds = load_dataset("osunlp/TravelPlanner", "test")["test"]

if SUBSET_SIZE is not None:
    test_ds = test_ds.select(range(SUBSET_SIZE))
    print(f"Evaluating on first {SUBSET_SIZE} examples")
else:
    print("Evaluating on all test examples")

num_examples = len(test_ds)
print(f"Total test examples: {num_examples}")

delivered = 0
passed = 0

for i, ex in enumerate(test_ds):
    query = ex["query"]
    dest = ex.get("dest")

    if dest:
        places_query = f"best attractions, restaurants, and hotels in {dest}"
    else:
        places_query = query

    places = search_places(places_query, limit=10)
    output = generate_itinerary(query, places)

    if is_delivered(output):
        delivered += 1
        if satisfies_constraints(ex, output):
            passed += 1

    if i < 3:
        print("\n EXAMPLE", i, "================")
        print("Query:", query)
        print("Places query:", places_query)
        print("Num places:", len(places))
        print("Generated plan:\n", output[:800], "...\n")

dr = delivered / num_examples
fpr = passed / num_examples

print("\nTask-Level Evaluation Results")
print(f"Delivery Rate (DR): {dr:.2%}")
print(f"Final Pass Rate (FPR): {fpr:.2%}")
print(f"Samples evaluated: {num_examples}")
print(f"Delivered plans: {delivered}")
print(f"Passed constraints: {passed}")
