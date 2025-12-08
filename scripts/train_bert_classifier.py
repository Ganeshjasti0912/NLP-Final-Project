from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from collections import Counter
import os
import random

# Labelling
LABELS = ["destination", "days", "budget", "style"]
label_to_id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for i, l in enumerate(LABELS)}

def generate_synthetic_data():
    budget_queries = [
        "Plan a cheap weekend trip under $300",
        "Looking for an affordable vacation on a tight budget",
        "What's the most economical way to visit Paris?",
        "I need a low-cost 3-day trip to New York",
        "Help me plan an inexpensive getaway",
        "Budget-friendly destinations for $500 or less",
        "Affordable travel options for a family of 4",
        "Cheap flight and hotel packages to Europe",
        "How can I travel to Japan on a student budget?",
        "Looking for cost-effective vacation ideas",
        "What can I do in Rome for under $1000?",
        "Frugal travel tips for Southeast Asia",
        "Low-budget honeymoon destinations",
        "Economical weekend trips from Chicago",
        "Best value destinations for budget travelers",
        "Cheap all-inclusive resorts in Mexico",
        "How to travel Italy without spending much money",
        "Budget hostels and cheap eats in Barcelona",
        "Planning a trip with maximum budget of $2000",
        "Affordable beach vacations for couples",
        "What are some cheap tropical destinations?",
        "Looking for inexpensive European cities to visit",
        "Budget travel to Australia under $3000",
        "Cheap ways to explore South America",
        "Need affordable accommodation in London",
        "Low-cost adventure trips for students",
        "Best budget airlines for domestic travel",
        "Economical road trip ideas in California",
        "Cheap vacation spots in the Caribbean",
        "How to save money while traveling Europe",
    ]
    
    style_queries = [
        "Plan a romantic honeymoon with spa and fine dining",
        "Looking for a luxury beach resort with premium amenities",
        "I want a relaxing vacation by the ocean",
        "Find me an adventure-packed trip with hiking and zip-lining",
        "Family-friendly destination with kids activities",
        "Romantic anniversary getaway for couples",
        "Looking for cultural experiences and museums",
        "I love art galleries and historic architecture",
        "Beach vacation with water sports and snorkeling",
        "Foodie destination with amazing restaurants",
        "Nightlife and party destinations for young adults",
        "Peaceful retreat for meditation and wellness",
        "Adventure sports like skydiving and bungee jumping",
        "Wine tasting tours in scenic vineyards",
        "Shopping destinations with designer boutiques",
        "Luxury cruise with gourmet dining",
        "Historic landmarks and ancient ruins",
        "Nature photography and wildlife safaris",
        "Culinary tour of street food markets",
        "Spa resort with massages and yoga classes",
        "Rock climbing and mountain trekking",
        "Romantic gondola rides and candlelit dinners",
        "Jazz clubs and live music venues",
        "Surfing and beach volleyball",
        "Backpacking through remote villages",
        "Five-star hotels with ocean views",
        "Cooking classes and food markets",
        "Theater shows and Broadway performances",
        "Scuba diving in coral reefs",
        "Upscale dining and Michelin-starred restaurants",
        "Cultural festivals and local celebrations",
        "Boutique hotels with unique character",
        "Horseback riding and ranch experiences",
        "Art museums and contemporary galleries",
        "Elegant afternoon tea and champagne",
    ]
     
    days_queries = [
        "How many days do I need to see Rome?",
        "What's a good 3-day itinerary for Tokyo?",
        "I have 5 nights available for a trip",
        "Planning a 4-day weekend getaway",
        "Is one week enough for Thailand?",
        "2-day city break ideas",
        "How to spend 7 days in Europe",
        "I only have 3 days off work",
        "10-day travel itinerary suggestions",
        "Quick 2-night escape from New York",
        "Is 5 days sufficient for Greece?",
        "6-day travel plan needed",
        "I can take a 4-day vacation in May",
        "How long should I spend in Iceland?",
        "3-night getaway recommendations",
        "Planning an 8-day road trip",
        "Do I need more than 5 days in Paris?",
        "Short 2-day trip from Boston",
        "How many days for a complete Egypt tour?",
        "I have exactly one week for travel",
        "4-night stay suggestions",
        "Is 3 days enough for New Orleans?",
        "Planning a 9-day vacation",
        "2-day itinerary for San Francisco",
        "How long to explore Vietnam?",
        "5-day trip timing advice",
        "I can only go for 3 nights",
        "7-day Europe tour possible?",
        "Quick 48-hour city visit",
        "Optimal days needed for Australia",
    ]
    
    destination_queries = [
        "Tell me about traveling to Iceland",
        "What should I visit in Barcelona?",
        "Help me plan a trip to Tokyo",
        "Is Morocco a good destination?",
        "What's there to do in Prague?",
        "Tell me about New Zealand travel",
        "Should I visit Portugal or Spain?",
        "What makes Bali special?",
        "Exploring destinations in Peru",
        "Is Norway worth visiting?",
        "Tell me about Croatia",
        "What can I see in Vietnam?",
        "Thinking about visiting Ireland",
        "Is Dubai an interesting destination?",
        "What's special about Costa Rica?",
        "Tell me about Scottish Highlands",
        "Is Argentina a good choice?",
        "What to expect in South Africa?",
        "Considering a trip to Malaysia",
        "Tell me about Greek islands",
        "Is Colombia safe for tourists?",
        "What's unique about Jordan?",
        "Should I visit Budapest?",
        "Tell me about Austrian Alps",
        "Is Philippines a good destination?",
        "What makes Istanbul interesting?",
        "Considering Poland for travel",
        "Tell me about Canadian Rockies",
        "Is Chile worth the trip?",
        "What can I discover in Morocco?",
    ]
    
    data = []
    
    for q in budget_queries:
        data.append({"query": q, "label": label_to_id["budget"]})
    
    for q in style_queries:
        data.append({"query": q, "label": label_to_id["style"]})
    
    for q in days_queries:
        data.append({"query": q, "label": label_to_id["days"]})
    
    for q in destination_queries:
        data.append({"query": q, "label": label_to_id["destination"]})
    
    random.shuffle(data)
    
    return data


def describe_distribution(name, labels):
    counts = Counter(labels)
    total = sum(counts.values())
    print(f"\n{'='*60}")
    print(f"{name} LABEL DISTRIBUTION")
    print(f"{'='*60}")
    print(f"{'Label':<15} {'ID':<5} {'Count':<10} {'Percentage':<10}")
    print(f"{'-'*60}")
    for idx, lab_name in enumerate(LABELS):
        c = counts.get(idx, 0)
        pct = 0.0 if total == 0 else c / total * 100
        print(f"{lab_name:<15} {idx:<5} {c:<10} {pct:>6.1f}%")
    print(f"{'-'*60}")
    print(f"{'TOTAL':<15} {'':<5} {total:<10} 100.0%")
    print(f"{'='*60}\n")


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print(" Generating synthetic training data")
    all_data = generate_synthetic_data()
    
    print(f" Generated {len(all_data)} training examples")
    
    train_data, val_data = train_test_split(
        all_data, test_size=0.2, random_state=42, stratify=[d["label"] for d in all_data]
    )
    
    print(f" Train set: {len(train_data)} examples")
    print(f" Val set: {len(val_data)} examples")
    
    train_labels = [d["label"] for d in train_data]
    val_labels = [d["label"] for d in val_data]
    
    describe_distribution("TRAIN", train_labels)
    describe_distribution("VALIDATION", val_labels)
    
    print("\n Sample training queries (5 per category):")
    print("="*80)
    for label_id, label_name in enumerate(LABELS):
        print(f"\n{label_name.upper()}:")
        examples = [d for d in train_data if d["label"] == label_id][:5]
        for ex in examples:
            print(f"  • {ex['query']}")
    print()
    
    train_labels_array = np.array(train_labels)
    unique_labels = np.unique(train_labels_array)
    
    class_weights_dict = compute_class_weight(
        class_weight='balanced',
        classes=unique_labels,
        y=train_labels_array
    )
    
    class_weights = np.ones(len(LABELS))
    for i, label in enumerate(unique_labels):
        class_weights[label] = class_weights_dict[i]
    
    print("  Class weights (for balancing):")
    for i, weight in enumerate(class_weights):
        print(f"  {LABELS[i]}: {weight:.3f}")
    print()

    print(" Tokenizing")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def tokenize_data(data):
        queries = [d["query"] for d in data]
        labels = [d["label"] for d in data]
        
        encodings = tokenizer(
            queries,
            truncation=True,
            padding="max_length",
            max_length=64,  
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(labels)
        }
    
    train_encodings = tokenize_data(train_data)
    val_encodings = tokenize_data(val_data)
    
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        
        def __len__(self):
            return len(self.encodings["labels"])
        
        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.encodings["labels"][idx]
            }
    
    train_dataset = SimpleDataset(train_encodings)
    val_dataset = SimpleDataset(val_encodings)

    print(" Loading BERT base model for classification")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label_to_id,
    )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            
            weight_tensor = torch.tensor(
                class_weights, dtype=torch.float32
            ).to(logits.device)
            
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight_tensor)
            loss = loss_fct(logits, labels)
            
            return (loss, outputs) if return_outputs else loss
    
    output_dir = "backend/models/bert_pref_extractor"
    
    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=10,  
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
    )
    
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print(" Starting training...\n")
    trainer.train()
    
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n" + "="*60)
    print(" Training complete!")
    print(f" Model saved to: {output_dir}")
    print("="*60)
    
    print("\n SANITY CHECK - Testing on diverse queries:\n")
    
    test_queries = [
        # Budget queries
        ("Plan a cheap 2-day weekend trip under $300 near New York.", "budget"),
        ("Looking for an affordable vacation on a tight budget.", "budget"),
        ("What can I do in Paris for less than $500?", "budget"),
        
        # Style queries
        ("Plan a 4-day luxury trip to Paris for a couple interested in art and fine dining.", "style"),
        ("Plan a relaxing one-week beach vacation in Bali.", "style"),
        ("I want a romantic honeymoon with spa and fine dining.", "style"),
        ("Looking for a family-friendly trip with kids activities.", "style"),
        ("Adventure vacation with hiking and rock climbing.", "style"),
         
        ("How many days do I need to see Rome?", "days"),
        ("What's a good 3-day itinerary for Tokyo?", "days"),
        ("I have 5 nights available for a trip.", "days"),
        ("Help me plan a trip to Tokyo next spring.", "destination"),
        ("What should I visit in Barcelona?", "destination"),
        ("Tell me about traveling to Iceland.", "destination"),
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    correct = 0
    total = len(test_queries)
    
    print(f"{'Query':<70} {'Expected':<12} {'Predicted':<12} {'Conf%':<8} {'Result'}")
    print("="*115)
    
    for query, expected in test_queries:
        inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        
        pred_id = int(torch.argmax(logits, dim=-1).item())
        predicted = LABELS[pred_id]
        probs = torch.softmax(logits, dim=-1)[0]
        confidence = probs[pred_id].item() * 100
        
        is_correct = predicted == expected
        if is_correct:
            correct += 1
        
        result = "correct" if is_correct else "wrong"
        query_short = query[:67] + "..." if len(query) > 70 else query
        
        print(f"{query_short:<70} {expected:<12} {predicted:<12} {confidence:>5.1f}%   {result}")
    
    print("="*115)
    print(f"\nAccuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print("\n" + "="*60)
    print(" Sanity check complete!")
    print("="*60)


if __name__ == "__main__":
    main()