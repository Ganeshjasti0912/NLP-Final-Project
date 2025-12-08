import os
import requests
from typing import List, Dict, Any

GOOGLE_PLACES_API_KEY = "AIzaSyAm0Qr0e19KwEHIHRSTUzzd-LETxKejPL4"
BASE_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"


def search_places(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    if not GOOGLE_PLACES_API_KEY:
        raise RuntimeError(
            "GOOGLE_PLACES_API_KEY is not set. "
            "Set it in your environment before running the backend."
        )

    params = {
        "query": query,
        "key": GOOGLE_PLACES_API_KEY,
    }

    response = requests.get(BASE_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    results: List[Dict[str, Any]] = []

    for item in data.get("results", [])[:limit]:
        results.append(
            {
                "name": item.get("name"),
                "address": item.get("formatted_address"),
                "rating": item.get("rating"),
                "types": item.get("types"),
            }
        )

    return results
