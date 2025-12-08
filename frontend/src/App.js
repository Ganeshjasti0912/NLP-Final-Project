import React, { useState } from "react";
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const API_URL ="https://ondemand.orc.gmu.edu/rnode/gpu018.orc.gmu.edu/43663/proxy/8000";

  async function generate() {
    if (!query.trim()) return;

    setLoading(true);
    setResponse(null);
    setError("");

    try {
      const res = await fetch(`${API_URL}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: query }),
      });

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`);
      }

      const data = await res.json();
      setResponse(data);
    } catch (e) {
      console.error(e);
      setError("Something went wrong while generating your itinerary.");
    } finally {
      setLoading(false);
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      generate();
    }
  };

  const prettyPreference =
    response?.preference &&
    response.preference.charAt(0).toUpperCase() + response.preference.slice(1);

  return (
    <div className="app-root">
      <div className="app-shell">
        <header className="app-header">
          <div className="logo-circle">✈️</div>
          <div>
            <h1 className="app-title">Travel Itinerary Planner</h1>
            <p className="app-subtitle">
              One prompt away from your next adventure 
            </p>
          </div>
        </header>

        <main>
          <section className="card input-card">
            <div className="card-header">
              <h2>Describe your dream trip</h2>
              <p className="card-helper">
                Mention things like <span>destination</span>, <span>days</span>,{" "}
                <span>budget</span>, or <span>style</span> (luxury, romantic, foodie,
                adventure…)
              </p>
            </div>

            <textarea
              rows="4"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Example: Plan a 4-day luxury trip to Paris for a couple interested in art and fine dining."
              className="query-textarea"
            />

            <div className="input-footer">
              <button
                onClick={generate}
                disabled={loading || !query.trim()}
                className="primary-button"
              >
                {loading ? (
                  <>
                    <span className="spinner" /> Generating…
                  </>
                ) : (
                  "Generate Itinerary"
                )}
              </button>
            </div>

            {error && <p className="error-text">{error}</p>}
          </section>

          <section className="suggestions">
            <p>Try one of these:</p>
            <div className="suggestion-chips">
              {[
                "Plan a 3-day budget trip to New York for two students who love street food.",
                "Create a 3-day relaxing beach vacation in Bali with spa and sunsets.",
                "Plan a 5-day family trip to Tokyo with kid-friendly activities.",
              ].map((s, idx) => (
                <button
                  key={idx}
                  type="button"
                  className="chip"
                  onClick={() => setQuery(s)}
                >
                  {s}
                </button>
              ))}
            </div>
          </section>

          {response && (
            <section className="card output-card">
              <div className="output-header">
                <h2>Your travel plan is ready. Time to explore</h2>
                {prettyPreference && (
                  <span
                    className={`pref-badge pref-${response.preference.toLowerCase()}`}
                  >
                    {prettyPreference} focus
                  </span>
                )}
              </div>

              <div className="output-grid">
                <div className="output-section">
                  <h3>Extracted Preference</h3>
                  <p className="preference-text">
                    {prettyPreference || "— not detected —"}
                  </p>
                </div>

                <div className="output-section itinerary-section">
                  <h3>Generated Itinerary</h3>
                  <pre className="itinerary-pre">{response.itinerary}</pre>
                </div>
              </div>
            </section>
          )}
        </main>

        <footer className="app-footer">
          <span>Built with React &amp; custom BERT + Google API + Mistral </span>
        </footer>
      </div>
    </div>
  );
}

export default App;