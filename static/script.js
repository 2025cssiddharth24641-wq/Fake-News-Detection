const newsInput = document.getElementById("newsInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");
const charCount = document.getElementById("charCount");
const loader = document.getElementById("loader");
const messageBox = document.getElementById("messageBox");
const resultCard = document.getElementById("resultCard");
const predictionBadge = document.getElementById("predictionBadge");
const confidenceText = document.getElementById("confidenceText");
const confidenceBar = document.getElementById("confidenceBar");
const guidanceText = document.getElementById("guidanceText");
const historyList = document.getElementById("historyList");

const STORAGE_KEY = "fake-news-history-v2";
const MAX_CHARS = 50000;
const MAX_HISTORY = 10;

function getHistory() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        return raw ? JSON.parse(raw) : [];
    } catch {
        return [];
    }
}

function saveHistory(items) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items.slice(0, MAX_HISTORY)));
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function updateCharCount() {
    const len = newsInput.value.length;
    charCount.textContent = `${len} / ${MAX_CHARS}`;
    
    if (len > MAX_CHARS) {
        newsInput.value = newsInput.value.slice(0, MAX_CHARS);
    }
}

function showMessage(text) {
    messageBox.textContent = text;
    messageBox.classList.remove("hidden");
}

function hideMessage() {
    messageBox.classList.add("hidden");
}

function setLoading(isLoading) {
    loader.classList.toggle("hidden", !isLoading);
    analyzeBtn.disabled = isLoading;
}

function renderResult(data) {
    const percentage = Math.round(data.confidence * 100);
    const predictionType = data.prediction === "REAL"
        ? "real"
        : (data.prediction === "FAKE" ? "fake" : "verify");
    
    predictionBadge.textContent = data.prediction;
    predictionBadge.className = "badge badge-" + predictionType;
    
    confidenceText.textContent = `${percentage}% Confidence`;
    confidenceBar.style.width = `${percentage}%`;
    
    guidanceText.textContent = data.guidance;
    resultCard.classList.remove("hidden");
}

function addHistoryEntry(text, result) {
    const items = getHistory();
    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const dateStr = now.toLocaleDateString();
    
    items.unshift({
        text: text.slice(0, 200),
        prediction: result.prediction,
        confidence: Math.round(result.confidence * 100),
        time: timeStr,
        date: dateStr
    });
    
    saveHistory(items);
    renderHistory();
}

function formatHistoryTimestamp(item) {
    const today = new Date().toLocaleDateString();
    return item.date === today ? item.time : item.date;
}

function renderHistory() {
    const items = getHistory();
    
    if (items.length === 0) {
        historyList.innerHTML = '<p style="color: var(--text-muted); text-align: center; padding: 20px; font-weight: 500;">No analyses yet. Start by pasting a news article above.</p>';
        return;
    }

    historyList.innerHTML = items
        .map(item => `
            <article class="history-item">
                <p class="history-text">${escapeHtml(item.text)}</p>
                <div class="history-meta">
                    <span class="history-badge history-badge-${historyBadgeType(item.prediction)}">
                        ${item.prediction}
                    </span>
                    <span>${item.confidence}%</span>
                    <span>${formatHistoryTimestamp(item)}</span>
                </div>
            </article>
        `)
        .join("");
}

function historyBadgeType(prediction) {
    if (prediction === "REAL") return "real";
    if (prediction === "FAKE") return "fake";
    return "verify";
}

async function analyzeText() {
    const text = newsInput.value.trim();
    hideMessage();

    if (!text) {
        showMessage("Please enter news text before analyzing.");
        return;
    }

    if (text.length < 30) {
        showMessage("Input too short. Paste a complete news article or statement for accurate analysis.");
        return;
    }

    const wordCount = text.split(/\s+/).filter(w => w.length > 0).length;
    if (wordCount < 5) {
        showMessage("Input must contain at least 5 words for reliable prediction.");
        return;
    }

    setLoading(true);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });

        const payload = await response.json();

        if (!response.ok) {
            throw new Error(payload.detail || "Failed to analyze. Please try again.");
        }

        renderResult(payload);
        addHistoryEntry(text, payload);
    } catch (error) {
        showMessage(error.message);
    } finally {
        setLoading(false);
    }
}

function handleClear() {
    newsInput.value = "";
    newsInput.focus();
    updateCharCount();
    hideMessage();
    resultCard.classList.add("hidden");
    confidenceBar.style.width = "0%";
}

function handleClearHistory() {
    if (confirm("Clear all analysis history? This cannot be undone.")) {
        localStorage.removeItem(STORAGE_KEY);
        renderHistory();
    }
}

newsInput.addEventListener("input", updateCharCount);
newsInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        analyzeText();
    }
});

analyzeBtn.addEventListener("click", analyzeText);
clearBtn.addEventListener("click", handleClear);
clearHistoryBtn.addEventListener("click", handleClearHistory);

updateCharCount();
renderHistory();
