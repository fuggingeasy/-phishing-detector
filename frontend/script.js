const emailInput = document.getElementById("email-input");
const charCount = document.getElementById("char-count");
const analyzeBtn = document.getElementById("analyze-btn");
const resultPanel = document.getElementById("result-panel");
const resultEmpty = document.getElementById("result-empty");
const resultContent = document.getElementById("result-content");
const resultError = document.getElementById("result-error");

const verdictIcon = document.getElementById("verdict-icon");
const verdictLabel = document.getElementById("verdict-label");
const verdictRisk = document.getElementById("verdict-risk");
const confidenceValue = document.getElementById("confidence-value");
const confidenceFill = document.getElementById("confidence-fill");
const signalsList = document.getElementById("signals-list");

emailInput.addEventListener("input", () => {
  charCount.textContent = `${emailInput.value.length} characters`;
});

analyzeBtn.addEventListener("click", analyzeEmail);
emailInput.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") analyzeEmail();
});

async function analyzeEmail() {
  const text = emailInput.value.trim();
  if (!text) {
    emailInput.focus();
    return;
  }

  setLoading(true);
  hideAllResultStates();

  try {
    const data = await apiRequest("/predict", "POST", { email: text }, true);
    renderResult(data);
  } catch (err) {
    renderError(err.message || "Could not reach the API. Is the backend running?");
  } finally {
    setLoading(false);
  }
}

function setLoading(isLoading) {
  analyzeBtn.disabled = isLoading;
  analyzeBtn.querySelector(".btn__label").textContent = isLoading
    ? "Analyzing..."
    : "Analyze email";
}

function hideAllResultStates() {
  resultEmpty.classList.add("hidden");
  resultContent.classList.add("hidden");
  resultError.classList.add("hidden");
  resultPanel.classList.add("has-content");
}

function renderResult(data) {
  const isPhishing = data.prediction === "PHISHING";

  verdictIcon.textContent = isPhishing ? "⚠" : "✓";
  verdictLabel.textContent = isPhishing ? "PHISHING DETECTED" : "LOOKS LEGITIMATE";
  verdictRisk.textContent = `Risk level: ${data.risk}`;

  const verdictWrap = verdictIcon.closest(".verdict");
  verdictWrap.className = "verdict " + (isPhishing ? "verdict--danger" : "verdict--safe");

  confidenceValue.textContent = `${data.confidence.toFixed(2)}%`;
  confidenceFill.style.width = `${data.confidence}%`;
  confidenceFill.className = "confidence__fill " + (isPhishing ? "confidence__fill--danger" : "confidence__fill--safe");

  signalsList.innerHTML = "";
  if (data.signals && data.signals.length > 0) {
    data.signals.forEach((s) => {
      const chip = document.createElement("span");
      chip.className = "signal-chip";
      chip.textContent = s;
      signalsList.appendChild(chip);
    });
  } else {
    const none = document.createElement("span");
    none.className = "signals__none";
    none.textContent = "No strong keyword signals found.";
    signalsList.appendChild(none);
  }

  resultContent.classList.remove("hidden");
}

function renderError(message) {
  resultPanel.classList.add("has-content");
  resultError.textContent = message;
  resultError.classList.remove("hidden");
}
