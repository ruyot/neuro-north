const root = document.documentElement;
const sentence = document.querySelector("#sentence");
const status = document.querySelector("#status");
const targets = document.querySelector("#targets");
const pagePill = document.querySelector("#page-pill");
const confidence = document.querySelector("#confidence");
const confidenceOutput = document.querySelector("#confidence-output");
const candidateList = document.querySelector("#candidate-list");
const candidateCount = document.querySelector("#candidate-count");
const metrics = document.querySelector("#metrics");
const trace = document.querySelector("#trace");
const positionCount = document.querySelector("#position-count");
const topControl = document.querySelector("#candidate-top");
const bottomControl = document.querySelector("#candidate-bottom");
const toast = document.querySelector("#toast");

let state = null;
let busy = false;
let toastTimer = null;

function percent(value) {
  return `${Math.round(value * 100)}%`;
}

function showError(message) {
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.hidden = true; }, 4200);
}

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const body = await response.json();
  if (!response.ok || !body.ok) throw new Error(body.error || "Request failed");
  return body.state;
}

async function act(action, payload = {}) {
  if (busy) return;
  busy = true;
  root.classList.add("busy");
  try {
    state = await request("/api/action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, ...payload }),
    });
    render();
  } catch (error) {
    showError(error.message);
  } finally {
    busy = false;
    root.classList.remove("busy");
  }
}

function renderSentence() {
  sentence.replaceChildren();
  if (!state.sentence) {
    const empty = document.createElement("span");
    empty.className = "placeholder";
    empty.textContent = "No words confirmed yet";
    sentence.append(empty);
  } else {
    sentence.textContent = state.sentence;
  }
  status.textContent = state.last_event;
}

function renderTargets() {
  targets.replaceChildren();
  state.targets.forEach((target, index) => {
    const button = document.createElement("button");
    button.className = "target";
    button.type = "button";
    button.setAttribute("aria-label", `Select ${target.range} at simulated ${target.frequency} hertz`);

    const shortcut = document.createElement("span");
    shortcut.className = "target-index";
    shortcut.textContent = `KEY ${index + 1}`;
    const frequency = document.createElement("span");
    frequency.className = "frequency";
    frequency.textContent = `${target.frequency} Hz`;
    const range = document.createElement("strong");
    range.className = "range-name";
    range.textContent = target.range;
    const caption = document.createElement("span");
    caption.className = "target-caption";
    caption.textContent = "Simulated focus";

    button.append(shortcut, frequency, range, caption);
    button.addEventListener("click", () => act("select", {
      range: target.range,
      confidence: Number(confidence.value) / 100,
    }));
    targets.append(button);
  });
  pagePill.textContent = `Page ${state.page_number} / ${state.page_count}`;
}

function setPredictionControl(button, candidate, direction) {
  const label = button.querySelector(".direction");
  const word = button.querySelector("strong");
  label.textContent = `${direction} prediction`;
  if (!candidate) {
    button.disabled = true;
    word.textContent = "—";
    button.onclick = null;
    return;
  }
  button.disabled = false;
  word.textContent = `${candidate.word} · ${percent(candidate.probability)}`;
  button.onclick = () => act("accept", { word: candidate.word });
}

function renderCandidates() {
  candidateList.replaceChildren();
  state.candidates.forEach((candidate) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.className = "candidate-row";
    button.type = "button";
    button.title = "Accept this word";

    const word = document.createElement("span");
    word.className = "candidate-word";
    word.textContent = candidate.word;
    const meta = document.createElement("span");
    meta.className = "candidate-meta";
    if (!candidate.complete && state.observation_count > 0) {
      const completion = document.createElement("span");
      completion.className = "completion-tag";
      completion.textContent = "completion";
      meta.append(completion);
    }
    const probability = document.createElement("span");
    probability.className = "probability";
    probability.textContent = percent(candidate.probability);
    meta.append(probability);

    button.append(word, meta);
    button.addEventListener("click", () => act("accept", { word: candidate.word }));
    item.append(button);
    candidateList.append(item);
  });

  candidateCount.textContent = `${state.candidate_count} candidate${state.candidate_count === 1 ? "" : "s"}`;
  const values = [
    state.ambiguity.top_probability,
    state.ambiguity.top_two_margin,
    state.ambiguity.normalized_entropy,
  ];
  [...metrics.querySelectorAll("strong")].forEach((node, index) => {
    node.textContent = percent(values[index]);
  });
  setPredictionControl(topControl, state.candidates[0], "Top");
  setPredictionControl(bottomControl, state.candidates[1], "Bottom");
}

function renderTrace() {
  trace.replaceChildren();
  if (!state.history.length) {
    const empty = document.createElement("span");
    empty.className = "placeholder";
    empty.textContent = "Click a target to add the first observation.";
    trace.append(empty);
  } else {
    state.history.forEach((event) => {
      const item = document.createElement("div");
      item.className = "trace-item";
      const position = document.createElement("div");
      position.className = "trace-position";
      position.textContent = `Position ${event.position} · Page ${event.page + 1}`;
      const choice = document.createElement("div");
      choice.className = "trace-choice";
      choice.textContent = event.selected;
      const probabilities = document.createElement("div");
      probabilities.className = "trace-probs";
      Object.entries(event.probabilities).forEach(([label, value]) => {
        const probability = document.createElement("span");
        probability.textContent = `${label} ${percent(value)}`;
        probabilities.append(probability);
      });
      item.append(position, choice, probabilities);
      trace.append(item);
    });
  }
  positionCount.textContent = `${state.observation_count} position${state.observation_count === 1 ? "" : "s"}`;
}

function render() {
  renderSentence();
  renderTargets();
  renderCandidates();
  renderTrace();
}

confidence.addEventListener("input", () => {
  confidenceOutput.textContent = `${confidence.value}%`;
});
document.querySelector("#page").addEventListener("click", () => act("page"));
document.querySelector("#space").addEventListener("click", () => act("space"));
document.querySelector("#backspace").addEventListener("click", () => act("backspace"));
document.querySelector("#clear").addEventListener("click", () => act("clear"));
document.querySelector("#reset").addEventListener("click", () => act("reset"));

document.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement) return;
  if (event.key === "1" || event.key === "2") {
    const target = state?.targets[Number(event.key) - 1];
    if (target) act("select", { range: target.range, confidence: Number(confidence.value) / 100 });
  } else if (event.key === "ArrowLeft") {
    event.preventDefault();
    act("page");
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    act("space");
  } else if (event.key === "Backspace") {
    event.preventDefault();
    act("backspace");
  }
});

request("/api/state")
  .then((initialState) => { state = initialState; render(); })
  .catch((error) => showError(error.message));
