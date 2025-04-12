const particlesContainer = document.getElementById('particles');
function createParticle() {
    const particle = document.createElement('div');
    particle.style.position = 'absolute';
    particle.style.width = '5px';
    particle.style.height = '5px';
    particle.style.background = `hsl(${Math.random() * 60 + 30}, 80%, 60%)`; // Golden hues
    particle.style.borderRadius = '50%';
    particle.style.opacity = '0.5';
    particle.style.left = Math.random() * 100 + 'vw';
    particle.style.top = Math.random() * 100 + 'vh';
    particle.style.animation = `float ${Math.random() * 6 + 6}s infinite ease-in-out`;
    particlesContainer.appendChild(particle);
    setTimeout(() => particle.remove(), 6000);
}

const styleSheet = document.createElement('style');
styleSheet.innerHTML = `
    @keyframes float {
        0% { transform: translateY(0) rotate(0deg); opacity: 0.5; }
        50% { opacity: 0.7; transform: translateY(-30vh) rotate(180deg) translateX(${Math.random() * 15 - 7.5}vw); }
        100% { transform: translateY(-100vh) rotate(360deg); opacity: 0; }
    }
`;
document.head.appendChild(styleSheet);
setInterval(createParticle, 250);

const clickSound = document.getElementById('clickSound');
const successSound = document.getElementById('successSound');
const newsInput = document.getElementById('newsInput');
const topicInput = document.getElementById('topicInput');
const spinner = document.getElementById('spinner');
const resultModal = document.getElementById('resultModal');
const modalTitle = document.getElementById('modalTitle');
const modalContent = document.getElementById('modalContent');

function showSpinner() {
    spinner.style.display = 'block';
    spinner.style.animation = 'pulse 1s infinite alternate';
}

function hideSpinner() {
    spinner.style.display = 'none';
}

function showModal(title, content) {
    modalTitle.textContent = title;
    modalContent.innerHTML = content;
    resultModal.style.display = 'flex';
    successSound.play();
    modalContent.style.animation = 'fadeIn 0.5s ease-in';
}

function closeModal() {
    resultModal.style.display = 'none';
    clickSound.play();
    modalContent.style.animation = '';
}

async function analyzeNews() {
    clickSound.play();
    const text = newsInput.value.trim();
    if (text === '') {
        newsInput.style.borderColor = '#ff4444';
        newsInput.style.transform = 'scale(1.05) rotate(-2deg)';
        setTimeout(() => {
            newsInput.style.borderColor = '#d4af37';
            newsInput.style.transform = 'scale(1)';
        }, 800);
        return;
    }
    showSpinner();
    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ text: text })
        });
        const data = await response.json();
        hideSpinner();
        if (response.ok) {
            const color = data.prediction === "Real" ? "#00ff00" : "#ff4444";
            showModal("Analysis Result", `This article is predicted to be: <strong style="color: ${color}">${data.prediction}</strong>`);
        } else {
            showModal("Error", data.error || "Failed to analyze.");
        }
    } catch (error) {
        hideSpinner();
        showModal("Error", "Could not connect to the server.");
    }
}

async function summarizeNews() {
    clickSound.play();
    const text = newsInput.value.trim();
    if (text === '') {
        newsInput.style.borderColor = '#ff4444';
        newsInput.style.transform = 'scale(1.05) rotate(-2deg)';
        setTimeout(() => {
            newsInput.style.borderColor = '#d4af37';
            newsInput.style.transform = 'scale(1)';
        }, 800);
        return;
    }
    showSpinner();
    try {
        const response = await fetch('http://localhost:5000/summarize', {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ text: text })
        });
        const data = await response.json();
        hideSpinner();
        if (response.ok) {
            showModal("Summary", `<span style="color: #f1e05a">${data.summary}</span>`);
        } else {
            showModal("Error", data.error || "Failed to summarize.");
        }
    } catch (error) {
        hideSpinner();
        showModal("Error", "Could not connect to the server.");
    }
}

async function fetchNewsByTopic() {
    clickSound.play();
    const topic = topicInput.value.trim();
    if (topic === '') {
        topicInput.style.borderColor = '#ff4444';
        topicInput.style.transform = 'scale(1.05) rotate(-2deg)';
        setTimeout(() => {
            topicInput.style.borderColor = '#d4af37';
            topicInput.style.transform = 'scale(1)';
        }, 800);
        return;
    }
    showSpinner();
    try {
        const response = await fetch('http://localhost:5000/fetch_predict', {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ topic: topic })
        });
        const data = await response.json();
        hideSpinner();
        if (response.ok) {
            const results = data.results.map(r => {
                const color = r.prediction === "Real" ? "#00ff00" : "#ff4444";
                return `${r.text} -> <strong style="color: ${color}">${r.prediction}</strong>`;
            }).join('<br>');
            showModal("Topic Analysis", results || "No results found.");
        } else {
            showModal("Error", data.error || "Failed to fetch and analyze.");
        }
    } catch (error) {
        hideSpinner();
        showModal("Error", "Could not connect to the server.");
    }
}

const placeholderText = "Paste your news article here...";
let i = 0;
function typePlaceholder() {
    if (i < placeholderText.length) {
        newsInput.setAttribute('placeholder', placeholderText.substring(0, i + 1));
        i++;
        setTimeout(typePlaceholder, 70);
    } else {
        setTimeout(() => { i = 0; typePlaceholder(); }, 1200);
    }
}
typePlaceholder();

document.querySelectorAll('button').forEach(button => {
    button.addEventListener('click', () => clickSound.play());
});

const styleSheet2 = document.createElement('style');
styleSheet2.innerHTML = `
    @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.1); } 100% { transform: scale(1); } }
    @keyframes fadeIn { 0% { opacity: 0; } 100% { opacity: 1; } }
`;
document.head.appendChild(styleSheet2);