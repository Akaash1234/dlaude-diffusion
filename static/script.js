// dlaude chat interface
const state = { currentConversationId: null, isGenerating: false, modelLoaded: false, isExtendedThinking: false };

const el = {
    messageInput: document.getElementById('messageInput'),
    inputPlaceholder: document.getElementById('inputPlaceholder'),
    sendBtn: document.getElementById('sendBtn'),
    messagesContainer: document.getElementById('messagesContainer'),
    messages: document.getElementById('messages'),
    welcomeScreen: document.getElementById('welcomeScreen'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    loadingStatus: document.getElementById('loadingStatus'),
    loadingDetail: document.getElementById('loadingDetail'),
    loadingFill: document.getElementById('loadingFill'),
    generationStatus: document.getElementById('generationStatus'),
    progressFill: document.getElementById('progressFill'),
    stepCounter: document.getElementById('stepCounter'),
    conversationList: document.getElementById('conversationList'),
    extendedThinkingBtn: document.getElementById('extendedThinkingBtn'),
    newChatBtn: document.getElementById('newChatBtn')
};

document.addEventListener('DOMContentLoaded', () => { initUI(); loadConversations(); checkModelStatus(); });

function initUI() {
    el.messageInput.addEventListener('input', () => {
        el.inputPlaceholder.style.display = el.messageInput.value ? 'none' : 'block';
        el.sendBtn.disabled = !el.messageInput.value.trim();
        // auto resize
        el.messageInput.style.height = 'auto';
        el.messageInput.style.height = Math.min(el.messageInput.scrollHeight, 200) + 'px';
    });

    el.messageInput.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });
    el.sendBtn.addEventListener('click', sendMessage);

    el.newChatBtn.addEventListener('click', () => {
        state.currentConversationId = null;
        el.messages.innerHTML = '';
        el.welcomeScreen.classList.remove('hidden');
        el.messagesContainer.classList.remove('active');
        el.messageInput.value = '';
        el.inputPlaceholder.style.display = 'block';
    });

    el.extendedThinkingBtn.addEventListener('click', () => {
        state.isExtendedThinking = !state.isExtendedThinking;
        el.extendedThinkingBtn.classList.toggle('active');
    });
}

async function checkModelStatus() {
    try {
        const res = await fetch('/api/model/status');
        const s = await res.json();
        if (s.loaded) {
            el.loadingOverlay.classList.add('hidden');
            state.modelLoaded = true;
        } else {
            el.loadingOverlay.classList.remove('hidden');
            el.loadingStatus.textContent = "Loading Dlaude...";
            el.loadingDetail.textContent = s.message || "Initializing...";
            el.loadingFill.style.width = `${s.progress || 0}%`;
            setTimeout(checkModelStatus, 1000);
        }
    } catch (e) {
        console.error(e);
        setTimeout(checkModelStatus, 2000);
    }
}

async function loadConversations() {
    try {
        const res = await fetch('/api/conversations');
        const chats = await res.json();
        el.conversationList.innerHTML = chats.map(c =>
            `<div class="recent-item" onclick="loadChat('${c.id}')">${escapeHtml(c.title)}</div>`
        ).join('');
    } catch (e) { console.error(e); }
}

async function loadChat(id) {
    state.currentConversationId = id;
    const res = await fetch(`/api/conversations/${id}`);
    const data = await res.json();
    el.welcomeScreen.classList.add('hidden');
    el.messagesContainer.classList.add('active');
    el.messages.innerHTML = data.messages.map(m =>
        `<div class="message-row ${m.role}"><div>${formatText(m.content)}</div></div>`
    ).join('');
    el.messagesContainer.scrollTop = el.messagesContainer.scrollHeight;
}
window.loadChat = loadChat;  // expose for onclick

async function sendMessage() {
    const text = el.messageInput.value.trim();
    if (!text || state.isGenerating) return;

    state.isGenerating = true;
    el.messageInput.value = '';
    el.inputPlaceholder.style.display = 'block';
    el.sendBtn.disabled = true;
    el.welcomeScreen.classList.add('hidden');
    el.messagesContainer.classList.add('active');

    appendMessage('user', text);

    const assistantDiv = document.createElement('div');
    assistantDiv.className = 'message-row assistant';
    el.messages.appendChild(assistantDiv);

    el.generationStatus.classList.add('active');
    updateProgress(0);

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                conversation_id: state.currentConversationId,
                message: text,
                steps: state.isExtendedThinking ? 64 : 32
            })
        });

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let fullText = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data:')) continue;
                const data = JSON.parse(line.slice(5));

                if (data.conversation_id) state.currentConversationId = data.conversation_id;
                if (data.text) {
                    fullText = data.text;
                    assistantDiv.innerHTML = formatText(fullText);
                    el.messagesContainer.scrollTop = el.messagesContainer.scrollHeight;
                }
                if (data.step !== undefined) updateProgress(data.step, data.total_steps || 32);
                if (data.full_response) {
                    fullText = data.full_response;
                    assistantDiv.innerHTML = formatText(fullText);
                }
            }
        }
    } catch (e) {
        assistantDiv.textContent = "Error: " + e.message;
    } finally {
        state.isGenerating = false;
        el.generationStatus.classList.remove('active');
        loadConversations();
    }
}

function appendMessage(role, text) {
    const div = document.createElement('div');
    div.className = `message-row ${role}`;
    div.innerHTML = formatText(text);
    el.messages.appendChild(div);
    el.messagesContainer.scrollTop = el.messagesContainer.scrollHeight;
}

function updateProgress(step, total = 32) {
    el.stepCounter.textContent = `Step ${step}/${total}`;
    el.progressFill.style.width = `${(step / total) * 100}%`;
}

function formatText(t) {
    return t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\n/g, "<br>");
}

function escapeHtml(t) {
    const d = document.createElement('div');
    d.textContent = t;
    return d.innerHTML;
}
