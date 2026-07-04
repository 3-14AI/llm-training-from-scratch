import re

with open("web_app/frontend/index.html", "r", encoding="utf-8") as f:
    html_content = f.read()

# CSS changes
css_insertion = """
        .dashboard {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        .dashboard button {
            background-color: var(--tg-theme-button-color, #4CAF50);
        }
"""
html_content = html_content.replace("</style>", css_insertion + "</style>")

# HTML Dashboard changes
dashboard_html = """
    <div class="dashboard">
        <button onclick="runScript('pretraining')">Pretraining</button>
        <button onclick="runScript('finetuning')">Fine-tuning</button>
        <button onclick="runScript('evaluation')">Evaluation</button>
    </div>

    <div id="terminal-container">
"""
html_content = html_content.replace('<div id="terminal-container">', dashboard_html)

# JS changes for runScript and WebSockets
js_insertion = """
        // WebSocket setup for logs
        let ws;
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/logs`;
            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                appendToTerminal('WebSocket connected to /ws/logs', 'command');
            };

            ws.onmessage = (event) => {
                appendToTerminal(event.data);
            };

            ws.onclose = () => {
                appendToTerminal('WebSocket connection closed.', 'error');
                // Optional: Attempt to reconnect after a delay
                // setTimeout(connectWebSocket, 5000);
            };

            ws.onerror = (error) => {
                appendToTerminal(`WebSocket Error: ${error}`, 'error');
            };
        }

        connectWebSocket();

        async function runScript(scriptType) {
            appendToTerminal(`> Starting script: ${scriptType}...`, 'command');
            try {
                const response = await fetch('/api/run_script', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ script_type: scriptType })
                });

                const data = await response.json();
                if (!response.ok) {
                    appendToTerminal(`Error starting script: ${data.detail}`, 'error');
                } else {
                    appendToTerminal(`Process started: ${data.message}`, 'command');
                }
            } catch (error) {
                appendToTerminal(`Network Error: ${error.message}`, 'error');
            }
        }
"""
html_content = html_content.replace("const initDataUnsafe = window.Telegram.WebApp.initDataUnsafe || {};", "const initDataUnsafe = window.Telegram.WebApp.initDataUnsafe || {};\n" + js_insertion)


with open("web_app/frontend/index.html", "w", encoding="utf-8") as f:
    f.write(html_content)
