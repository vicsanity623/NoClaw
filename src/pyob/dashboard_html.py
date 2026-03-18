OBSERVER_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PyOB // ARCHITECT HUD</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;600&display=swap" rel="stylesheet">
    <style>
        :root { --bg: #0a0a0c; --card: #141417; --accent: #00ffa3; --text: #e0e0e6; --dim: #88888e; --err: #ff4d4d; }
        * { box-sizing: border-box; }
        body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; margin: 0; padding: 15px; line-height: 1.5; }
        .hud-container { max-width: 1200px; margin: 0 auto; display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        /* Typography & Glow */
        h1 { grid-column: span 2; font-family: 'JetBrains Mono'; font-size: 1.2rem; letter-spacing: 2px; color: var(--accent); text-transform: uppercase; margin: 10px 0; display: flex; justify-content: space-between; }
        .glow { text-shadow: 0 0 15px var(--accent); }
        /* Component Cards */
        .card { background: var(--card); border: 1px solid #2a2a30; border-radius: 8px; padding: 20px; overflow: hidden; position: relative; }
        .card::before { content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 2px; background: linear-gradient(90deg, transparent, var(--accent), transparent); opacity: 0.3; }
        .label { font-size: 0.7rem; font-weight: 600; color: var(--dim); text-transform: uppercase; margin-bottom: 12px; letter-spacing: 1px; display: flex; align-items: center; gap: 8px; }
        .label::before { content: ''; width: 6px; height: 6px; background: var(--accent); border-radius: 50%; box-shadow: 0 0 8px var(--accent); }
        /* Data Displays */
        .data-box { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; height: 250px; overflow-y: auto; background: #00000044; border-radius: 4px; padding: 12px; color: #ced4e0; scrollbar-width: thin; }
        .stat-grid { grid-column: span 2; display: flex; gap: 40px; background: var(--card); padding: 15px 25px; border-radius: 8px; border: 1px solid #2a2a30; }
        .stat-item { display: flex; flex-direction: column; }
        .stat-val { font-size: 1.5rem; font-weight: 700; font-family: 'JetBrains Mono'; color: #fff; }
        .stat-lbl { font-size: 0.6rem; color: var(--dim); }
        /* Mobile Specifics */
        @media (max-width: 768px) {
            .hud-container { grid-template-columns: 1fr; }
            h1, .stat-grid { grid-column: 1; }
            .stat-grid { flex-wrap: wrap; gap: 20px; }
        }
        .status-pill { padding: 4px 12px; border-radius: 20px; font-size: 0.7rem; font-weight: 800; background: #222; }
        .evolving { color: var(--accent); border: 1px solid var(--accent); box-shadow: 0 0 10px #00ffa344; }
        input, textarea { background: #000; border: 1px solid #2a2a30; color: var(--accent); padding: 10px; border-radius: 4px; width: 100%; font-family: 'JetBrains Mono'; margin-bottom: 10px; }
        button { width: 100%; padding: 12px; background: var(--accent); color: #000; border: none; border-radius: 4px; font-weight: 700; cursor: pointer; transition: 0.2s; }
        button:hover { filter: brightness(1.2); }
        /* Specific styles for queue items */
        .queue-item {
            margin-bottom: 8px;
            padding: 8px;
            background: #00000066;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-family: 'JetBrains Mono';
            font-size: 0.8em;
            color: #ced4e0;
        }
        .queue-item button {
            width: auto;
            padding: 5px 10px;
            font-size: 0.7em;
            margin-left: 5px;
            border-radius: 3px;
        }
        .queue-item .move-btn { background: #4a4a50; color: var(--text); }
        .queue-item .remove-btn { background: #cc0000; color: #fff; }
    </style>
    <!-- Add Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>
        <span>PyOB // Evolution Engine</span>
        <span id="status-pill" class="status-pill">READY</span>
    </h1>
    <div class="hud-container">
        <div class="stat-grid">
            <div class="stat-item"><span class="stat-lbl">Iteration</span><span id="iteration" class="stat-val">--</span></div>
            <div class="stat-item"><span class="stat-lbl">Symbolic Ledger</span><span id="ledger" class="stat-val">--</span></div>
            <div class="stat-item"><span class="stat-lbl">Pending Cascades</span><span id="queue-count" class="stat-val">--</span></div>
        </div>
        <!-- Chart Container -->
        <div class="card" style="grid-column: span 2;">
            <div class="label">Interactive Visualization</div>
            <canvas id="statsChart"></canvas>
        </div>
        <div class="card">
            <div class="label">Logic Memory (MEMORY.md)</div>
            <textarea id="memory" class="data-box" style="height: 250px;">Initializing brain...</textarea>
            <button onclick="saveMemory()" style="margin-top: 10px;">SAVE MEMORY</button>
        </div>
        <div class="card">
            <div class="label">System Logs (HISTORY.md)</div>
            <div id="history" class="data-box">No history yet.</div>
        </div>
        <div class="card" style="grid-column: span 2;">
            <div class="label">Architectural Analysis</div>
            <div id="analysis" class="data-box" style="height: 350px;">Scanning structure...</div>
        </div>
        <div class="card" style="grid-column: span 2;">
            <div class="label">Pending Patch Reviews</div>
            <div id="pending-patches" class="data-box" style="height: 250px;">No pending patches.</div>
        </div>
        <div class="card">
            <div class="label">Manual Override</div>
            <input type="text" id="manualTargetFile" placeholder="src/pyob/target.py">
            <button onclick="setManualTarget()">FORCE TARGET</button>
        </div>
        <div class="card">
            <div class="label">Manual Cascade Injection</div>
            <input type="text" id="cascadeItem" placeholder="e.g., src/pyob/new_feature.py or 'Refactor dashboard'">
            <button onclick="addCascadeItem()" style="margin-top: 10px;">ADD TO QUEUE</button>
        </div>
        <div class="card">
            <div class="label">Queue Status</div>
            <div id="queue" class="data-box" style="height: 200px;">IDLE</div>
        </div>
    </div>

<!-- New Diff Modal HTML -->
<div id="diffModal" style="display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.8);">
    <div style="background-color: var(--card); margin: 10% auto; padding: 20px; border: 1px solid #888; width: 80%; max-width: 900px; border-radius: 8px; position: relative;">
        <span onclick="closeDiffModal()" style="color: var(--dim); float: right; font-size: 28px; font-weight: bold; cursor: pointer;">&times;</span>
        <h2 style="color: var(--accent); font-family: 'JetBrains Mono'; margin-bottom: 15px;">Patch Diff <span id="diffModalPatchId"></span></h2>
        <pre id="diffContent" class="data-box" style="height: 400px; white-space: pre-wrap; word-wrap: break-word;"></pre>
    </div>
</div>

    <script>
        async function updateStats() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                document.getElementById('iteration').innerText = data.iteration || "0";
                document.getElementById('ledger').innerText = (data.ledger_stats?.definitions || 0) + " SYM";
                document.getElementById('queue-count').innerText = data.cascade_queue?.length || "0";
                const pill = document.getElementById('status-pill');
                const isEvolving = data.cascade_queue?.length > 0 || data.patches_count > 0;
                pill.innerText = isEvolving ? "EVOLVING" : "STABLE";
                pill.className = isEvolving ? "status-pill evolving" : "status-pill";
                document.getElementById('memory').value = data.memory || "Brain empty.";
                document.getElementById('history').innerText = data.history || "No logs.";
                // Handle structured analysis data for interactive issues
                const analysisContainer = document.getElementById('analysis');
                analysisContainer.innerHTML = ''; // Clear existing content

                // The acknowledgeIssue function is now defined globally below updateStats.
                // No local definition needed here.

                if (Array.isArray(data.analysis) && data.analysis.length > 0) {
                    data.analysis.forEach(issue => {
                        const issueElement = document.createElement('div');
                        issueElement.style.marginBottom = '8px';
                        issueElement.style.padding = '8px';
                        issueElement.style.background = '#00000066';
                        issueElement.style.borderRadius = '4px';
                        issueElement.style.display = 'flex';
                        issueElement.style.alignItems = 'center';
                        issueElement.style.justifyContent = 'space-between';
                        issueElement.style.fontFamily = 'JetBrains Mono';
                        issueElement.style.fontSize = '0.8em';
                        issueElement.style.color = '#ced4e0';

                        const statusColor = issue.status === 'acknowledged' ? 'var(--dim)' : 'var(--err)';
                        const statusText = issue.status === 'acknowledged' ? 'ACKNOWLEDGED' : 'PENDING';

                        issueElement.innerHTML = `
                            <span style="flex-grow: 1;">${issue.description}</span>
                            <span style="font-size: 0.7em; font-weight: 600; color: ${statusColor}; margin-left: 10px;">${statusText}</span>
                            <button onclick="acknowledgeIssue('${issue.id}')"
                                    style="width: auto; padding: 5px 10px; font-size: 0.7em; border-radius: 3px; margin-left: 10px;
                                           background: ${issue.status === 'acknowledged' ? '#4a4a50' : 'var(--accent)'};
                                           color: ${issue.status === 'acknowledged' ? 'var(--text)' : '#000'};"
                                    ${issue.status === 'acknowledged' ? 'disabled' : ''}>
                                ${issue.status === 'acknowledged' ? 'ACKNOWLEDGED' : 'ACKNOWLEDGE'}
                            </button>
                        `;
                        analysisContainer.appendChild(issueElement);
                    });
                } else {
                    analysisContainer.innerText = data.analysis || "No issues found."; // Fallback if not array or empty
                }

                // Update chart after fetching data
                updateChart(data);

                const queueDiv = document.getElementById('queue');
                queueDiv.innerHTML = '';
                if (data.cascade_queue && data.cascade_queue.length > 0) {
                    data.cascade_queue.forEach((item, index) => {
                        const itemElement = document.createElement('div');
                        itemElement.className = 'queue-item';

                        const textSpan = document.createElement('span');
                        textSpan.style.flexGrow = '1';
                        textSpan.innerText = item; // Use innerText for safety

                        const buttonDiv = document.createElement('div');
                        buttonDiv.style.display = 'flex';
                        buttonDiv.style.gap = '5px';

                        const moveUpBtn = document.createElement('button');
                        moveUpBtn.className = 'move-btn';
                        moveUpBtn.innerHTML = '&#x25B2;';
                        moveUpBtn.onclick = () => moveQueueItem(item, 'up');

                        const moveDownBtn = document.createElement('button');
                        moveDownBtn.className = 'move-btn';
                        moveDownBtn.innerHTML = '&#x25BC;';
                        moveDownBtn.onclick = () => moveQueueItem(item, 'down');

                        const removeBtn = document.createElement('button');
                        removeBtn.className = 'remove-btn';
                        removeBtn.innerHTML = 'X';
                        removeBtn.onclick = () => removeQueueItem(item);

                        buttonDiv.appendChild(moveUpBtn);
                        buttonDiv.appendChild(moveDownBtn);
                        buttonDiv.appendChild(removeBtn);

                        itemElement.appendChild(textSpan);
                        itemElement.appendChild(buttonDiv);
                        queueDiv.appendChild(itemElement);
                    });
                } else {
                    queueDiv.innerText = "EMPTY";
                }

                await updatePendingPatches();
            } catch (e) { document.getElementById('status-pill').innerText = "OFFLINE"; console.error("Error updating stats:", e); }
        }

        async function acknowledgeIssue(issueId) {
            try {
                const response = await fetch(`/api/acknowledge-issue/${issueId}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ status: 'acknowledged' })
                });
                if (response.ok) {
                    await updateStats(); // Refresh dashboard to show updated status
                } else {
                    console.error(`Failed to acknowledge issue ${issueId}:`, await response.text());
                    alert('Error acknowledging issue.');
                }
            } catch (e) {
                console.error(`Failed to acknowledge issue ${issueId}:`, e);
                alert('Network error acknowledging issue.');
            }
        }

        async function setManualTarget() {
            const targetFile = document.getElementById('manualTargetFile').value;
            if (!targetFile) return;
            await fetch('/api/set_target_file', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ target_file: targetFile })
            });
            document.getElementById('manualTargetFile').value = '';
        }

        async function updatePendingPatches() {
            try {
                const response = await fetch('/api/pending_patches');
                const data = await response.json();
                const patchesDiv = document.getElementById('pending-patches');
                patchesDiv.innerHTML = '';

                if (data.patches && data.patches.length > 0) {
                    data.patches.forEach(patch => {
                        const patchElement = document.createElement('div');
                        patchElement.style.marginBottom = '10px';
                        patchElement.innerHTML = `
                            <span style="color: var(--accent); font-weight: 700;">Patch ID: ${patch.id}</span><br>
                            <span class="patch-file" style="font-size: 0.8em; color: var(--dim);">File: ${patch.file || 'N/A'}</span><br>
                            <span class="patch-description" style="font-size: 0.8em; color: var(--dim);">Description: ${patch.description || 'No description'}</span><br>
<button onclick="viewPatchDiff('${patch.id}')" style="width: 31%; margin-right: 2%; background: #007bff; color: #fff;">VIEW DIFF</button>
<button onclick="reviewPatch('${patch.id}', 'approved')" style="width: 31%; margin-right: 2%; background: #00cc00; color: #000;">APPROVE</button>
<button onclick="reviewPatch('${patch.id}', 'reject')" style="width: 23%; background: #cc0000; color: #fff;">REJECT</button>
                        `;
                        patchesDiv.appendChild(patchElement);
                    });
                } else {
                    patchesDiv.innerText = "No pending patches.";
                }
            } catch (e) {
                console.error("Failed to fetch pending patches:", e);
                document.getElementById('pending-patches').innerText = "Error loading patches.";
            }
        }

        async function reviewPatch(patchId, action) {
            try {
                if (action === 'approved') {
                    await fetch('/api/approve_patch', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ patch_id: patchId })
                    });
                } else {
                    await fetch('/api/review_patch', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ patch_id: patchId, action: action })
                    });
                }
                await updateStats();
            } catch (e) {
                console.error(`Failed to ${action} patch ${patchId}:`, e);
            }
        }

        async function saveMemory() {
            const memoryContent = document.getElementById('memory').value;
            try {
                const response = await fetch('/api/update_memory', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ content: memoryContent })
                });
                if (response.ok) {
                    alert('Logic Memory saved successfully!');
                    await updateStats();
                }
            } catch (e) {
                console.error("Failed to save Logic Memory:", e);
            }
        }

        async function addCascadeItem() {
            const item = document.getElementById('cascadeItem').value;
            if (!item) return;
            try {
                const response = await fetch('/api/cascade_queue/add', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ item: item })
                });
                if (response.ok) {
                    document.getElementById('cascadeItem').value = '';
                    await updateStats();
                }
            } catch (e) {
                console.error("Failed to add item to cascade queue:", e);
            }
        }

        async function moveQueueItem(itemId, direction) {
            try {
                await fetch('/api/cascade_queue/move', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ item_id: itemId, direction: direction })
                });
                await updateStats();
            } catch (e) {
                console.error(`Failed to move item ${itemId} ${direction}:`, e);
            }
        }

        async function removeQueueItem(itemId) {
            if (!confirm(`Are you sure you want to remove "${itemId}" from the queue?`)) return;
            try {
                await fetch('/api/cascade_queue/remove', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ item_id: itemId })
                });
                await updateStats();
            } catch (e) {
                console.error(`Failed to remove item ${itemId}:`, e);
            }
        }

        // New JavaScript functions

    async function viewPatchDiff(patchId) {
        try {
            const response = await fetch(`/api/patch_diff/${patchId}`);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json();
            document.getElementById('diffModalPatchId').innerText = `(ID: ${patchId})`;
            document.getElementById('diffContent').innerText = data.diff || "No diff available.";
            openDiffModal();
        } catch (e) {
            console.error(`Failed to fetch diff for patch ${patchId}:`, e);
            alert(`Error viewing diff: ${e.message}`);
        }
    }

    function openDiffModal() {
        document.getElementById('diffModal').style.display = 'block';
    }

    function closeDiffModal() {
        document.getElementById('diffModal').style.display = 'none';
        document.getElementById('diffContent').innerText = ''; // Clear content on close
        document.getElementById('diffModalPatchId').innerText = '';
    }

        setInterval(updateStats, 3000);
        updateStats();

        function updateChart(data) {
            const ctx = document.getElementById('statsChart').getContext('2d');
            const iterationData = data.iterationHistory ? data.iterationHistory.map((val, index) => { return { x: index, y: val }; }) : [];

            if (window.statsChart) {
                window.statsChart.data.labels = iterationData.map(d => d.x);
                window.statsChart.data.datasets[0].data = iterationData.map(d => d.y);
                window.statsChart.update();
            } else {
                window.statsChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: iterationData.map(d => d.x),
                        datasets: [{
                            label: 'Iterations Over Time',
                            data: iterationData.map(d => d.y),
                            borderColor: 'rgb(75, 192, 192)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                });
            }
        }
    </script>
</body>
</html>
"""
