let signalChart;
let bpmChart;

function createCharts() {
    const signalCtx = document.getElementById("signalChart").getContext("2d");
    const bpmCtx = document.getElementById("bpmChart").getContext("2d");

    signalChart = new Chart(signalCtx, {
        type: "line",
        data: {
            labels: [],
            datasets: [{
                label: "Signal",
                data: [],
                borderWidth: 2,
                fill: false,
                tension: 0.25
            }]
        },
        options: {
            responsive: true,
            animation: false
        }
    });

    bpmChart = new Chart(bpmCtx, {
        type: "line",
        data: {
            labels: [],
            datasets: [{
                label: "BPM",
                data: [],
                borderWidth: 2,
                fill: false,
                tension: 0.25
            }]
        },
        options: {
            responsive: true,
            animation: false
        }
    });
}

async function fetchStats() {
    const response = await fetch("/stats");
    const data = await response.json();

    document.getElementById("currentBpm").innerText = data.current_bpm;
    document.getElementById("avgBpm").innerText = data.avg_bpm;
    document.getElementById("minBpm").innerText = data.min_bpm;
    document.getElementById("maxBpm").innerText = data.max_bpm;
    document.getElementById("samplesCount").innerText = data.samples;
    document.getElementById("modeText").innerText = "Mode: " + data.mode;
}

async function fetchCharts() {
    const response = await fetch("/analysis_data");
    const data = await response.json();

    signalChart.data.labels = data.signal_times.map(x => x.toFixed(1));
    signalChart.data.datasets[0].data = data.signal_values;
    signalChart.update();

    bpmChart.data.labels = data.bpm_times.map(x => x.toFixed(1));
    bpmChart.data.datasets[0].data = data.bpm_values;
    bpmChart.update();
}

async function resetSession() {
    await fetch("/reset", {
        method: "POST"
    });
}

async function enableDemo() {
    await fetch("/demo_mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: true })
    });
}

async function disableDemo() {
    await fetch("/demo_mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: false })
    });
}

createCharts();

setInterval(() => {
    fetchStats();
    fetchCharts();
}, 1500);
