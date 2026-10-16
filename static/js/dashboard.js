/**
 * PoultryScan AI - Analytics Dashboard JavaScript
 * Handles: Chart.js Doughnut & Bar graphs, live polling of /api/stats every 5 seconds.
 */

document.addEventListener("DOMContentLoaded", () => {
  if (typeof Chart === "undefined" || !window.DASH) return;

  const dataCounts = window.DASH.counts || {};
  const labels = Object.keys(dataCounts);
  const values = Object.values(dataCounts);

  // Harmonious Agri-Tech Color Palette:
  // Coccidiosis: Amber (#f59e0b)
  // Healthy: Emerald Green (#16a34a)
  // New Castle Disease: Crimson Red (#ef4444)
  // Salmonella: Dark Coral (#e11d48)
  const colorMap = {
    "Coccidiosis": "#f59e0b",
    "Fowl Pox": "#8b5cf6",
    "Healthy": "#16a34a",
    "Newcastle Disease": "#ef4444",
    "New Castle Disease": "#ef4444",
    "Salmonella": "#e11d48"
  };

  const backgroundColors = labels.map((l) => colorMap[l] || "#15803d");

  // 1. Initialize Doughnut Chart
  const doughnutCtx = document.getElementById("doughnutChart");
  let doughnutChart = null;

  if (doughnutCtx) {
    doughnutChart = new Chart(doughnutCtx, {
      type: "doughnut",
      data: {
        labels: labels,
        datasets: [
          {
            data: values,
            backgroundColor: backgroundColors,
            borderColor: "#ffffff",
            borderWidth: 2,
            hoverOffset: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: {
              boxWidth: 12,
              padding: 16,
              font: {
                family: "'Plus Jakarta Sans', sans-serif",
                weight: "600",
                size: 12,
              },
            },
          },
        },
        cutout: "68%",
      },
    });
  }

  // 2. Initialize Bar Chart
  const barCtx = document.getElementById("barChart");
  let barChart = null;

  if (barCtx) {
    barChart = new Chart(barCtx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Detected Cases",
            data: values,
            backgroundColor: backgroundColors,
            borderRadius: 6,
            maxBarThickness: 36,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
        },
        scales: {
          y: {
            beginAtZero: true,
            ticks: { precision: 0 },
            grid: { color: "#f1f5f9" },
          },
          x: {
            grid: { display: false },
            ticks: {
              font: {
                family: "'Plus Jakarta Sans', sans-serif",
                weight: "600",
                size: 11,
              },
            },
          },
        },
      },
    });
  }

  // 3. Live Polling Loop (/api/stats every 5 seconds)
  const statsUrl = window.DASH.statsUrl;
  if (!statsUrl) return;

  setInterval(() => {
    fetch(statsUrl)
      .then((res) => res.json())
      .then((stats) => {
        // Update KPI values
        const statTotal = document.getElementById("statTotal");
        const statDiseased = document.getElementById("statDiseased");
        const statHealthy = document.getElementById("statHealthy");
        const statAvg = document.getElementById("statAvg");

        if (statTotal) statTotal.textContent = stats.total;
        if (statDiseased) statDiseased.textContent = stats.diseased;
        if (statHealthy) statHealthy.textContent = stats.healthy;
        if (statAvg) statAvg.textContent = stats.avg;

        // If new scans occurred, refresh chart datasets
        if (stats.counts) {
          const newValues = labels.map((l) => stats.counts[l] || 0);

          if (doughnutChart) {
            doughnutChart.data.datasets[0].data = newValues;
            doughnutChart.update();
          }

          if (barChart) {
            barChart.data.datasets[0].data = newValues;
            barChart.update();
          }

          const emptyChart = document.getElementById("emptyChart");
          const chartBox = document.getElementById("chartBox");
          const chartBox2 = document.getElementById("chartBox2");
          if (stats.total > 0 && emptyChart) {
            emptyChart.style.display = "none";
            if (chartBox) chartBox.style.display = "block";
            if (chartBox2) chartBox2.style.display = "block";
          }
        }
      })
      .catch((err) => {
        console.warn("Live stats sync error:", err);
      });
  }, 5000);
});
