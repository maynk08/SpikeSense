// Capture application screenshots for the project report using Puppeteer
// (bundled with mermaid-cli). Requires the API (8000) and dashboard (8501) running.
//
//   node report/scripts/screenshots.mjs
//
import puppeteer from "puppeteer";
import { setTimeout as sleep } from "node:timers/promises";

const OUT = "report/figures";
const DASH = "http://localhost:8501";
const API = "http://localhost:8000";

async function clickByText(page, selector, text) {
  const handle = await page.evaluateHandle(
    (sel, txt) => {
      const els = Array.from(document.querySelectorAll(sel));
      return els.find((e) => e.textContent.trim().includes(txt)) || null;
    },
    selector,
    text,
  );
  const el = handle.asElement();
  if (el) {
    await el.click();
    return true;
  }
  return false;
}

async function waitForChart(page, ms = 90000) {
  // Streamlit scores the full series via the API (slow, per-window) and shows a
  // "Scoring…" spinner. Wait until the spinner is gone AND the Plotly chart exists.
  const deadline = Date.now() + ms;
  while (Date.now() < deadline) {
    const state = await page.evaluate(() => {
      const scoring = document.body.innerText.includes("Scoring time-series");
      const chart = !!document.querySelector(".js-plotly-plot");
      return { scoring, chart };
    });
    if (!state.scoring && state.chart) break;
    await sleep(1500);
  }
  await sleep(3000);
}

const browser = await puppeteer.launch({
  headless: "new",
  args: ["--no-sandbox", "--window-size=1680,1200"],
  defaultViewport: { width: 1680, height: 1150 },
});

try {
  const page = await browser.newPage();

  // 1. Dashboard main view (auto-scores on load -> shows real anomalies)
  console.log("dashboard main...");
  await page.goto(DASH, { waitUntil: "networkidle2", timeout: 60000 });
  await waitForChart(page);
  await page.screenshot({ path: `${OUT}/shot_dashboard_main.png`, fullPage: true });

  // 2. Inject a spike, then capture the detection
  console.log("spike injection...");
  const clicked = await clickByText(page, "button", "Inject Spike");
  if (clicked) {
    await sleep(7000);
    await waitForChart(page);
    await page.screenshot({ path: `${OUT}/shot_dashboard_inject.png`, fullPage: true });
  } else {
    console.log("  inject button not found");
  }

  // 3. Evaluation results panel (expand it)
  console.log("evaluation panel...");
  await clickByText(page, "summary, [data-testid='stExpander'] *", "Model Evaluation Results");
  await sleep(2500);
  await page.screenshot({ path: `${OUT}/shot_dashboard_eval.png`, fullPage: true });

  // 4. Swagger UI overview
  console.log("swagger...");
  await page.goto(`${API}/docs`, { waitUntil: "networkidle2", timeout: 60000 });
  await page.waitForSelector(".swagger-ui", { timeout: 15000 });
  await sleep(2500);
  await page.screenshot({ path: `${OUT}/shot_swagger.png`, fullPage: true });

  // 5. /alerts JSON response in browser
  console.log("alerts json...");
  await page.goto(`${API}/alerts?limit=5`, { waitUntil: "networkidle2", timeout: 30000 });
  await sleep(800);
  await page.screenshot({ path: `${OUT}/shot_api_alerts.png`, fullPage: false });

  // 6. /stats JSON response
  console.log("stats json...");
  await page.goto(`${API}/stats`, { waitUntil: "networkidle2", timeout: 30000 });
  await sleep(600);
  await page.screenshot({ path: `${OUT}/shot_api_stats.png`, fullPage: false });

  console.log("screenshots done");
} finally {
  await browser.close();
}
