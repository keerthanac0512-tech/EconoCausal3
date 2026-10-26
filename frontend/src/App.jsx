import { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import Papa from "papaparse";
import "./App.css";

function App() {
  const [budget, setBudget] = useState(5000);
  const [discount, setDiscount] = useState(20);
  const [fileName, setFileName] = useState("");
  const [activePage, setActivePage] = useState("Dashboard");

  const [upliftData, setUpliftData] = useState([]);
  const [dmlData, setDmlData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [dataError, setDataError] = useState("");

  const navigation = [
    "Dashboard",
    "Campaign Data",
    "Budget Optimization",
    "Causal Analysis",
    "Uplift Analysis",
  ];

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        setDataError("");

        const [upliftResponse, dmlResponse] = await Promise.all([
          fetch("/uplift_evaluation_v2.csv"),
          fetch("/dml_treatment_effects_v2.csv"),
        ]);

        if (!upliftResponse.ok || !dmlResponse.ok) {
          throw new Error(
            "Could not load the CSV files. Check that both files are in the public folder."
          );
        }

        const [upliftText, dmlText] = await Promise.all([
          upliftResponse.text(),
          dmlResponse.text(),
        ]);

        const parseCSV = (text) =>
          Papa.parse(text, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
          }).data;

        const parsedUplift = parseCSV(upliftText)
          .map((row) => ({
            customer_rank: Number(row.customer_rank),
            fraction_targeted: Number(row.fraction_targeted),
            cumulative_uplift:
              row.cumulative_uplift === "" ||
              row.cumulative_uplift == null
                ? null
                : Number(row.cumulative_uplift),
            qini_curve:
              row.qini_curve === "" || row.qini_curve == null
                ? null
                : Number(row.qini_curve),
            random_targeting:
              row.random_targeting === "" || row.random_targeting == null
                ? null
                : Number(row.random_targeting),
          }))
          .filter(
            (row) =>
              Number.isFinite(row.fraction_targeted) &&
              Number.isFinite(row.customer_rank)
          );

        const parsedDml = parseCSV(dmlText)
          .map((row) => ({
            ...row,
            estimated_treatment_effect: Number(
              row.estimated_treatment_effect
            ),
            true_treatment_effect: Number(row.true_treatment_effect),
            treatment: Number(row.treatment),
            outcome: Number(row.outcome),
          }))
          .filter((row) =>
            Number.isFinite(row.estimated_treatment_effect)
          );

        if (!parsedUplift.length || !parsedDml.length) {
          throw new Error("The CSV files loaded, but no usable rows were found.");
        }

        setUpliftData(parsedUplift);
        setDmlData(parsedDml);
      } catch (error) {
        setDataError(error.message);
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, []);

  function handleFileUpload(event) {
    const file = event.target.files[0];

    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".csv")) {
      alert("Please select a CSV file.");
      event.target.value = "";
      return;
    }

    setFileName(file.name);
  }

  function renderDashboard() {
    const positiveEffects = dmlData.filter(
      (row) => row.estimated_treatment_effect > 0
    ).length;

    const meanEffect = dmlData.length
      ? dmlData.reduce(
          (sum, row) => sum + row.estimated_treatment_effect,
          0
        ) / dmlData.length
      : null;

    return (
      <>
        <section className="hero-banner">
          <div className="hero-text">
            <span className="hero-tag">CAUSAL INTELLIGENCE</span>
            <h2>
              Smarter discounts.
              <br />
              <span>Measurable impact.</span>
            </h2>
            <p>
              Identify persuadable customers and allocate promotional
              budgets using individual treatment effects.
            </p>
            <button
              className="primary-button"
              onClick={() => setActivePage("Campaign Data")}
            >
              Explore Campaign Data →
            </button>
          </div>

          <div className="hero-graphic">
            <div className="orbit orbit-one"></div>
            <div className="orbit orbit-two"></div>
            <div className="orbit orbit-three"></div>
            <div className="orbit-center">DML</div>
            <div className="orbit-node node-one">ITE</div>
            <div className="orbit-node node-two">AI</div>
            <div className="orbit-node node-three">ROI</div>
          </div>
        </section>

        <section className="section-heading">
          <div>
            <h2>Model Overview</h2>
            <p>Metrics calculated from your saved DML output</p>
          </div>
          <span className="sample-badge">
            {dmlData.length ? "LOADED MODEL RESULTS" : "LOADING RESULTS"}
          </span>
        </section>

        <section className="metric-grid">
          <article className="metric-card">
            <span>Customers Evaluated</span>
            <h3>{dmlData.length.toLocaleString("en-US")}</h3>
            <p>Rows in the DML evaluation file</p>
          </article>

          <article className="metric-card">
            <span>Positive Estimated ITE</span>
            <h3>{positiveEffects.toLocaleString("en-US")}</h3>
            <p>Customers with positive estimated effects</p>
          </article>

          <article className="metric-card">
            <span>Mean Estimated Treatment Effect</span>
            <h3>
              {meanEffect === null ? "—" : meanEffect.toFixed(4)}
            </h3>
            <p>Mean model estimate, not guaranteed campaign gain</p>
          </article>

          <article className="metric-card">
            <span>Campaign Budget</span>
            <h3>${budget.toLocaleString("en-US")}</h3>
            <p>Current frontend setting</p>
          </article>
        </section>

        <section className="section-heading">
          <div>
            <h2>Project Modules</h2>
            <p>Explore your causal pricing workflow</p>
          </div>
        </section>

        <section className="module-grid">
          {[
            {
              title: "Campaign Data",
              description: "Select historical campaign CSV data.",
              icon: "↑",
            },
            {
              title: "Budget Optimization",
              description: "Configure budgets and discount offers.",
              icon: "$",
            },
            {
              title: "Causal Analysis",
              description: "Explore estimated customer treatment effects.",
              icon: "◎",
            },
            {
              title: "Uplift Analysis",
              description: "Inspect the saved uplift and Qini evaluation.",
              icon: "⌁",
            },
          ].map((module) => (
            <button
              className="module-card"
              key={module.title}
              onClick={() => setActivePage(module.title)}
            >
              <div className="module-icon">{module.icon}</div>
              <h3>{module.title}</h3>
              <p>{module.description}</p>
              <span className="module-arrow">Open module →</span>
            </button>
          ))}
        </section>
      </>
    );
  }

  function renderCampaignData() {
    return (
      <section className="content-card">
        <h2>Historical Campaign Data</h2>
        <p>Select a CSV file to prepare it for analysis.</p>

        <label className="upload-area">
          <div className="upload-symbol">↑</div>
          <h3>Upload campaign data</h3>
          <p>Choose a CSV file from your computer.</p>
          <span className="upload-button">Browse CSV Files</span>
          <input
            type="file"
            accept=".csv,text/csv"
            onChange={handleFileUpload}
          />
        </label>

        {fileName && (
          <div className="file-result">
            <strong>Selected file:</strong> {fileName}
          </div>
        )}

        <div className="info-box">
          <strong>Note</strong>
          <p>
            Selecting a file currently updates the interface only. It
            does not retrain the model or send data to Python.
          </p>
        </div>
      </section>
    );
  }

  function renderBudgetOptimization() {
    return (
      <section className="content-card">
        <h2>Budget Optimization</h2>
        <p>Configure your promotional campaign constraints.</p>

        <div className="settings-grid">
          <div className="input-group">
            <label htmlFor="budget">Total campaign budget ($)</label>
            <input
              id="budget"
              type="number"
              min="0"
              value={budget}
              onChange={(event) =>
                setBudget(Math.max(0, Number(event.target.value)))
              }
            />
          </div>

          <div className="input-group">
            <label htmlFor="discount">Discount offer</label>
            <select
              id="discount"
              value={discount}
              onChange={(event) => setDiscount(Number(event.target.value))}
            >
              <option value={10}>$10 Discount</option>
              <option value={20}>$20 Discount</option>
            </select>
          </div>
        </div>

        <div className="budget-preview">
          <div>
            <span>Available budget</span>
            <strong>${budget.toLocaleString("en-US")}</strong>
          </div>
          <div>
            <span>Selected discount</span>
            <strong>${discount}</strong>
          </div>
        </div>

        <div className="info-box">
          <strong>Frontend preview</strong>
          <p>
            These controls update the interface only. The Python budget
            optimizer is not connected yet.
          </p>
        </div>
      </section>
    );
  }

  function renderCausalAnalysis() {
    const topCustomers = [...dmlData]
      .sort(
        (a, b) =>
          b.estimated_treatment_effect - a.estimated_treatment_effect
      )
      .slice(0, 10);

    return (
      <section className="content-card">
        <h2>Causal Analysis</h2>
        <p>
          Explore individual treatment effects estimated using EconML
          Double Machine Learning.
        </p>

        {loading ? (
          <p>Loading model results...</p>
        ) : dataError ? (
          <div className="info-box">
            <strong>Unable to load results</strong>
            <p>{dataError}</p>
          </div>
        ) : (
          <>
            <div className="analysis-callout">
              <div className="analysis-icon">◎</div>
              <div>
                <h3>Linear Double Machine Learning</h3>
                <p>
                  Loaded {dmlData.length.toLocaleString("en-US")} evaluation
                  records from the saved model output.
                </p>
              </div>
            </div>

            <h3>Top 10 estimated treatment effects</h3>
            <div className="table-wrapper">
              <table className="results-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Age</th>
                    <th>Income</th>
                    <th>Treatment</th>
                    <th>Outcome</th>
                    <th>Estimated ITE</th>
                  </tr>
                </thead>
                <tbody>
                  {topCustomers.map((row, index) => (
                    <tr key={`${index}-${row.estimated_treatment_effect}`}>
                      <td>{index + 1}</td>
                      <td>{row.age ?? "—"}</td>
                      <td>
                        {Number.isFinite(Number(row.income))
                          ? Number(row.income).toLocaleString("en-US")
                          : "—"}
                      </td>
                      <td>{row.treatment ?? "—"}</td>
                      <td>{row.outcome ?? "—"}</td>
                      <td>{row.estimated_treatment_effect.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="info-box">
              <strong>Interpretation</strong>
              <p>
                These are model-estimated individual treatment effects.
                They should not be interpreted as guaranteed purchases
                or revenue.
              </p>
            </div>
          </>
        )}
      </section>
    );
  }

  function renderUpliftAnalysis() {
    if (loading) {
      return (
        <section className="content-card">
          <h2>Uplift and Qini Analysis</h2>
          <p>Loading evaluation data...</p>
        </section>
      );
    }

    if (dataError) {
      return (
        <section className="content-card">
          <h2>Uplift and Qini Analysis</h2>
          <div className="info-box">
            <strong>Unable to load charts</strong>
            <p>{dataError}</p>
          </div>
        </section>
      );
    }

    const validQiniRows = upliftData.filter(
      (row) =>
        Number.isFinite(row.qini_curve) &&
        Number.isFinite(row.random_targeting)
    );

    const validUpliftRows = upliftData.filter(
      (row) => Number.isFinite(row.cumulative_uplift)
    );

    const plotLayout = {
      autosize: true,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#dce5f5" },
      margin: { t: 45, r: 25, b: 60, l: 65 },
      legend: { orientation: "h", y: -0.25 },
      xaxis: {
        title: "Fraction of customers targeted",
        gridcolor: "rgba(148,163,184,0.15)",
        tickformat: ".0%",
      },
      yaxis: {
        title: "Cumulative Qini score",
        gridcolor: "rgba(148,163,184,0.15)",
      },
    };

    return (
      <section className="content-card">
        <h2>Uplift and Qini Analysis</h2>
        <p>
          Inspect model-ranked targeting using the saved evaluation output.
        </p>

        {validQiniRows.length > 0 ? (
          <div className="chart-placeholder">
            <Plot
              data={[
                {
                  x: validQiniRows.map((row) => row.fraction_targeted),
                  y: validQiniRows.map((row) => row.qini_curve),
                  type: "scatter",
                  mode: "lines",
                  name: "Model Qini curve",
                  connectgaps: false,
                },
                {
                  x: validQiniRows.map((row) => row.fraction_targeted),
                  y: validQiniRows.map((row) => row.random_targeting),
                  type: "scatter",
                  mode: "lines",
                  name: "Random targeting",
                  connectgaps: false,
                },
              ]}
              layout={{
                ...plotLayout,
                title: "Qini Curve",
              }}
              useResizeHandler
              style={{ width: "100%", height: "430px" }}
              config={{ responsive: true, displaylogo: false }}
            />
          </div>
        ) : (
          <div className="info-box">
            <strong>No valid Qini points found</strong>
            <p>Check the numeric values in uplift_evaluation_v2.csv.</p>
          </div>
        )}

        {validUpliftRows.length > 0 && (
          <>
            <h3>Cumulative uplift by targeting fraction</h3>
            <div className="chart-placeholder">
              <Plot
                data={[
                  {
                    x: validUpliftRows.map((row) => row.fraction_targeted),
                    y: validUpliftRows.map((row) => row.cumulative_uplift),
                    type: "scatter",
                    mode: "lines",
                    name: "Cumulative uplift",
                    connectgaps: false,
                  },
                ]}
                layout={{
                  ...plotLayout,
                  title: "Cumulative Uplift",
                  yaxis: {
                    title: "Cumulative uplift",
                    gridcolor: "rgba(148,163,184,0.15)",
                  },
                }}
                useResizeHandler
                style={{ width: "100%", height: "430px" }}
                config={{ responsive: true, displaylogo: false }}
              />
            </div>
          </>
        )}

        <div className="info-box">
          <strong>Evaluation note</strong>
          <p>
            The charts visualize the saved CSV values. The current Qini-style
            evaluation should be treated as preliminary until its calculation
            is validated against an appropriate uplift-evaluation method.
            Missing cumulative-uplift values are omitted rather than replaced
            with zero.
          </p>
        </div>
      </section>
    );
  }

  function renderPage() {
    if (activePage === "Dashboard") return renderDashboard();
    if (activePage === "Campaign Data") return renderCampaignData();
    if (activePage === "Budget Optimization") return renderBudgetOptimization();
    if (activePage === "Causal Analysis") return renderCausalAnalysis();
    if (activePage === "Uplift Analysis") return renderUpliftAnalysis();
    return renderDashboard();
  }

  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">E</div>
          <div>
            <h2>EconoCausal</h2>
            <p>CAUSAL AI PLATFORM</p>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>

        <nav className="navigation">
          {navigation.map((page, index) => (
            <button
              key={page}
              className={activePage === page ? "nav-item active" : "nav-item"}
              onClick={() => setActivePage(page)}
            >
              <span className="nav-icon">
                {["▦", "↑", "◈", "◎", "⌁"][index]}
              </span>
              {page}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="status-dot"></div>
          <div>
            <strong>Model Workspace</strong>
            <span>Linear Double ML</span>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace / <strong>{activePage}</strong>
          </div>
          <div className="profile">
            <span className="online-dot"></span>
            <span>Local Environment</span>
            <div className="avatar">EC</div>
          </div>
        </header>

        <div className="page-content">
          <section className="page-heading">
            <div>
              <div className="eyebrow">ECONOMICS × MACHINE LEARNING</div>
              <h1>{activePage}</h1>
              <p>
                Optimize promotional strategies using causal machine learning.
              </p>
            </div>
            <div className="model-badge">DML WORKSPACE</div>
          </section>

          {renderPage()}

          <footer className="footer">
            <span>EconoCausal</span>
            <span>Dynamic Pricing via Double Machine Learning</span>
          </footer>
        </div>
      </main>
    </div>
  );
}

export default App;
