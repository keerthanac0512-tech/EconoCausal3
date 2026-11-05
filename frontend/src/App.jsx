
import { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import "./App.css";

const API_BASE_URL = "http://127.0.0.1:5000";

function App() {
  const [budget, setBudget] = useState(5000);
  const [fileName, setFileName] = useState("");
  const [activePage, setActivePage] = useState("Dashboard");

  const [upliftData, setUpliftData] = useState([]);
  const [dmlData, setDmlData] = useState([]);

  const [comparisonData, setComparisonData] = useState([]);
  const [comparisonLoading, setComparisonLoading] = useState(true);
  const [comparisonError, setComparisonError] = useState("");

  const [loading, setLoading] = useState(true);
  const [dataError, setDataError] = useState("");

  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [uploadError, setUploadError] = useState("");

  const [driftFile, setDriftFile] = useState(null);
  const [driftLoading, setDriftLoading] = useState(false);
  const [driftResult, setDriftResult] = useState(null);
  const [driftError, setDriftError] = useState("");

  const [optimizing, setOptimizing] = useState(false);
  const [budgetResult, setBudgetResult] = useState(null);
  const [budgetError, setBudgetError] = useState("");

  const navigation = [
    "Dashboard",
    "Campaign Data",
    "Budget Optimization",
    "Causal Analysis",
    "Uplift Analysis",
    "Targeting Comparison",
    "Data Drift",
  ];

  // ==================================================
  // LOAD MODEL RESULTS
  // ==================================================

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        setDataError("");

        const [upliftResponse, dmlResponse] = await Promise.all([
          fetch(`${API_BASE_URL}/api/uplift-results`),
          fetch(`${API_BASE_URL}/api/treatment-effects`),
        ]);

        if (!upliftResponse.ok || !dmlResponse.ok) {
          throw new Error(
            "Unable to load model results. Check that Flask is running."
          );
        }

        const [upliftResult, dmlResult] = await Promise.all([
          upliftResponse.json(),
          dmlResponse.json(),
        ]);

        if (
          upliftResult.status !== "success" ||
          dmlResult.status !== "success"
        ) {
          throw new Error("The backend could not return the model results.");
        }

        const parsedUplift = (upliftResult.data || [])
          .map((row) => ({
            customer_rank: Number(row.customer_rank),
            fraction_targeted: Number(row.fraction_targeted),

            cumulative_uplift:
              row.cumulative_uplift == null ||
              row.cumulative_uplift === ""
                ? null
                : Number(row.cumulative_uplift),

            qini_curve:
              row.qini_curve == null || row.qini_curve === ""
                ? null
                : Number(row.qini_curve),

            random_targeting:
              row.random_targeting == null ||
              row.random_targeting === ""
                ? null
                : Number(row.random_targeting),
          }))
          .filter(
            (row) =>
              Number.isFinite(row.customer_rank) &&
              Number.isFinite(row.fraction_targeted)
          );

        const parsedDml = (dmlResult.data || [])
          .map((row) => ({
            ...row,

            estimated_treatment_effect: Number(
              row.estimated_treatment_effect
            ),

            true_treatment_effect:
              row.true_treatment_effect == null
                ? null
                : Number(row.true_treatment_effect),

            treatment:
              row.treatment == null ? null : Number(row.treatment),

            outcome:
              row.outcome == null ? null : Number(row.outcome),
          }))
          .filter((row) =>
            Number.isFinite(row.estimated_treatment_effect)
          );

        if (!parsedUplift.length || !parsedDml.length) {
          throw new Error("The backend returned no usable records.");
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

  // ==================================================
  // LOAD TARGETING COMPARISON
  // ==================================================

  useEffect(() => {
    async function loadComparison() {
      try {
        setComparisonLoading(true);
        setComparisonError("");

        const response = await fetch(
          `${API_BASE_URL}/api/targeting-comparison`
        );

        const result = await response.json();

        if (!response.ok || result.status !== "success") {
          throw new Error(
            result.message || "Unable to load targeting comparison."
          );
        }

        const parsedRows = (result.data || [])
          .map((row) => ({
            strategy: String(row.strategy || ""),
            customers_selected: Number(row.customers_selected),
            estimated_incremental_outcome: Number(
              row.estimated_incremental_outcome
            ),
          }))
          .filter(
            (row) =>
              row.strategy &&
              Number.isFinite(row.customers_selected) &&
              Number.isFinite(row.estimated_incremental_outcome)
          );

        if (!parsedRows.length) {
          throw new Error("The comparison API returned no usable records.");
        }

        setComparisonData(parsedRows);

      } catch (error) {
        setComparisonError(
          `${error.message} Check that Flask is running.`
        );
      } finally {
        setComparisonLoading(false);
      }
    }

    loadComparison();
  }, []);

  // ==================================================
  // CAMPAIGN CSV UPLOAD
  // ==================================================

  async function handleFileUpload(event) {
    const file = event.target.files?.[0];

    setUploadResult(null);
    setUploadError("");

    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".csv")) {
      setUploadError("Please select a CSV file.");
      event.target.value = "";
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setUploading(true);

      const response = await fetch(
        `${API_BASE_URL}/api/upload-campaign`,
        {
          method: "POST",
          body: formData,
        }
      );

      const result = await response.json();

      if (!response.ok || result.status !== "success") {
        throw new Error(result.message || "CSV upload failed.");
      }

      setFileName(file.name);
      setUploadResult(result);

    } catch (error) {
      setUploadError(
        `${error.message} Check that the Flask backend is running.`
      );
    } finally {
      setUploading(false);
    }
  }

  // ==================================================
  // DATA DRIFT DETECTION
  // ==================================================

  async function handleDataDrift() {
    setDriftError("");
    setDriftResult(null);

    if (!driftFile) {
      setDriftError("Please choose a CSV file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", driftFile);

    try {
      setDriftLoading(true);

      const response = await fetch(
        `${API_BASE_URL}/api/data-drift`,
        {
          method: "POST",
          body: formData,
        }
      );

      const result = await response.json();

      if (!response.ok || result.status !== "success") {
        throw new Error(result.message || "Data drift analysis failed.");
      }

      setDriftResult(result);

    } catch (error) {
      setDriftError(
        `${error.message} Check that the Flask backend is running.`
      );
    } finally {
      setDriftLoading(false);
    }
  }

  // ==================================================
  // BUDGET OPTIMIZATION API
  // ==================================================

  async function handleOptimizeBudget() {
    setBudgetError("");
    setBudgetResult(null);

    const budgetValue = Number(budget);
    if (!Number.isFinite(budgetValue) || budgetValue < 0) {
      setBudgetError("Enter a valid non-negative budget.");
      return;
    }

    try {
      setOptimizing(true);

      const response = await fetch(
        `${API_BASE_URL}/api/optimize-budget`,
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
          },

          body: JSON.stringify({
            budget: budgetValue,
          }),
        }
      );

      const result = await response.json();

      if (!response.ok || result.status !== "success") {
        throw new Error(
          result.message || "Budget optimization failed."
        );
      }

      setBudgetResult(result);

    } catch (error) {
      setBudgetError(
        `${error.message} Check that the Flask backend is running.`
      );
    } finally {
      setOptimizing(false);
    }
  }

  // ==================================================
  // DASHBOARD
  // ==================================================

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
            <p>Metrics calculated from saved DML output</p>
          </div>

          <span className="sample-badge">
            {loading ? "LOADING RESULTS" : "LOADED MODEL RESULTS"}
          </span>
        </section>

        {dataError && (
          <div className="info-box">
            <strong>Backend connection error</strong>
            <p>{dataError}</p>
          </div>
        )}

        <section className="metric-grid">
          <article className="metric-card">
            <span>Customers Evaluated</span>
            <h3>{dmlData.length.toLocaleString("en-US")}</h3>
            <p>Records returned by the backend</p>
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
            <p>Model estimate, not guaranteed campaign gain</p>
          </article>

          <article className="metric-card">
            <span>Campaign Budget</span>
            <h3>${Number(budget).toLocaleString("en-US")}</h3>
            <p>Current budget setting</p>
          </article>
        </section>

      </>
    );
  }

  // ==================================================
  // CAMPAIGN DATA
  // ==================================================

  function renderCampaignData() {
    const preview = uploadResult?.data || [];

    return (
      <section className="content-card">
        <h2>Historical Campaign Data</h2>

        <p>
          Upload a CSV file to send it to the Flask backend for validation.
        </p>

        <label className="upload-area">
          <div className="upload-symbol">↑</div>
          <h3>Upload campaign data</h3>
          <p>Choose a CSV file from your computer.</p>

          <span className="upload-button">
            {uploading ? "Uploading..." : "Browse CSV Files"}
          </span>

          <input
            type="file"
            accept=".csv,text/csv"
            onChange={handleFileUpload}
            disabled={uploading}
          />
        </label>

        {fileName && (
          <div className="file-result">
            <strong>Selected file:</strong> {fileName}
          </div>
        )}

        {uploading && (
          <div className="info-box">
            Uploading and validating your CSV...
          </div>
        )}

        {uploadError && (
          <div className="info-box">
            <strong>Upload failed</strong>
            <p>{uploadError}</p>
          </div>
        )}

        {uploadResult && (
          <>
            <div className="info-box">
              <strong>Upload successful</strong>
              <p>{uploadResult.message}</p>
              <p>
                Rows: {uploadResult.rows?.toLocaleString("en-US")}
              </p>
              <p>Columns: {uploadResult.columns?.join(", ")}</p>
            </div>

            {preview.length > 0 && (
              <>
                <h3>CSV Preview</h3>

                <div className="table-wrapper">
                  <table className="results-table">
                    <thead>
                      <tr>
                        {uploadResult.columns.map((column) => (
                          <th key={column}>{column}</th>
                        ))}
                      </tr>
                    </thead>

                    <tbody>
                      {preview.map((row, index) => (
                        <tr key={index}>
                          {uploadResult.columns.map((column) => (
                            <td key={column}>
                              {row[column] == null
                                ? "—"
                                : String(row[column])}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </>
        )}

        <div className="info-box">
          <strong>Important</strong>
          <p>
            The backend validates and previews the uploaded CSV.
            Uploading does not automatically train the DML model.
          </p>
        </div>
      </section>
    );
  }

  // ==================================================
  // DATA DRIFT
  // ==================================================

  function renderDataDrift() {
    const report = driftResult?.results;
    const featureResults = report?.results
      ? Object.entries(report.results)
      : [];

    const statusLabel = report?.overall_status || "Not analysed";

    const formatMetric = (value) =>
      value == null || !Number.isFinite(Number(value))
        ? "—"
        : Number(value).toFixed(4);

    return (
      <section className="content-card">
        <h2>Data Drift Detection</h2>

        <p>
          Upload a new campaign CSV to compare its feature distributions
          against the saved reference dataset.
        </p>

        <label className="upload-area">
          <div className="upload-symbol">↑</div>
          <h3>Choose campaign CSV</h3>

          <p>
            {driftFile
              ? `Selected file: ${driftFile.name}`
              : "Select the latest campaign data to analyse."}
          </p>

          <span className="upload-button">Browse CSV Files</span>

          <input
            type="file"
            accept=".csv,text/csv"
            onChange={(event) => {
              const selectedFile = event.target.files?.[0] || null;
              setDriftFile(selectedFile);
              setDriftResult(null);
              setDriftError("");
            }}
            disabled={driftLoading}
          />
        </label>

        <button
          className="primary-button"
          onClick={handleDataDrift}
          disabled={driftLoading || !driftFile}
        >
          {driftLoading ? "Analysing data..." : "Detect Data Drift"}
        </button>

        {driftError && (
          <div className="info-box">
            <strong>Drift analysis failed</strong>
            <p>{driftError}</p>
          </div>
        )}

        {driftResult && (
          <>
            <div className="info-box">
              <strong>Analysis completed</strong>
              <p>{driftResult.message}</p>
            </div>

            {report && (
              <>
                <section className="metric-grid">
                  <article className="metric-card">
                    <span>Overall Drift Status</span>
                    <h3>{statusLabel}</h3>
                  </article>

                  <article className="metric-card">
                    <span>Total Features</span>
                    <h3>
                      {report.total_features ?? featureResults.length}
                    </h3>
                  </article>

                  <article className="metric-card">
                    <span>Drifted Features</span>
                    <h3>{report.drifted_features ?? 0}</h3>
                  </article>

                  <article className="metric-card">
                    <span>Warning Features</span>
                    <h3>{report.warning_features ?? 0}</h3>
                  </article>
                </section>

                {featureResults.length > 0 && (
                  <>
                    <h3>Feature-wise drift results</h3>

                    <div className="table-wrapper">
                      <table className="results-table">
                        <thead>
                          <tr>
                            <th>Feature</th>
                            <th>Status</th>
                            <th>PSI</th>
                            <th>p-value</th>
                            <th>Reference Mean</th>
                            <th>Current Mean</th>
                          </tr>
                        </thead>

                        <tbody>
                          {featureResults.map(([feature, values]) => (
                            <tr key={feature}>
                              <td>{feature}</td>
                              <td>{values.status || "—"}</td>
                              <td>{formatMetric(values.psi)}</td>
                              <td>{formatMetric(values.p_value)}</td>
                              <td>{formatMetric(values.reference_mean)}</td>
                              <td>{formatMetric(values.current_mean)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                )}
              </>
            )}
          </>
        )}
      </section>
    );
  }

  // ==================================================
  // BUDGET OPTIMIZATION
  // ==================================================

  function renderBudgetOptimization() {
    const summary = budgetResult?.summary;
    const allocation = budgetResult?.allocation_matrix || [];

    return (
      <section className="content-card">
        <h2>Budget and Discount Strategy Analysis</h2>

        <p>
          Compare the estimated response to 10% and 20% discounts
          using customer-level treatment effects from the DML model.
        </p>

        <div className="settings-grid">
          <div className="input-group">
            <label htmlFor="budget">
              Campaign Budget Reference ($)
            </label>

            <input
              id="budget"
              type="number"
              min="0"
              value={budget}
              onChange={(event) => {
                setBudget(event.target.value);
                setBudgetResult(null);
                setBudgetError("");
              }}
            />
          </div>
        </div>

        <div className="info-box">
          <strong>How the recommendation works</strong>
          <p>
            Customers with a positive estimated treatment effect are
            recommended for the 20% discount. Customers with a zero or
            negative effect are recommended to remain at 10%.
          </p>
          <p>
            The budget is shown as a planning reference only. Without
            order value or verified discount-cost data, it cannot
            determine how many customers the campaign can afford to target.
          </p>
        </div>

        <button
          className="primary-button"
          onClick={handleOptimizeBudget}
          disabled={optimizing}
        >
          {optimizing ? "Analysing discounts..." : "Compare Discount Strategies"}
        </button>

        {budgetError && (
          <div className="info-box">
            <strong>Analysis failed</strong>
            <p>{budgetError}</p>
          </div>
        )}

        {summary && (
          <>
            <section className="section-heading">
              <div>
                <h2>Discount Strategy Results</h2>
                <p>Customer recommendations based on estimated ITE</p>
              </div>

              <span className="sample-badge">
                ANALYSIS COMPLETE
              </span>
            </section>

            <section className="metric-grid">
              <article className="metric-card">
                <span>Customers Evaluated</span>
                <h3>
                  {Number(summary.total_records).toLocaleString("en-US")}
                </h3>
                <p>Unique customers with valid DML estimates</p>
              </article>

              <article className="metric-card">
                <span>Recommended for 10%</span>
                <h3>
                  {Number(summary.records_at_10).toLocaleString("en-US")}
                </h3>
                <p>No positive estimated benefit from upgrading</p>
              </article>

              <article className="metric-card">
                <span>Recommended for 20%</span>
                <h3>
                  {Number(summary.records_at_20).toLocaleString("en-US")}
                </h3>
                <p>Positive estimated benefit from upgrading</p>
              </article>

              <article className="metric-card">
                <span>Estimated Incremental Purchases</span>
                <h3>
                  {Number(
                    summary.estimated_incremental_purchases || 0
                  ).toFixed(2)}
                </h3>
                <p>Expected difference versus giving selected customers 10%</p>
              </article>
            </section>

            <div className="info-box">
              <strong>Budget reference</strong>
              <p>
                ${Number(summary.budget_reference || 0).toLocaleString("en-US")}
              </p>
              <p>{budgetResult.budget_note}</p>
            </div>

            <h3>Customers by Recommended Discount</h3>

            <div className="chart-placeholder">
              <Plot
                data={[
                  {
                    x: allocation.map(
                      (row) => `${row.discount}% Discount`
                    ),
                    y: allocation.map(
                      (row) => row.records_allocated
                    ),
                    type: "bar",
                    text: allocation.map(
                      (row) => row.records_allocated
                    ),
                    textposition: "auto",
                    hovertemplate:
                      "%{x}<br>Customers: %{y}<extra></extra>",
                  },
                ]}
                layout={{
                  autosize: true,
                  title: "Customer Recommendations by Discount",
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                  font: { color: "#dce5f5" },
                  margin: { t: 50, r: 20, b: 60, l: 60 },
                  xaxis: { title: "Recommended discount" },
                  yaxis: {
                    title: "Number of customers",
                    gridcolor: "rgba(148,163,184,0.15)",
                  },
                }}
                useResizeHandler
                style={{ width: "100%", height: "350px" }}
                config={{ responsive: true, displaylogo: false }}
              />
            </div>

            <h3>Allocation Summary</h3>

            <div className="table-wrapper">
              <table className="results-table">
                <thead>
                  <tr>
                    <th>Recommended Discount</th>
                    <th>Customers</th>
                    <th>Mean Estimated ITE</th>
                  </tr>
                </thead>

                <tbody>
                  {allocation.map((row) => (
                    <tr key={row.discount}>
                      <td>{row.discount}%</td>
                      <td>
                        {Number(row.records_allocated).toLocaleString("en-US")}
                      </td>
                      <td>
                        {Number(
                          row.mean_estimated_treatment_effect || 0
                        ).toFixed(4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <h3>Recommended Customer Allocation</h3>

            <p>
              Top 20 customers ordered by estimated treatment effect.
            </p>

            <div className="table-wrapper">
              <table className="results-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Customer ID</th>
                    <th>Age</th>
                    <th>Income</th>
                    <th>Estimated ITE</th>
                    <th>Recommended Discount</th>
                    <th>Reason</th>
                  </tr>
                </thead>

                <tbody>
                  {(budgetResult.data || [])
                    .slice(0, 20)
                    .map((row, index) => (
                      <tr key={`${row.person}-${index}`}>
                        <td>{row.effect_rank ?? index + 1}</td>
                        <td>{row.person ?? "—"}</td>
                        <td>{row.age ?? "—"}</td>
                        <td>
                          {row.income == null
                            ? "—"
                            : Number(row.income).toLocaleString("en-US")}
                        </td>
                        <td>
                          {Number(
                            row.estimated_treatment_effect
                          ).toFixed(4)}
                        </td>
                        <td>{row.recommended_discount}%</td>
                        <td>{row.allocation_reason ?? "—"}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>

            <div className="info-box">
              <strong>Interpretation and limitation</strong>
              <p>{budgetResult.effect_note}</p>
              <p>
                The estimated incremental purchase figure is a model-based
                expected outcome, not a guaranteed number of purchases.
                This analysis does not calculate discount expenditure,
                revenue, profit, ROI, or budget utilization.
              </p>
            </div>
          </>
        )}
      </section>
    );
  }

  // ==================================================
  // CAUSAL ANALYSIS
  // ==================================================

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
                  Loaded {dmlData.length.toLocaleString("en-US")} records
                  returned by the backend.
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
                      <td>
                        {row.estimated_treatment_effect.toFixed(4)}
                      </td>
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

  // ==================================================
  // UPLIFT ANALYSIS
  // ==================================================

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
            <p>
              Check the columns in uplift_evaluation_v2.csv and the
              backend response.
            </p>
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
            The charts visualize saved evaluation values. The Qini-style
            evaluation remains preliminary until its calculation is
            validated against an appropriate uplift-evaluation method.
          </p>
        </div>
      </section>
    );
  }

  // ==================================================
  // TARGETING COMPARISON
  // ==================================================

  function renderTargetingComparison() {
    const formatNumber = (value) =>
      Number.isFinite(value) ? value.toFixed(4) : "—";

    return (
      <section className="content-card">
        <h2>Targeting Strategy Comparison</h2>

        <p>
          Compare model-estimated incremental outcomes across blanket,
          causal and random targeting strategies.
        </p>

        {comparisonLoading ? (
          <div className="info-box">
            Loading targeting comparison...
          </div>
        ) : comparisonError ? (
          <div className="info-box">
            <strong>Unable to load comparison</strong>
            <p>{comparisonError}</p>
          </div>
        ) : (
          <>
            <h3>Strategy results</h3>

            <div className="table-wrapper">
              <table className="results-table">
                <thead>
                  <tr>
                    <th>Strategy</th>
                    <th>Customers Selected</th>
                    <th>Estimated Incremental Outcome</th>
                  </tr>
                </thead>

                <tbody>
                  {comparisonData.map((row) => (
                    <tr key={row.strategy}>
                      <td>{row.strategy}</td>
                      <td>
                        {row.customers_selected.toLocaleString("en-US")}
                      </td>
                      <td>
                        {formatNumber(row.estimated_incremental_outcome)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <h3>Estimated incremental outcome by strategy</h3>

            <div className="chart-placeholder">
              <Plot
                data={[
                  {
                    x: comparisonData.map((row) => row.strategy),
                    y: comparisonData.map(
                      (row) => row.estimated_incremental_outcome
                    ),
                    type: "bar",
                    name: "Estimated incremental outcome",
                    hovertemplate:
                      "%{x}<br>Estimated outcome: %{y:.4f}<extra></extra>",
                  },
                ]}
                layout={{
                  autosize: true,
                  title: "Targeting Strategy Outcomes",
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                  font: { color: "#dce5f5" },
                  margin: { t: 50, r: 25, b: 80, l: 65 },
                  xaxis: { title: "Targeting strategy" },
                  yaxis: {
                    title: "Estimated incremental outcome",
                    gridcolor: "rgba(148,163,184,0.15)",
                  },
                }}
                useResizeHandler
                style={{ width: "100%", height: "400px" }}
                config={{ responsive: true, displaylogo: false }}
              />
            </div>

            <div className="info-box">
              <strong>Interpretation</strong>
              <p>
                These are model-based estimated outcomes from the saved
                comparison file. They are not verified purchases, revenue,
                profit, realized savings, or discount expenditure.
              </p>
            </div>
          </>
        )}
      </section>
    );
  }

  // ==================================================
  // PAGE ROUTING
  // ==================================================

  function renderPage() {
    if (activePage === "Dashboard") return renderDashboard();

    if (activePage === "Campaign Data") return renderCampaignData();

    if (activePage === "Budget Optimization") {
      return renderBudgetOptimization();
    }

    if (activePage === "Causal Analysis") {
      return renderCausalAnalysis();
    }

    if (activePage === "Uplift Analysis") {
      return renderUpliftAnalysis();
    }

    if (activePage === "Targeting Comparison") {
      return renderTargetingComparison();
    }

    if (activePage === "Data Drift") {
      return renderDataDrift();
    }

    return renderDashboard();
  }

  // ==================================================
  // MAIN APPLICATION UI
  // ==================================================

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
              className={
                activePage === page ? "nav-item active" : "nav-item"
              }
              onClick={() => setActivePage(page)}
            >
              <span className="nav-icon">
                {["▦", "↑", "◈", "◎", "⌁", "⇄", "⌁"][index]}
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
              <div className="eyebrow">
                ECONOMICS × MACHINE LEARNING
              </div>

              <h1>{activePage}</h1>

              <p>
                Optimize promotional strategies using causal machine
                learning.
              </p>
            </div>

            <div className="model-badge">DML WORKSPACE</div>
          </section>

          {renderPage()}

          <footer className="footer">
            <span>EconoCausal</span>
            <span>
              Dynamic Pricing via Double Machine Learning
            </span>
          </footer>
        </div>
      </main>
    </div>
  );
}

export default App;
