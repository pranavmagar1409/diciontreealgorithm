"""
LoanLens - Loan Approval Predictor
Flask app serving a scikit-learn DecisionTreeClassifier (decision.pkl).

Local run :  python app.py
Render    :  gunicorn app:app
"""
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "decision.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = list(model.feature_names_in_)

# Label encodings (alphabetical order, as produced by sklearn LabelEncoder)
GENDER = {"female": 0, "male": 1}
EDUCATION = {"Associate": 0, "Bachelor": 1, "Doctorate": 2, "High School": 3, "Master": 4}
HOME = {"MORTGAGE": 0, "OTHER": 1, "OWN": 2, "RENT": 3}
INTENT = {"DEBTCONSOLIDATION": 0, "EDUCATION": 1, "HOMEIMPROVEMENT": 2,
          "MEDICAL": 3, "PERSONAL": 4, "VENTURE": 5}
DEFAULTS = {"No": 0, "Yes": 1}

LABELS = {
    "person_age": "Age",
    "person_gender": "Gender",
    "person_education": "Education",
    "person_income": "Annual income",
    "person_emp_exp": "Work experience",
    "person_home_ownership": "Home ownership",
    "loan_amnt": "Loan amount",
    "loan_intent": "Loan purpose",
    "loan_int_rate": "Interest rate",
    "loan_percent_income": "Loan-to-income ratio",
    "cb_person_cred_hist_length": "Credit history length",
    "credit_score": "Credit score",
    "previous_loan_defaults_on_file": "Previous defaults",
}

app = Flask(__name__)


def build_row(d):
    """Validate the request payload and build a one-row DataFrame in model order."""
    income = float(d["person_income"])
    loan = float(d["loan_amnt"])
    if income <= 0 or loan <= 0:
        raise ValueError("Income and loan amount must be greater than zero.")
    age = float(d["person_age"])
    exp = float(d["person_emp_exp"])
    if not 18 <= age <= 100:
        raise ValueError("Age must be between 18 and 100.")
    if exp < 0 or exp > age - 14:
        raise ValueError("Work experience looks inconsistent with age.")
    row = {
        "person_age": age,
        "person_gender": GENDER[d["person_gender"]],
        "person_education": EDUCATION[d["person_education"]],
        "person_income": income,
        "person_emp_exp": exp,
        "person_home_ownership": HOME[d["person_home_ownership"]],
        "loan_amnt": loan,
        "loan_intent": INTENT[d["loan_intent"]],
        "loan_int_rate": float(d["loan_int_rate"]),
        "loan_percent_income": round(loan / income, 4),
        "cb_person_cred_hist_length": float(d["cb_person_cred_hist_length"]),
        "credit_score": float(d["credit_score"]),
        "previous_loan_defaults_on_file": DEFAULTS[d["previous_loan_defaults_on_file"]],
    }
    return pd.DataFrame([row], columns=FEATURES), row


def key_factors(X, row, top=5):
    """Features on the tree's decision path for this applicant, ranked by importance."""
    nodes = model.decision_path(X).indices
    tree = model.tree_
    used = {int(tree.feature[n]) for n in nodes if tree.children_left[n] != tree.children_right[n]}
    ranked = sorted(used, key=lambda i: model.feature_importances_[i], reverse=True)[:top]
    out = []
    for i in ranked:
        name = FEATURES[i]
        out.append({"label": LABELS[name], "key": name, "value": row[name],
                    "importance": round(float(model.feature_importances_[i]) * 100, 1)})
    return out


@app.route("/")
def home():
    return PAGE


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        X, row = build_row(request.get_json(force=True))
        pred = int(model.predict(X)[0])
        return jsonify(
            approved=bool(pred == 1),
            loan_percent_income=row["loan_percent_income"],
            factors=key_factors(X, row),
        )
    except (KeyError, ValueError, TypeError) as e:
        return jsonify(error=f"Invalid input: {e}"), 400


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LoanLens | Instant Loan Approval Check</title>
<style>
:root{
  --bg:#0b1020; --card:rgba(255,255,255,.06); --border:rgba(255,255,255,.12);
  --text:#e9ecf6; --muted:#9aa3bd; --accent:#6c8cff; --accent2:#9b6cff;
  --ok:#2ee59d; --bad:#ff5d7a;
}
*{box-sizing:border-box;margin:0;padding:0}
body{
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  color:var(--text);min-height:100vh;
  background:
    radial-gradient(900px 500px at 10% -10%,rgba(108,140,255,.28),transparent 60%),
    radial-gradient(800px 500px at 100% 0%,rgba(155,108,255,.22),transparent 60%),
    var(--bg);
}
.wrap{max-width:1080px;margin:0 auto;padding:28px 20px 60px}
nav{display:flex;align-items:center;justify-content:space-between;margin-bottom:34px}
.logo{display:flex;align-items:center;gap:10px;font-weight:800;font-size:1.25rem;letter-spacing:.3px}
.logo i{width:34px;height:34px;border-radius:10px;display:grid;place-items:center;font-style:normal;
  background:linear-gradient(135deg,var(--accent),var(--accent2))}
.pill{font-size:.78rem;color:var(--muted);border:1px solid var(--border);padding:6px 12px;border-radius:99px}
.hero{text-align:center;margin-bottom:34px}
.hero h1{font-size:clamp(1.9rem,4.5vw,3rem);line-height:1.15;font-weight:800}
.hero h1 span{background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;background-clip:text;color:transparent}
.hero p{color:var(--muted);margin-top:12px;font-size:1.05rem}
.grid{display:grid;grid-template-columns:1.5fr 1fr;gap:22px;align-items:start}
@media(max-width:860px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--border);border-radius:20px;padding:24px;
  backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px)}
.card h2{font-size:.82rem;text-transform:uppercase;letter-spacing:1.4px;color:var(--accent);margin:4px 0 14px}
.card h2:not(:first-child){margin-top:26px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:520px){.row{grid-template-columns:1fr}}
label{display:block;font-size:.82rem;color:var(--muted);margin-bottom:6px}
input,select{width:100%;padding:12px 14px;border-radius:12px;border:1px solid var(--border);
  background:rgba(255,255,255,.05);color:var(--text);font-size:.95rem;outline:none;transition:.2s}
select option{color:#111}
input:focus,select:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(108,140,255,.25)}
.range-head{display:flex;justify-content:space-between;align-items:baseline}
.range-head b{color:var(--text);font-size:.95rem}
input[type=range]{padding:0;height:6px;accent-color:var(--accent);margin-top:6px}
.full{grid-column:1/-1}
button{width:100%;margin-top:26px;padding:15px;border:0;border-radius:14px;cursor:pointer;
  font-size:1rem;font-weight:700;color:#fff;
  background:linear-gradient(135deg,var(--accent),var(--accent2));transition:transform .15s,box-shadow .2s;
  box-shadow:0 10px 30px rgba(108,140,255,.35)}
button:hover{transform:translateY(-2px)}
button:disabled{opacity:.6;cursor:wait}
.side{position:sticky;top:20px}
.idle{text-align:center;color:var(--muted);padding:40px 10px}
.idle .big{font-size:3rem;margin-bottom:10px}
.result{display:none;text-align:center;animation:pop .45s ease}
@keyframes pop{from{opacity:0;transform:scale(.94)}to{opacity:1;transform:none}}
.badge{width:96px;height:96px;margin:6px auto 14px;border-radius:50%;display:grid;place-items:center;font-size:2.6rem}
.result.ok .badge{background:rgba(46,229,157,.15);box-shadow:0 0 0 8px rgba(46,229,157,.08),0 0 40px rgba(46,229,157,.35)}
.result.bad .badge{background:rgba(255,93,122,.15);box-shadow:0 0 0 8px rgba(255,93,122,.08),0 0 40px rgba(255,93,122,.35)}
.verdict{font-size:1.7rem;font-weight:800}
.result.ok .verdict{color:var(--ok)} .result.bad .verdict{color:var(--bad)}
.sub{color:var(--muted);font-size:.92rem;margin:6px 0 18px}
.ratio{display:flex;justify-content:space-between;background:rgba(255,255,255,.05);border:1px solid var(--border);
  padding:10px 14px;border-radius:12px;font-size:.88rem;margin-bottom:18px}
.factors{text-align:left}
.factors h3{font-size:.78rem;text-transform:uppercase;letter-spacing:1.2px;color:var(--muted);margin-bottom:10px}
.f{margin-bottom:11px}
.f .top{display:flex;justify-content:space-between;font-size:.88rem;margin-bottom:5px}
.f .top span:last-child{color:var(--muted)}
.bar{height:6px;border-radius:9px;background:rgba(255,255,255,.08);overflow:hidden}
.bar div{height:100%;border-radius:9px;background:linear-gradient(90deg,var(--accent),var(--accent2));width:0;transition:width .8s ease}
.err{display:none;margin-top:14px;padding:12px;border-radius:12px;background:rgba(255,93,122,.12);color:var(--bad);font-size:.88rem}
.note{margin-top:22px;text-align:center;color:var(--muted);font-size:.78rem;line-height:1.5}
</style>
</head>
<body>
<div class="wrap">
  <nav>
    <div class="logo"><i>◎</i>LoanLens</div>
    <div class="pill">Decision Tree · ML powered</div>
  </nav>

  <section class="hero">
    <h1>Know your loan outcome <span>in seconds</span></h1>
    <p>Enter a few details and our machine-learning model predicts whether the loan is likely to be approved.</p>
  </section>

  <div class="grid">
    <form class="card" id="f" autocomplete="off">
      <h2>Applicant</h2>
      <div class="row">
        <div><label>Age</label><input type="number" name="person_age" value="30" min="18" max="100" required></div>
        <div><label>Gender</label>
          <select name="person_gender"><option value="male">Male</option><option value="female">Female</option></select></div>
        <div><label>Education</label>
          <select name="person_education" >
            <option>High School</option><option>Associate</option><option selected>Bachelor</option>
            <option>Master</option><option>Doctorate</option></select></div>
        <div><label>Work experience (years)</label><input type="number" name="person_emp_exp" value="5" min="0" max="60" required></div>
        <div><label>Annual income (USD)</label><input type="number" name="person_income" value="60000" min="1000" required></div>
        <div><label>Home ownership</label>
          <select name="person_home_ownership">
            <option>RENT</option><option>MORTGAGE</option><option>OWN</option><option>OTHER</option></select></div>
      </div>

      <h2>Loan</h2>
      <div class="row">
        <div><label>Loan amount (USD)</label><input type="number" name="loan_amnt" value="10000" min="500" required></div>
        <div><label>Loan purpose</label>
          <select name="loan_intent">
            <option value="PERSONAL">Personal</option><option value="EDUCATION">Education</option>
            <option value="MEDICAL">Medical</option><option value="VENTURE">Venture / Business</option>
            <option value="HOMEIMPROVEMENT">Home improvement</option><option value="DEBTCONSOLIDATION">Debt consolidation</option></select></div>
        <div class="full">
          <div class="range-head"><label>Interest rate</label><b id="rv">11.0%</b></div>
          <input type="range" name="loan_int_rate" min="5" max="20" step="0.1" value="11"></div>
      </div>

      <h2>Credit profile</h2>
      <div class="row">
        <div class="full">
          <div class="range-head"><label>Credit score</label><b id="cv">650</b></div>
          <input type="range" name="credit_score" min="390" max="850" step="1" value="650"></div>
        <div><label>Credit history (years)</label><input type="number" name="cb_person_cred_hist_length" value="5" min="0" max="50" required></div>
        <div><label>Previous loan defaults?</label>
          <select name="previous_loan_defaults_on_file"><option>No</option><option>Yes</option></select></div>
      </div>

      <button type="submit" id="btn">Check my approval</button>
      <div class="err" id="err"></div>
    </form>

    <aside class="side card">
      <div class="idle" id="idle"><div class="big">🏦</div>Your result will appear here.<br>Fill in the form and tap <b>Check my approval</b>.</div>
      <div class="result" id="res">
        <div class="badge" id="badge"></div>
        <div class="verdict" id="verdict"></div>
        <div class="sub" id="sub"></div>
        <div class="ratio"><span>Loan-to-income ratio</span><b id="ratio"></b></div>
        <div class="factors"><h3>Key factors considered</h3><div id="fl"></div></div>
      </div>
      <p class="note">This is an automated estimate for educational purposes and not a financial offer or guarantee.</p>
    </aside>
  </div>
</div>

<script>
const f = document.getElementById('f');
const $ = id => document.getElementById(id);
f.loan_int_rate.oninput = e => $('rv').textContent = Number(e.target.value).toFixed(1) + '%';
f.credit_score.oninput  = e => $('cv').textContent = e.target.value;

const fmt = (k, v) => {
  if (['person_income','loan_amnt'].includes(k)) return '$' + Number(v).toLocaleString();
  if (k === 'loan_percent_income') return (v * 100).toFixed(1) + '%';
  if (k === 'loan_int_rate') return v + '%';
  if (k === 'person_gender') return v ? 'Male' : 'Female';
  if (k === 'previous_loan_defaults_on_file') return v ? 'Yes' : 'No';
  if (['person_education','person_home_ownership','loan_intent'].includes(k)) return '';
  return v;
};
const names = {
  person_education: ['Associate','Bachelor','Doctorate','High School','Master'],
  person_home_ownership: ['Mortgage','Other','Own','Rent'],
  loan_intent: ['Debt consolidation','Education','Home improvement','Medical','Personal','Venture']
};

f.addEventListener('submit', async e => {
  e.preventDefault();
  const btn = $('btn'), err = $('err');
  err.style.display = 'none'; btn.disabled = true; btn.textContent = 'Analyzing…';
  const data = Object.fromEntries(new FormData(f).entries());
  try {
    const r = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(data)});
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || 'Something went wrong');
    show(j);
  } catch (ex) {
    err.textContent = ex.message; err.style.display = 'block';
  } finally {
    btn.disabled = false; btn.textContent = 'Check my approval';
  }
});

function show(j) {
  $('idle').style.display = 'none';
  const res = $('res');
  res.className = 'result ' + (j.approved ? 'ok' : 'bad');
  res.style.display = 'block';
  $('badge').textContent = j.approved ? '✓' : '✕';
  $('verdict').textContent = j.approved ? 'Likely Approved' : 'Likely Rejected';
  $('sub').textContent = j.approved
    ? 'Your profile matches applicants who were typically approved.'
    : 'Your profile matches applicants who were typically declined. Try a lower loan amount or improving your credit score.';
  $('ratio').textContent = (j.loan_percent_income * 100).toFixed(1) + '%';
  const max = Math.max(...j.factors.map(x => x.importance), 1);
  $('fl').innerHTML = j.factors.map(x => {
    let v = fmt(x.key, x.value);
    if (names[x.key]) v = names[x.key][x.value];
    return `<div class="f"><div class="top"><span>${x.label}</span><span>${v}</span></div>
            <div class="bar"><div data-w="${(x.importance / max) * 100}"></div></div></div>`;
  }).join('');
  requestAnimationFrame(() => document.querySelectorAll('.bar div').forEach(b => b.style.width = b.dataset.w + '%'));
  if (innerWidth < 860) res.scrollIntoView({behavior:'smooth', block:'start'});
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
