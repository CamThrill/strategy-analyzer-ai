from flask import Blueprint, render_template, request, jsonify, session
import os, io, uuid
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from werkzeug.utils import secure_filename

bp = Blueprint("main", __name__)
ALLOWED_EXT = {".xlsx"}

# ---- In-memory per-session store ----
# STORE[sid] = {"strategies": {name: DataFrame}}
STORE = {}

def _sid():
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    return session["sid"]

# ---------- helpers ----------
def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    required = {"entry time", "exit time", "profit", "market pos."}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["entry time"] = pd.to_datetime(df["entry time"])
    df["exit time"]  = pd.to_datetime(df["exit time"])
    df["trade date"] = pd.to_datetime(df["entry time"].dt.date)

    df["profit"] = (df["profit"].astype(str)
                    .str.replace(r'[\$,]', '', regex=True)
                    .str.replace(r'\((.*?)\)', r'-\1', regex=True)
                    .astype(float))
    df["action"] = df["market pos."].str.lower().map({"long": "buy", "short": "sell"})
    return df.sort_values("trade date").reset_index(drop=True)

def _apply_trade_filter(df: pd.DataFrame, trade_filter: str) -> pd.DataFrame:
    """trade_filter in {'all','long','short'}"""
    if trade_filter == "long":
        return df[df["action"] == "buy"]
    if trade_filter == "short":
        return df[df["action"] == "sell"]
    return df

def _max_drawdown(profits: pd.Series) -> float:
    cum = profits.cumsum()
    return float((cum.cummax() - cum).max() or 0.0)

def _sharpe(returns: np.ndarray, rf: float, years: float) -> float:
    if returns.size == 0: return 0.0
    mu, sd = returns.mean(), returns.std()
    if years > 0:
        mu *= 252
        sd *= np.sqrt(252)
    return float((mu - rf) / sd) if sd > 0 else 0.0

def _sortino(returns: np.ndarray, rf: float) -> float:
    if returns.size == 0: return 0.0
    mean_adj = returns.mean() - rf
    downside = returns[returns < rf]
    dd = downside.std() if downside.size else 0.0
    return float(mean_adj / dd) if dd > 0 else 0.0

def _filtered_strategies(strategies: dict, trade_filter: str, selected: list) -> dict:
    """Select subset and apply trade filter."""
    subset = {k: v for k, v in strategies.items() if k in selected}
    return {k: _apply_trade_filter(df, trade_filter) for k, df in subset.items()}

def _pnl_figure(strategies: dict, show_total: bool, by_dates: bool) -> str:
    fig = go.Figure()
    x_label = "Trade"

    if show_total:
        if not strategies:
            return ""
        stacks = [d[["trade date", "profit"]] for d in strategies.values()]
        all_df = pd.concat(stacks, ignore_index=True)
        all_df["trade date"] = pd.to_datetime(all_df["trade date"])
        g = (all_df.groupby("trade date", as_index=False)["profit"]
             .sum().sort_values("trade date"))
        g["cumulative_pnl"] = g["profit"].cumsum()
        x = g["trade date"] if by_dates else g.index
        x_label = "Date" if by_dates else "Trade"
        fig.add_trace(go.Scatter(x=x, y=g["cumulative_pnl"], mode="lines", name="Total PnL"))
    else:
        for name, d in strategies.items():
            net = d["profit"].sum()
            x = d["trade date"] if by_dates else d.index
            x_label = "Date" if by_dates else "Trade"
            fig.add_trace(go.Scatter(x=x, y=d["profit"].cumsum(), mode="lines",
                                     name=f"{name}: ${net:,.0f}"))

    fig.update_layout(title="Cumulative PnL", xaxis_title=x_label, yaxis_title="PnL", height=520)
    return fig.to_html(full_html=False, include_plotlyjs="cdn")

def _corr_heatmap(strategies: dict) -> str:
    if len(strategies) < 2:
        return ""
    pnl = {name: df.groupby("trade date")["profit"].sum() for name, df in strategies.items()}
    pnl_df = pd.DataFrame(pnl).fillna(0.0)
    if pnl_df.shape[1] < 2:
        return ""
    corr = pnl_df.corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1,
                    title="Correlation Matrix (Daily PnL)")
    return fig.to_html(full_html=False, include_plotlyjs="cdn")

def _compute_summary(strategies: dict):
    if not strategies:
        return None
    all_df = (pd.concat([d[["trade date", "profit"]] for d in strategies.values()],
                        ignore_index=True).sort_values("trade date"))
    max_dd = _max_drawdown(all_df["profit"])
    total_profit = float(all_df["profit"].sum())
    total_trades = int(len(all_df))
    win_rate = float((all_df["profit"] > 0).mean() * 100) if total_trades else 0.0
    start, end = all_df["trade date"].min(), all_df["trade date"].max()
    years = max(((end - start).days / 365.25), 0.0)
    initial_capital = 10_000.0
    cagr = ((initial_capital + total_profit) / initial_capital) ** (1/years) - 1 if years > 0 else 0.0
    largest_loss = float(all_df["profit"].min())
    required_capital = float(max_dd * 2 + abs(largest_loss))
    returns = all_df["profit"].values / initial_capital
    sharpe = _sharpe(returns, rf=0.0, years=years)
    sortino = _sortino(returns, rf=0.0)
    return {
        "portfolio_profits": total_profit,
        "start": str(start.date()),
        "end": str(end.date()),
        "cagr": cagr,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "largest_loss": largest_loss,
        "required_capital": required_capital,
        "strategies": list(strategies.keys()),
    }

def _per_strategy_metrics(strategies: dict):
    """Return list[{name, profit, trades, win_rate, avg, avg_win, avg_loss, max_win, max_loss}]"""
    rows = []
    for name, df in strategies.items():
        if df.empty:
            rows.append({
                "name": name, "profit": 0.0, "trades": 0, "win_rate": 0.0,
                "avg": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "max_win": 0.0, "max_loss": 0.0
            })
            continue
        profit = float(df["profit"].sum())
        trades = int(len(df))
        win_rate = float((df["profit"] > 0).mean() * 100) if trades else 0.0
        avg = float(df["profit"].mean()) if trades else 0.0
        wins = df[df["profit"] > 0]["profit"]
        losses = df[df["profit"] < 0]["profit"]
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        max_win = float(wins.max()) if len(wins) else 0.0
        max_loss = float(losses.min()) if len(losses) else 0.0
        rows.append({
            "name": name, "profit": profit, "trades": trades, "win_rate": win_rate,
            "avg": avg, "avg_win": avg_win, "avg_loss": avg_loss,
            "max_win": max_win, "max_loss": max_loss
        })
    # sort by profit descending
    rows.sort(key=lambda r: r["profit"], reverse=True)
    return rows

# ---------- routes ----------
@bp.route("/")
def index():
    return render_template("index.html", active="home")

@bp.route("/about")
def about():
    return render_template("about.html", active="about")



@bp.route("/health")
def health():
    return jsonify(status="ok")

@bp.route("/version")
def version():
    return jsonify(app="strategy-analyzer", version="0.4.0", env=os.environ.get("ENV", "dev"))

# Upload new files (sets/overwrites session cache)
@bp.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        settings = {"show_total": True, "by_dates": True, "trade_filter": "all"}
        return render_template("upload.html", active="upload", result=None, charts=None, error=None,
                               settings=settings, names=[], selected=[], per_strategy=[])

    # toggles during upload
    show_total   = request.form.get("show_total") == "on"
    by_dates     = request.form.get("by_dates") == "on"
    trade_filter = request.form.get("trade_filter", "all")  # 'all'|'long'|'short'
    settings = {"show_total": show_total, "by_dates": by_dates, "trade_filter": trade_filter}

    files = request.files.getlist("file")
    if not files:
        return render_template("upload.html", result=None, charts=None,
                               error="No files provided", settings=settings,
                               names=[], selected=[], per_strategy=[])

    strategies = {}
    for f in files:
        if not f or not f.filename:
            continue
        ext = os.path.splitext(secure_filename(f.filename))[1].lower()
        if ext not in ALLOWED_EXT:
            return render_template("upload.html", result=None, charts=None,
                                   error=f"Unsupported file: {f.filename}",
                                   settings=settings, names=[], selected=[], per_strategy=[])
        buf = io.BytesIO(f.read())
        df = pd.read_excel(buf, engine="openpyxl")
        strategies[os.path.splitext(secure_filename(f.filename))[0]] = _normalize(df)

    if not strategies:
        return render_template("upload.html", result=None, charts=None,
                               error="No valid files", settings=settings,
                               names=[], selected=[], per_strategy=[])

    # Save to session store
    sid = _sid()
    STORE[sid] = {"strategies": strategies}

    # By default, all strategies selected
    names = list(strategies.keys())
    selected = names[:]

    # Apply filter & build output
    filtered = _filtered_strategies(strategies, trade_filter, selected)
    result = _compute_summary(filtered)
    per_strategy = _per_strategy_metrics(filtered)
    charts = {
        "pnl": _pnl_figure(filtered, show_total=show_total, by_dates=by_dates),
        "correlation": _corr_heatmap(filtered),
    }
    return render_template("upload.html", result=result, charts=charts, error=None,
                           settings=settings, names=names, selected=selected,
                           per_strategy=per_strategy)

# Re-render from cache without re-uploading; also supports selection & trade filter
@bp.route("/render", methods=["POST"])
def render_from_cache():
    sid = _sid()
    cache = STORE.get(sid)
    if not cache or "strategies" not in cache:
        return render_template("upload.html", result=None, charts=None,
                               error="No uploaded datasets found for this session. Upload files first.",
                               settings={"show_total": True, "by_dates": True, "trade_filter": "all"},
                               names=[], selected=[], per_strategy=[])

    strategies = cache["strategies"]

    # Read toggles + selection
    show_total   = request.form.get("show_total") == "on"
    by_dates     = request.form.get("by_dates") == "on"
    trade_filter = request.form.get("trade_filter", "all")
    selected     = request.form.getlist("include")  # multiple checkbox values
    names        = list(strategies.keys())

    if not selected:
        selected = names[:]

    filtered = _filtered_strategies(strategies, trade_filter, selected)
    if not filtered:
        return render_template("upload.html", result=None, charts=None,
                               error="Select at least one dataset.",
                               settings={"show_total": show_total, "by_dates": by_dates, "trade_filter": trade_filter},
                               names=names, selected=[], per_strategy=[])

    result = _compute_summary(filtered)
    per_strategy = _per_strategy_metrics(filtered)
    charts = {
        "pnl": _pnl_figure(filtered, show_total=show_total, by_dates=by_dates),
        "correlation": _corr_heatmap(filtered),
    }
    settings = {"show_total": show_total, "by_dates": by_dates, "trade_filter": trade_filter}
    return render_template("upload.html", result=result, charts=charts, error=None,
                           settings=settings, names=names, selected=selected,
                           per_strategy=per_strategy)
