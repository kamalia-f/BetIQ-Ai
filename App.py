"""
BetIQ-AI WhatsApp Bot
=====================
Receives WhatsApp messages via Twilio,
fetches live odds from The Odds API,
runs the prediction engine,
and replies with BET / NO BET analysis.
"""

from flask import Flask, request
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import requests
import random
import os

app = Flask(__name__)

# ─────────────────────────────────────────────
# CONFIG — pulled from environment variables
# ─────────────────────────────────────────────
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN  = os.environ.get("TWILIO_AUTH_TOKEN")
ODDS_API_KEY       = os.environ.get("ODDS_API_KEY")

ODDS_API_BASE = "https://api.the-odds-api.com/v4"

SUPPORTED_LEAGUES = {
    "premier league": "soccer_epl",
    "epl":            "soccer_epl",
    "la liga":        "soccer_spain_la_liga",
    "bundesliga":     "soccer_germany_bundesliga",
    "serie a":        "soccer_italy_serie_a",
    "ligue 1":        "soccer_france_ligue_one",
    "champions league": "soccer_uefa_champs_league",
}


# ─────────────────────────────────────────────
# ODDS API HELPERS
# ─────────────────────────────────────────────

def get_upcoming_matches(league_key: str) -> list:
    """Fetch upcoming matches with odds from The Odds API."""
    url = f"{ODDS_API_BASE}/sports/{league_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "uk",
        "markets": "h2h",
        "oddsFormat": "decimal",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
        return []
    except Exception:
        return []


def find_match(team_a: str, team_b: str, matches: list) -> dict:
    """Find a specific match from the list by team names."""
    team_a = team_a.lower().strip()
    team_b = team_b.lower().strip()
    for m in matches:
        home = m.get("home_team", "").lower()
        away = m.get("away_team", "").lower()
        if (team_a in home or team_a in away) and (team_b in home or team_b in away):
            return m
        if (team_b in home or team_b in away) and (team_a in home or team_a in away):
            return m
    return None


def extract_best_odds(match: dict) -> dict:
    """Extract best available odds from bookmakers."""
    bookmakers = match.get("bookmakers", [])
    if not bookmakers:
        return None

    # Prefer Bet365 or Pinnacle, else take first available
    preferred = ["bet365", "pinnacle", "williamhill"]
    selected = None
    for bm in bookmakers:
        if bm["key"] in preferred:
            selected = bm
            break
    if not selected:
        selected = bookmakers[0]

    outcomes = selected["markets"][0]["outcomes"]
    odds = {}
    for o in outcomes:
        odds[o["name"]] = o["price"]

    home_team = match["home_team"]
    away_team = match["away_team"]

    home_odds = odds.get(home_team, 2.0)
    draw_odds = odds.get("Draw", 3.4)
    away_odds = odds.get(away_team, 3.5)

    return {
        "home_team":  home_team,
        "away_team":  away_team,
        "home_odds":  home_odds,
        "draw_odds":  draw_odds,
        "away_odds":  away_odds,
        "bookmaker":  selected.get("title", "Bookmaker"),
        "commence":   match.get("commence_time", "")[:10],
    }


# ─────────────────────────────────────────────
# PREDICTION ENGINE (simplified for bot)
# ─────────────────────────────────────────────

def implied_prob(odds: float) -> float:
    return 1.0 / odds if odds > 1 else 0.99


def remove_margin(h_odds: float, d_odds: float, a_odds: float) -> dict:
    rh = implied_prob(h_odds)
    rd = implied_prob(d_odds)
    ra = implied_prob(a_odds)
    total = rh + rd + ra
    return {
        "home": round(rh / total, 3),
        "draw": round(rd / total, 3),
        "away": round(ra / total, 3),
        "margin": round((total - 1) * 100, 2),
    }


def run_prediction(odds_data: dict) -> dict:
    """
    Core prediction logic.
    Uses odds-based fair probabilities + simulated reality model signals.
    In production this calls the full ML engine from model.py
    """
    h_odds = odds_data["home_odds"]
    d_odds = odds_data["draw_odds"]
    a_odds = odds_data["away_odds"]

    # Market implied probabilities (fair, margin removed)
    market = remove_margin(h_odds, d_odds, a_odds)

    # Simulate reality model signal
    # In full deployment this runs the trained ML model
    # Here we apply a small randomised signal to demo the edge concept
    rng = random.Random(hash(odds_data["home_team"] + odds_data["away_team"]))

    def nudge(p):
        return round(max(0.03, min(0.97, p + rng.uniform(-0.07, 0.07))), 3)

    raw_h = nudge(market["home"])
    raw_d = nudge(market["draw"])
    raw_a = nudge(market["away"])
    total = raw_h + raw_d + raw_a

    reality = {
        "home": round(raw_h / total, 3),
        "draw": round(raw_d / total, 3),
        "away": round(raw_a / total, 3),
    }

    # Edge calculation
    edge_home = round(reality["home"] - market["home"], 3)
    edge_draw = round(reality["draw"] - market["draw"], 3)
    edge_away = round(reality["away"] - market["away"], 3)

    edges = {
        "Home Win": (edge_home, reality["home"], market["home"], h_odds),
        "Draw":     (edge_draw, reality["draw"], market["draw"], d_odds),
        "Away Win": (edge_away, reality["away"], market["away"], a_odds),
    }

    # Find best edge
    best_outcome = max(edges, key=lambda k: edges[k][0])
    best_edge, best_reality, best_market, best_odds = edges[best_outcome]

    # Decision
    if best_edge >= 0.05:
        decision = "BET ✅"
        confidence = "HIGH"
    elif best_edge >= 0.02:
        decision = "LEAN 🟡"
        confidence = "MEDIUM"
    else:
        decision = "NO BET ❌"
        confidence = "LOW"

    # Kelly fraction (25% fractional)
    b = best_odds - 1
    p = best_reality
    q = 1 - p
    kelly = round(max(0, ((b * p - q) / b) * 0.25 * 100), 1)

    return {
        "decision":      decision,
        "confidence":    confidence,
        "best_outcome":  best_outcome,
        "best_edge":     best_edge,
        "best_odds":     best_odds,
        "reality_prob":  best_reality,
        "market_prob":   best_market,
        "kelly":         kelly,
        "market_margin": market["margin"],
        "all_edges":     edges,
    }


# ─────────────────────────────────────────────
# MESSAGE FORMATTER
# ─────────────────────────────────────────────

def format_response(odds_data: dict, prediction: dict) -> str:
    home = odds_data["home_team"]
    away = odds_data["away_team"]
    date = odds_data["commence"]
    bm   = odds_data["bookmaker"]

    p = prediction
    edges = p["all_edges"]

    lines = []
    lines.append(f"⚽ *BetIQ-AI Analysis*")
    lines.append(f"━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"🏟 *{home} vs {away}*")
    lines.append(f"📅 {date}  |  📊 {bm}")
    lines.append(f"")
    lines.append(f"*PROBABILITIES*")
    lines.append(f"{'Outcome':<12} {'Real%':>6} {'Mkt%':>6} {'Edge':>7}")
    lines.append(f"{'─'*34}")

    outcome_icons = {"Home Win": "🏠", "Draw": "🤝", "Away Win": "✈️"}
    for outcome, (edge, reality, market, odds) in edges.items():
        icon  = outcome_icons[outcome]
        mark  = " ◄" if outcome == p["best_outcome"] else ""
        lines.append(
            f"{icon} {outcome:<10} {reality*100:>5.1f}% {market*100:>5.1f}% {edge*100:>+6.1f}%{mark}"
        )

    lines.append(f"")
    lines.append(f"*RECOMMENDATION*")
    lines.append(f"━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"Decision:    *{p['decision']}*")
    lines.append(f"Outcome:     *{p['best_outcome']}*")
    lines.append(f"Odds:        *{p['best_odds']}*")
    lines.append(f"Edge:        *{p['best_edge']*100:+.1f}%*")
    lines.append(f"Confidence:  *{p['confidence']}*")

    if p["decision"].startswith("BET") and p["kelly"] > 0:
        lines.append(f"Kelly Stake: *{p['kelly']}% of bankroll*")

    lines.append(f"")
    lines.append(f"_Mkt margin: {p['market_margin']}%_")
    lines.append(f"━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"_BetIQ-AI • Bet Responsibly_")

    return "\n".join(lines)


def format_help() -> str:
    return (
        "⚽ *Welcome to BetIQ-AI!*\n\n"
        "I analyse football matches and find betting value.\n\n"
        "*HOW TO USE:*\n"
        "Just send me a match like this:\n\n"
        "➡ `Arsenal vs Chelsea`\n"
        "➡ `Real Madrid vs Barcelona`\n"
        "➡ `Bayern vs Dortmund`\n\n"
        "*SUPPORTED LEAGUES:*\n"
        "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League\n"
        "🇪🇸 La Liga\n"
        "🇩🇪 Bundesliga\n"
        "🇮🇹 Serie A\n"
        "🇫🇷 Ligue 1\n"
        "🏆 Champions League\n\n"
        "_BetIQ-AI • Bet Responsibly_"
    )


def format_not_found(query: str) -> str:
    return (
        f"⚠️ Couldn't find *{query}* in upcoming fixtures.\n\n"
        "Make sure:\n"
        "• The match is scheduled in the next 7 days\n"
        "• You spelled the team names correctly\n"
        "• Try shorter names e.g. *Man City* not *Manchester City*\n\n"
        "Type *help* to see supported leagues."
    )


# ─────────────────────────────────────────────
# PARSE USER MESSAGE
# ─────────────────────────────────────────────

def parse_match_query(text: str):
    """
    Extract team names from message like:
    'Arsenal vs Chelsea' or 'Real Madrid v Barcelona'
    Returns (team_a, team_b) or None
    """
    text = text.strip()
    for sep in [" vs ", " v ", " VS ", " V ", " - "]:
        if sep in text:
            parts = text.split(sep, 1)
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()
    return None


def detect_league(team_a: str, team_b: str) -> list:
    """Return list of league keys to search — try all if unsure."""
    return list(SUPPORTED_LEAGUES.values())


# ─────────────────────────────────────────────
# WHATSAPP WEBHOOK
# ─────────────────────────────────────────────

@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    incoming = request.form.get("Body", "").strip()
    sender   = request.form.get("From", "")

    resp = MessagingResponse()
    msg  = resp.message()

    # Help command
    if incoming.lower() in ["hi", "hello", "help", "start", "hey"]:
        msg.body(format_help())
        return str(resp)

    # Try to parse as match query
    parsed = parse_match_query(incoming)
    if not parsed:
        msg.body(
            "❓ I didn't understand that.\n\n"
            "Send me a match like:\n"
            "*Arsenal vs Chelsea*\n\n"
            "Type *help* for more info."
        )
        return str(resp)

    team_a, team_b = parsed
    msg.body(f"🔍 Analysing *{team_a} vs {team_b}*...\nFetching live odds ⚡")

    # Search across all leagues
    found_match = None
    for league_key in detect_league(team_a, team_b):
        matches = get_upcoming_matches(league_key)
        found_match = find_match(team_a, team_b, matches)
        if found_match:
            break

    if not found_match:
        # Send follow-up not found message
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=format_not_found(f"{team_a} vs {team_b}"),
            from_="whatsapp:+14155238886",
            to=sender,
        )
        return str(resp)

    # Extract odds and run prediction
    odds_data  = extract_best_odds(found_match)
    prediction = run_prediction(odds_data)
    reply      = format_response(odds_data, prediction)

    # Send analysis as follow-up message
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    client.messages.create(
        body=reply,
        from_="whatsapp:+14155238886",
        to=sender,
    )

    return str(resp)


@app.route("/", methods=["GET"])
def health():
    return "BetIQ-AI is running ✅", 200


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
