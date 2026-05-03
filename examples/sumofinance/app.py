"""SumoFinance — Streamlit UI with full auth + dashboard + insights + simulation + chat."""
import streamlit as st
import requests
import json
import qrcode
import io
from datetime import datetime, timedelta

API = "http://localhost:8000"

st.set_page_config(page_title="SumoFinance AI Copilot", page_icon="💸", layout="wide")

st.markdown("""<style>
.stApp{background:linear-gradient(135deg,#0f0c29,#1a1a2e,#16213e);color:#e0e0e0}
div[data-testid="stMetric"]{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);border-radius:12px;padding:1rem}
.badge{display:inline-block;padding:.2rem .6rem;border-radius:16px;font-size:.8rem;font-weight:600}
.badge-ok{background:#00c853;color:#000}.badge-warn{background:#ff9100;color:#000}.badge-err{background:#ff1744;color:#fff}
</style>""", unsafe_allow_html=True)

# ─── Session defaults ─────────────────────────────────────────────────────────
for k, v in {"auth_state": "landing", "user_id": None, "access_token": None,
             "session_id": "s_" + datetime.now().strftime("%Y%m%d%H%M%S"),
             "messages": [], "qr_uri": None, "user_name": None, "user_email": None,
             "pending_user_id": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def hdr():
    h = {}
    if st.session_state.access_token:
        h["Authorization"] = f"Bearer {st.session_state.access_token}"
    return h

def qr_img(uri):
    q = qrcode.QRCode(version=1, box_size=8, border=3)
    q.add_data(uri); q.make(fit=True)
    buf = io.BytesIO(); q.make_image().save(buf, format="PNG"); return buf.getvalue()

# ═══════════════════════════════════════════════════════════════════════════════
# AUTH PAGES
# ═══════════════════════════════════════════════════════════════════════════════

def page_landing():
    st.markdown("# 💸 SumoFinance"); st.markdown("### AI-Powered Financial Decision Engine")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🆕 New User?")
        if st.button("Register", use_container_width=True, type="primary"):
            st.session_state.auth_state = "register"; st.rerun()
    with c2:
        st.markdown("#### 🔐 Returning User?")
        if st.button("Login", use_container_width=True):
            st.session_state.auth_state = "login"; st.rerun()

def page_register():
    st.markdown("# 📝 Register")
    if st.button("← Back"): st.session_state.auth_state = "landing"; st.rerun()
    with st.form("reg"):
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        pw = st.text_input("Password", type="password")
        phone = st.text_input("Phone (optional)")
        go = st.form_submit_button("Register", type="primary")
    if go and name and email and pw:
        r = requests.post(f"{API}/auth/register", json={"full_name": name, "email": email,
            "password": pw, "phone_number": phone})
        if r.status_code == 200:
            d = r.json()
            st.session_state.update(pending_user_id=d["user_id"], qr_uri=d["qr_code_uri"],
                                    user_name=name, user_email=email, auth_state="register_2fa")
            # Seed demo data
            requests.post(f"{API}/seed/{d['user_id']}")
            st.rerun()
        else: st.error(r.json().get("detail", "Error"))

def page_register_2fa():
    st.markdown("# 🔑 Set Up 2FA")
    st.success(f"Account created for **{st.session_state.user_name}**!")
    if st.session_state.qr_uri:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2: st.image(qr_img(st.session_state.qr_uri), caption="Scan with Authenticator App", width=280)
        with st.expander("📋 Manual key"):
            st.code(st.session_state.qr_uri.split("secret=")[1].split("&")[0])
    with st.form("v2fa"):
        code = st.text_input("6-Digit Code", max_chars=6)
        go = st.form_submit_button("Verify", type="primary")
    if go and code:
        r = requests.post(f"{API}/auth/verify-2fa", json={"user_id": st.session_state.pending_user_id, "otp_code": code})
        if r.status_code == 200:
            d = r.json()
            st.session_state.update(access_token=d["access_token"], user_id=d["user_id"],
                                    qr_uri=None, auth_state="authenticated")
            st.balloons(); st.rerun()
        else: st.error(r.json().get("detail", "Invalid"))

def page_login():
    st.markdown("# 🔐 Login")
    if st.button("← Back"): st.session_state.auth_state = "landing"; st.rerun()
    with st.form("login"):
        email = st.text_input("Email")
        pw = st.text_input("Password", type="password")
        go = st.form_submit_button("Continue", type="primary")
    if go and email and pw:
        r = requests.post(f"{API}/auth/login", json={"email": email, "password": pw})
        if r.status_code == 200:
            d = r.json()
            st.session_state.update(pending_user_id=d["user_id"], user_email=email, auth_state="login_2fa")
            st.rerun()
        else: st.error(r.json().get("detail", "Invalid"))

def page_login_2fa():
    st.markdown("# 🔐 Enter 2FA Code")
    with st.form("l2fa"):
        code = st.text_input("6-Digit Code", max_chars=6)
        go = st.form_submit_button("Verify & Login", type="primary")
    if go and code:
        r = requests.post(f"{API}/auth/login/2fa", json={"user_id": st.session_state.pending_user_id, "otp_code": code})
        if r.status_code == 200:
            d = r.json()
            st.session_state.update(access_token=d["access_token"], user_id=d["user_id"], auth_state="authenticated")
            st.rerun()
        else: st.error(r.json().get("detail", "Invalid"))

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def page_dashboard():
    uid = st.session_state.user_id

    # Sidebar
    with st.sidebar:
        st.markdown("### 💸 SumoFinance")
        st.divider()
        if st.session_state.user_name: st.markdown(f"👤 **{st.session_state.user_name}**")
        if st.session_state.user_email: st.caption(st.session_state.user_email)
        st.markdown('<span class="badge badge-ok">🔒 2FA Verified</span>', unsafe_allow_html=True)
        # Nudge count
        try:
            nc = requests.get(f"{API}/nudges/{uid}/unread-count", headers=hdr()).json().get("count", 0)
            if nc > 0: st.warning(f"🔔 {nc} unread nudge{'s' if nc > 1 else ''}")
        except: pass
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()

    # Load summary
    try:
        summary = requests.get(f"{API}/dashboard/summary/{uid}", headers=hdr()).json()
    except:
        summary = None

    if not summary:
        st.error("API server not responding. Start with: TMPDIR=/mnt/data/tmp uvicorn api:app --reload")
        return

    user_info = summary.get("user", {})
    spending = summary.get("spending", {})
    s2s = summary.get("safe_to_spend", 0)

    if not st.session_state.user_name:
        st.session_state.user_name = user_info.get("name", "User")

    st.title(f"Hello, {user_info.get('name', 'User')} 👋")

    # ─── Tabs ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Dashboard", "🔍 Insights", "🎯 Goals", "🧪 Simulation", "🤖 Chat", "🔔 Nudges", "💳 Admin"])

    # ─── TAB 1: Dashboard ─────────────────────────────────────────────────
    with tab1:
        c1, c2, c3, c4, c5 = st.columns(5)
        m = spending.get("month", {})
        w = spending.get("week", {})
        c1.metric("Balance", f"₹{user_info.get('balance', 0):,.0f}")
        c2.metric("Avg Income", f"₹{user_info.get('income', 0):,.0f}")
        c3.metric("Spent (Month)", f"₹{m.get('total_spent', 0):,.0f}")
        c4.metric("Spent (Week)", f"₹{w.get('total_spent', 0):,.0f}")
        c5.metric("Safe to Spend", f"₹{s2s:,.0f}")

        st.subheader("Spending Trend")
        try:
            trend = requests.get(f"{API}/dashboard/spending-trend/{uid}?view=daily", headers=hdr()).json()
            if trend.get("data"):
                import pandas as pd
                df = pd.DataFrame(trend["data"])
                st.line_chart(df.set_index("date")["amount"])
        except: st.info("No trend data yet.")

        st.subheader("Alerts")
        try:
            alerts = requests.get(f"{API}/dashboard/alerts/{uid}", headers=hdr()).json()
            for a in alerts.get("alerts", []):
                if a["type"] == "critical": st.error(f"🚨 **{a['title']}** — {a['description']}")
                else: st.warning(f"⚡ **{a['title']}** — {a['description']}")
            if not alerts.get("alerts"): st.success("No alerts — you're on track!")
        except: pass
        
        st.subheader("💡 Smart Investments (Discovery Agent)")
        try:
            schemes_data = requests.get(f"{API}/recommendations/schemes/{uid}", headers=hdr()).json()
            if schemes_data.get("schemes"):
                st.info(f"**Surplus:** ₹{schemes_data['surplus']:,.0f} | **Target Goal:** {schemes_data['goal']}")
                
                cols = st.columns(len(schemes_data["schemes"]))
                for i, s in enumerate(schemes_data["schemes"]):
                    with cols[i]:
                        st.markdown(f"**{s['name']}**")
                        st.caption(f"{s['category']} | Risk: {s['risk_level']}")
                        st.markdown(f"## {s['returns_percentage']}%")
                        st.markdown(f"*Lock-in: {s['lock_in_months']} mo*")
                        st.markdown(f"> {s['fine_print']}")
                        st.button("Invest", key=f"inv_{s['scheme_id']}", use_container_width=True)
            else:
                st.write(schemes_data.get("message", "Keep building your surplus to unlock investment recommendations!"))
        except:
            st.error("Could not load investment recommendations.")

    # ─── TAB 2: Insights ──────────────────────────────────────────────────
    with tab2:
        st.header("Spending Insights")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Category Breakdown")
            try:
                bd = requests.get(f"{API}/insights/category-breakdown/{uid}", headers=hdr()).json()
                for c in bd.get("categories", []):
                    st.progress(min(c["percentage"]/100, 1.0),
                               text=f"{c['name']}: ₹{c['amount']:,.0f} ({c['percentage']}%)")
            except: st.info("No data.")

        with col2:
            st.subheader("Top Merchants")
            try:
                ma = requests.get(f"{API}/insights/merchant-analysis/{uid}", headers=hdr()).json()
                for m in ma.get("merchants", [])[:7]:
                    st.markdown(f"**{m['merchant']}** ({m['category']}) — ₹{m['total_spent']:,.0f} ({m['frequency']}x)")
            except: pass

        st.subheader("AI Observations")
        try:
            obs = requests.get(f"{API}/insights/ai-observations/{uid}", headers=hdr()).json()
            for o in obs.get("observations", []):
                icon = {"critical": "🚨", "warning": "⚡", "info": "💡"}.get(o["severity"], "📌")
                st.markdown(f"{icon} **{o['title']}** — {o['description']}")
                st.caption(f"→ {o.get('suggested_action', '')}")
            if not obs.get("observations"): st.success("No observations — your finances look healthy!")
        except: pass

        st.subheader("Month-over-Month Comparison")
        try:
            comp = requests.get(f"{API}/insights/comparison/{uid}", headers=hdr()).json()
            for d in comp.get("differences", []):
                delta = f"{d['change_pct']:+.0f}%"
                color = "🔴" if d["behavior"] == "worsened" else "🟢" if d["behavior"] == "improved" else "⚪"
                st.markdown(f"{color} **{d['category']}**: {delta} (₹{d['current']:,.0f} vs ₹{d['previous']:,.0f})")
        except: pass

    # ─── TAB 3: Goals ─────────────────────────────────────────────────────
    with tab3:
        st.header("Financial Goals")
        try:
            prog = requests.get(f"{API}/progress/{uid}", headers=hdr()).json()
            st.metric("Monthly Savings Progress", f"{prog.get('progress', 0):.0f}%",
                      delta=f"₹{prog.get('current_saving', 0):,.0f} / ₹{prog.get('target', 0):,.0f}")
            for g in prog.get("goals", []):
                col = "🟢" if g["status"] == "active" else "⏸️"
                st.progress(min(g["progress"]/100, 1.0), text=f"{col} {g['name']}: {g['progress']:.0f}%")
        except: pass

        with st.form("new_goal"):
            st.subheader("Create New Goal")
            g_name = st.text_input("Goal Name", "New Laptop")
            g_target = st.number_input("Target ₹", value=50000.0, step=1000.0)
            g_start = st.date_input("Start", datetime.now())
            g_end = st.date_input("End", datetime.now() + timedelta(days=90))
            if st.form_submit_button("Create", type="primary"):
                r = requests.post(f"{API}/goals/{uid}", json={"name": g_name, "target_amount": g_target,
                    "start_date": g_start.isoformat(), "end_date": g_end.isoformat()}, headers=hdr())
                if r.status_code == 200: st.success("Goal created!"); st.rerun()

    # ─── TAB 4: Simulation ────────────────────────────────────────────────
    with tab4:
        st.header("What-If Simulator")
        try:
            bd = requests.get(f"{API}/insights/category-breakdown/{uid}", headers=hdr()).json()
            cat_names = [c["name"] for c in bd.get("categories", [])]
        except:
            cat_names = ["Food", "Shopping", "Entertainment", "Transport"]

        sel_cat = st.selectbox("Category to reduce", cat_names)
        slider = st.slider("Reduction %", 5, 50, 20, 5)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Instant Impact")
            try:
                r = requests.post(f"{API}/simulation/interactive/{uid}",
                                  json={"category": sel_cat, "slider_value": slider}, headers=hdr()).json()
                st.metric("Monthly Saving", f"₹{r.get('monthly_saving', 0):,.0f}")
                st.metric("Yearly Saving", f"₹{r.get('yearly_saving', 0):,.0f}")
                st.metric("Success Probability", f"{r.get('success_probability', 0)*100:.0f}%")
            except: pass

        with c2:
            st.subheader("6-Month Projection")
            try:
                r = requests.post(f"{API}/simulation/run/{uid}",
                                  json={"category": sel_cat, "target_reduction": slider,
                                        "time_horizon_months": 6}, headers=hdr()).json()
                proj = r.get("monthly_projection", [])
                if proj:
                    import pandas as pd
                    df = pd.DataFrame(proj)
                    st.line_chart(df.set_index("month")["savings"])
                hm = r.get("habit_model", {})
                st.markdown(f"**Resistance**: {hm.get('resistance', '?')} | **Adoption**: {hm.get('expected_adoption', '?')}")
                sr = r.get("stochastic_range", {})
                st.caption(f"Expected savings: ₹{sr.get('expected', 0):,.0f} ± ₹{sr.get('variance', 0):,.0f}")
            except: pass

        st.subheader("Recommendations")
        c1, c2 = st.columns(2)
        with c1:
            try:
                strat = requests.get(f"{API}/recommendations/strategy-summary/{uid}", headers=hdr()).json()
                st.info(strat.get("strategy", "No strategy available."))
            except: pass
            
            plan = st.radio("Plan", ["easy", "moderate", "aggressive"], index=1, horizontal=True)
            try:
                recs = requests.get(f"{API}/recommendations/list/{uid}?plan={plan}", headers=hdr()).json()
                for r in recs.get("recommendations", []):
                    st.markdown(f"• **{r['title']}** — {r['description']} (₹{r['impact_yearly']:,.0f}/yr, {r['success_probability']*100:.0f}% success)")
                st.metric("Total Potential Savings", f"₹{recs.get('total_savings', 0):,.0f}/year")
            except: pass
            
        st.divider()
        st.subheader("📊 Behavioral Patterns & Forecast")
        try:
            pred = requests.get(f"{API}/simulation/prediction/{uid}", headers=hdr()).json()
            if pred:
                col1, col2, col3 = st.columns(3)
                col1.metric("Avg Weekday", f"₹{pred.get('avg_weekday', 0):,.0f}")
                col2.metric("Avg Weekend", f"₹{pred.get('avg_weekend', 0):,.0f}", 
                            delta=f"{(pred.get('avg_weekend', 0)/max(pred.get('avg_weekday', 1), 1) - 1)*100:+.0f}%" if pred.get('avg_weekday', 0) > 0 else None,
                            delta_color="inverse")
                col3.metric("Projected EOM Spend", f"₹{pred.get('projected_eom', 0):,.0f}")
                
                for w in pred.get("warnings", []):
                    if "🚨" in w: st.error(w)
                    elif "💡" in w: st.info(w)
                    else: st.warning(w)
            else:
                st.info("Not enough data to predict patterns yet.")
        except:
            st.error("Could not load predictive insights.")
            
        with c2:
            st.subheader("What If Ignored?")
            try:
                rh = requests.get(f"{API}/recommendations/history/{uid}", headers=hdr()).json()
                history = rh.get("recommendations", [])
                if history:
                    rec_id = st.selectbox("Select Recommendation", [r["rec_id"] for r in history], format_func=lambda x: next(r["title"] for r in history if r["rec_id"] == x))
                    if st.button("Check Impact"):
                        wi = requests.post(f"{API}/simulation/what-if-not-followed/{uid}", json={"recommendation_id": rec_id}, headers=hdr()).json()
                        st.error(wi.get("impact", {}).get("message", "No impact calculated."))
                else:
                    st.info("No recommendation history available.")
            except: pass

    # ─── TAB 5: Chat ──────────────────────────────────────────────────────
    with tab5:
        st.header("Chat with Sumo 🤖")
        st.caption("Ask questions, give commands, get insights — the chat IS the control panel.")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about spending, set goals, mark merchants..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Sumo is thinking..."):
                    try:
                        r = requests.post(f"{API}/chat/{uid}/{st.session_state.session_id}",
                                          json={"message": prompt}, headers=hdr())
                        resp = r.json().get("response", "Error getting response.")
                    except Exception as e:
                        resp = f"Connection error: {e}"
                    st.markdown(resp)
                    st.session_state.messages.append({"role": "assistant", "content": resp})
                    
        if st.button("Reset Chat Session"):
            requests.post(f"{API}/chat/{uid}/reset-session", headers=hdr())
            st.session_state.messages = []
            st.session_state.session_id = "s_" + datetime.now().strftime("%Y%m%d%H%M%S")
            st.rerun()

    # ─── TAB 6: Nudges ────────────────────────────────────────────────────
    with tab6:
        st.header("🔔 Nudges & Notifications")
        try:
            nudges = requests.get(f"{API}/nudges/{uid}", headers=hdr()).json().get("nudges", [])
            for n in nudges:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{'🔴' if n['read'] == 0 else '⚪'}** {n['message']}")
                    st.caption(f"Received: {n['created_at'][:16]}")
                with col2:
                    if n['read'] == 0:
                        if st.button("Mark Read", key=f"read_{n['nudge_id']}"):
                            requests.post(f"{API}/nudges/{uid}/{n['nudge_id']}/read", headers=hdr())
                            st.rerun()
                st.divider()
            if not nudges: st.success("No nudges!")
        except Exception as e: st.error(f"Could not load nudges: {e}")

    # ─── TAB 7: Admin & Transactions ──────────────────────────────────────
    with tab7:
        c1, c2 = st.columns(2)
        with c1:
            st.header("💳 Recent Transactions")
            try:
                txns = requests.get(f"{API}/transactions/{uid}?limit=10", headers=hdr()).json().get("transactions", [])
                import pandas as pd
                if txns:
                    df = pd.DataFrame(txns)[["timestamp", "merchant", "category", "amount", "type"]]
                    st.dataframe(df, hide_index=True)
                else: st.info("No transactions found.")
            except: pass
            
            with st.form("add_txn"):
                st.subheader("Mock Transaction")
                m = st.text_input("Merchant", "Starbucks")
                a = st.number_input("Amount ₹", value=350.0)
                if st.form_submit_button("Add Transaction", type="primary"):
                    requests.post(f"{API}/transaction/{uid}", json={"merchant": m, "amount": a}, headers=hdr())
                    st.success("Added!"); st.rerun()

        with c2:
            st.header("⚙️ Feature Flags")
            try:
                flags = requests.get(f"{API}/features/{uid}", headers=hdr()).json().get("features", {})
                for f_name, f_val in flags.items():
                    new_val = st.toggle(f_name.replace("_", " ").title(), value=f_val)
                    if new_val != f_val:
                        requests.patch(f"{API}/features/{uid}/{f_name}?enabled={str(new_val).lower()}", headers=hdr())
                        st.rerun()
            except: pass
            
            st.divider()
            st.subheader("Admin Tools")
            if st.button("🌱 Regenerate Seed Data", type="primary"):
                requests.post(f"{API}/seed/{uid}", headers=hdr())
                st.success("Seeded 150+ transactions!"); st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

{"landing": page_landing, "register": page_register, "register_2fa": page_register_2fa,
 "login": page_login, "login_2fa": page_login_2fa, "authenticated": page_dashboard
}.get(st.session_state.auth_state, page_landing)()
