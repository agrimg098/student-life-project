import streamlit as st
import pandas as pd
import os, hashlib
from datetime import datetime
import plotly.express as px
from sklearn.linear_model import LogisticRegression

# ---------- CONFIG ----------
st.set_page_config(page_title="AI ML Student System", layout="wide")

USER_FILE = "users.csv"
DATA_FOLDER = "user_data"

os.makedirs(DATA_FOLDER, exist_ok=True)
if not os.path.exists(USER_FILE):
    pd.DataFrame(columns=["username","password"]).to_csv(USER_FILE,index=False)

# ---------- STYLE ----------
st.markdown("""
<style>
body {background-color:#0e1117;color:white;}
.stButton>button {background:#4CAF50;color:white;border-radius:10px;}
</style>
""", unsafe_allow_html=True)

# ---------- SECURITY ----------
def hash_pass(p): return hashlib.sha256(p.encode()).hexdigest()

# ---------- USER ----------
def load_users():
    try: return pd.read_csv(USER_FILE)
    except: return pd.DataFrame(columns=["username","password"])

def signup(u,p):
    users=load_users()
    u,p=u.strip(),p.strip()
    if u=="" or p=="": return "empty"
    if u in users["username"].astype(str).values: return "exists"

    new=pd.DataFrame({"username":[u],"password":[hash_pass(p)]})
    pd.concat([users,new]).to_csv(USER_FILE,index=False)
    return "success"

def login(u,p):
    users=load_users()
    return not users[
        (users["username"]==u.strip()) &
        (users["password"]==hash_pass(p.strip()))
    ].empty

# ---------- DATA ----------
def file(u): return f"{DATA_FOLDER}/{u}.csv"

def load_data(u):
    f=file(u)
    if not os.path.exists(f):
        pd.DataFrame(columns=["Date","Sleep","Study","Spend","Screen","Mood","Stress","Burnout"]).to_csv(f,index=False)
    return pd.read_csv(f)

def save_data(u,df): df.to_csv(file(u),index=False)

# ---------- LABEL ----------
def label(d):
    score=0
    if d["Sleep"]<6: score+=1
    if d["Screen"]>8: score+=1
    if d["Stress"]>7: score+=1
    if d["Mood"]<5: score+=1
    return 1 if score>=2 else 0

# ---------- ML ----------
def train(df):
    if len(df)<10: return None
    X=df[["Sleep","Study","Spend","Screen","Mood","Stress"]]
    y=df["Burnout"]
    m=LogisticRegression()
    m.fit(X,y)
    return m

def predict(m, d):
    if m is None: return None, None
    X=[[d["Sleep"],d["Study"],d["Spend"],d["Screen"],d["Mood"],d["Stress"]]]
    prob=m.predict_proba(X)[0][1]
    return ("High" if prob>0.5 else "Low"), round(prob*100,1)

# ---------- HEALTH ----------
def health(d):
    return round((d["Sleep"]/8*25)+(d["Study"]/4*20)+(d["Mood"]/10*15),1)

# ---------- SESSION ----------
if "login" not in st.session_state:
    st.session_state.login=False
    st.session_state.user=""

# ---------- UI ----------
st.title("🤖 AI + ML Student Wellness System")

# ---------- LOGIN ----------
if not st.session_state.login:
    menu=st.sidebar.radio("Menu",["Login","Signup"])
    u=st.text_input("Username")
    p=st.text_input("Password",type="password")

    if menu=="Signup":
        if st.button("Signup"):
            r=signup(u,p)
            st.success("Created") if r=="success" else st.warning("Exists") if r=="exists" else st.error("Empty")

    else:
        if st.button("Login"):
            if login(u,p):
                st.session_state.login=True
                st.session_state.user=u
                st.rerun()
            else:
                st.error("Wrong credentials")

# ---------- DASHBOARD ----------
else:
    u=st.session_state.user
    df=load_data(u)

    st.sidebar.success(u)
    if st.sidebar.button("Logout"):
        st.session_state.login=False
        st.rerun()

    page=st.sidebar.radio("Navigate",["Enter Data","Dashboard"])

    # ---------- ENTRY ----------
    if page=="Enter Data":
        st.subheader("Enter Daily Data")

        with st.form("f"):
            d=st.date_input("Date",datetime.today())
            s=st.slider("Sleep",0,12,7)
            stt=st.slider("Study",0,12,4)
            sp=st.number_input("Spend",0,5000,100)
            sc=st.slider("Screen",0,12,5)
            m=st.slider("Mood",1,10,7)
            sttress=st.slider("Stress",1,10,4)

            if st.form_submit_button("Save"):
                new=pd.DataFrame({
                    "Date":[str(d)],"Sleep":[s],"Study":[stt],
                    "Spend":[sp],"Screen":[sc],"Mood":[m],"Stress":[sttress]
                })
                new["Burnout"]=new.apply(label,axis=1)

                df=pd.concat([df,new])
                save_data(u,df)
                st.success("Saved")
                st.rerun()

    # ---------- DASHBOARD ----------
    else:
        if df.empty:
            st.info("Add data first")
        else:
            df["Date"]=pd.to_datetime(df["Date"])
            latest=df.iloc[-1]

            model=train(df)
            pred,conf=predict(model,latest)

            st.subheader("Analytics")

            st.metric("Health Score",health(latest))

            if pred:
                st.markdown(f"### 🤖 ML Prediction: **{pred} Risk ({conf}% confidence)**")
            else:
                st.info("Add 10+ entries for ML prediction")

            # Alerts
            if latest["Sleep"]<6: st.error("Low Sleep")
            if latest["Stress"]>7: st.error("High Stress")
            if latest["Screen"]>8: st.warning("High Screen")

            # AI Insight
            st.subheader("🤖 AI Insight")
            if latest["Stress"]>7:
                st.error("High stress detected. Consider breaks.")
            elif latest["Sleep"]<6:
                st.warning("Sleep is low. Improve rest.")
            else:
                st.success("Lifestyle looks balanced.")

            # Charts
            st.plotly_chart(px.line(df,x="Date",y="Sleep",markers=True),use_container_width=True)
            st.plotly_chart(px.line(df,x="Date",y="Study",markers=True),use_container_width=True)

            # Weekly
            w=df.tail(7)
            st.subheader("Weekly Summary")
            st.write(f"Sleep: {w.Sleep.mean():.1f} hrs")
            st.write(f"Study: {w.Study.mean():.1f} hrs")