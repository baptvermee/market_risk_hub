import streamlit as st

st.set_page_config(
    page_title="Market Risk Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

overview = st.Page("pages/1_Market_Overview.py", title="Market Overview", icon="📊", default=True)
risk = st.Page("pages/2_Risk_Analystics.py", title="Risk Analytics", icon="⚠️")
vanilla = st.Page("pages/3_Vanilla_Option_Pricer.py", title="Vanilla Option Pricer", icon="🎯")
exotic = st.Page("pages/4_Exotic_Option_Pricer.py", title="Exotic Options", icon="🎲")
bond = st.Page("pages/5_Bond_Pricer.py", title="Bond Pricer", icon="🏛️")

pg = st.navigation([overview, risk, vanilla, exotic, bond])
pg.run()