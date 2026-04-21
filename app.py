import streamlit as st

st.set_page_config(
    page_title="Market Risk Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

overview = st.Page("pages/1_Market_Overview.py", title="Market Overview", icon="📊", default=True)
market_data = st.Page("pages/2_Market_Data.py", title="Market Data", icon="📈")
risk = st.Page("pages/3_Risk_Analystics.py", title="Risk Analytics", icon="⚠️")
vanilla = st.Page("pages/4_Vanilla_Option_Pricer.py", title="Vanilla Option Pricer", icon="🎯")
exotic = st.Page("pages/5_Exotic_Option_Pricer.py", title="Exotic Options", icon="🎲")
bond = st.Page("pages/6_Bond_Pricer.py", title="Bond Pricer", icon="🏛️")

pg = st.navigation([overview, market_data, risk, vanilla, exotic, bond])
pg.run()