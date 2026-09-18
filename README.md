# Grounding the Electorate

Interactive explorer of a master's thesis by Jorge Pulgarin: can gpt-oss-20b, given NVIDIA Nemotron synthetic personas, reproduce electoral patterns? USA 2024, El Salvador 2024 and Brazil 2026, three prompt contexts, two languages. Every number comes from saved model answers; no model is called.

- App: `app/streamlit_app.py` (Streamlit 1.64). Run locally with `streamlit run app/streamlit_app.py`.
- Data: on first start the app downloads the 2 GB index (all 1.4 million answers embedded) from the public dataset `Jorge18tu/grounding-the-electorate-data`.
- Built from the thesis repository with `streamlit_app/tools/build_cloud_bundle.py`.
