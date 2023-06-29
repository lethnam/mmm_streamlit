# MMM Streamlit app

Main goal: build a web app (based on Streamlit) to run Marketing Mix Modeling experiments. The MMM is built around LightWeightMMM (https://lightweight-mmm.readthedocs.io/).

Current state: 
- Demo app: https://mmmapp-demo.streamlit.app
- The app generates a model with dummy data, and shows diagnostics / posteriors / media effects / budget allocator.
- Not yet capable of reading real data and logging experiments

To launch the app locally: run <code>streamlit run Home_Page.py</code>


>To be added / improved:
>- A file uploader
>- Experiment logging
>- Comparison of experiments
>- Extensive custom prior settings
>- Unittests
>- ...