# MMM Streamlit app

####Main goal
Build a web app (based on Streamlit https://streamlit.io/) to run Marketing Mix Modeling experiments. The MMM is built around LightWeightMMM (https://lightweight-mmm.readthedocs.io/).

####Current state 
- Demo app: https://mmmapp-demo.streamlit.app
- The app generates a model with dummy data, and shows diagnostics / posteriors / media effects / budget allocator.
- Not yet capable of reading real data and logging experiments

To launch the app locally: run <code>streamlit run Home_Page.py</code>

####To be added / improved:
- A file uploader
- Experiment logging
- Comparison of experiments
- Options to choose different media transformations (currently default Hill Adstock)
- Extensive custom prior settings
- Proper unit tests
- ...

####Project structure
- Scripts:
    - <code>scripts/</code>:
        - <code>mmm.py</code>: LightweightMMM wrapper
        - <code>mmmstreamlit.py</code>: A class to prepare visuals for Streamlit pages
        - <code>utils.py</code>: Helpers to generate and save dummy demo model
- Streamlit pages:
    - <code>Home_Page.py</code>
    - <code>pages/</code>:
        - <code>1_EDA_Train_Data.py</code>
        - <code>2_Diagnostics.py</code>    
        - <code>3_Media_Posteriors.py</code> 
        - <code>4_Media_Effects.py</code>
        - <code>5_Optimal_Budget_Allocator.py</code>
        - <code>6_Retrain_with_Customer_Priors.py</code>