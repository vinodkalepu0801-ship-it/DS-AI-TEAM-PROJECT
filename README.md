Install requirements
pip install -r requirements.txt
The first run with Sentence-BERT may download the model sentence-transformers/all-MiniLM-L6-v2. Use internet access for the first run unless the model is already cached.

Run the full project with one command
python run_project.py
By default, it uses all candidates and the first 25 job descriptions. This prevents the full candidate-job matrix from becoming too slow during demonstration.

Start the application
streamlit run streamlit_inference_app.py
(This should be run after the run_project.py is performed least once )

#REQUIREMENTS
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
sentence-transformers>=2.2.2
streamlit

