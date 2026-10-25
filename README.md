# ⚡ FocusFlash
A virtual consumer focus group in 60 seconds.

## What it does
A traditional consumer study is expensive and takes weeks. FocusFlash lets a
marketing team pre-test a product idea in one minute, with a panel of virtual
personas built from real customer data.

## How it works
1. **Real data**: about 2,200 customers from a Kaggle dataset
2. **Segmentation**: customers are grouped into segments with K-Means (scikit-learn)
3. **Personas**: each segment is embodied by a virtual consumer (Mistral AI)
4. **Verification**: an independent sentiment model checks the reviews (Hugging Face)
5. **Summary**: verdict, priority target, recommended price and recommendations

## Installation
1. Install Python 3.10 or later.
2. Install the dependencies from the project folder:
   ```
   python -m pip install -r requirements.txt
   ```
   (on Windows, `py` can be used instead of `python`)
3. Data: download `marketing_campaign.csv` from Kaggle
   (https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)
   and place it in the `data/` folder.
   If the file is missing, the app tries to download it automatically with `kagglehub`
   (a Kaggle API key may be required).
4. Create a free API key at https://console.mistral.ai.

## Running the app
```
python -m streamlit run app.py
```
The app opens in your browser. The first time the panel is run, the Hugging Face
model is downloaded (a few minutes).

## How to use it
1. Paste your Mistral API key in the sidebar.
2. "Les segments" tab: explore the customer segments.
3. "Le panel" tab: describe the product, enter a price, and click "Lancer le panel".
4. "La synthèse" tab: read the verdict, the priority target and the recommendations.

## Project structure
```
focusflash/
├── app.py              # application code
├── requirements.txt    # Python libraries
├── README.md           # this guide
├── prompts.md          # prompts used (AI tools)
└── data/
    └── marketing_campaign.csv
```

## Sources
- Data: *Customer Personality Analysis*, Kaggle (imakash3011)
- Sentiment model: `nlptown/bert-base-multilingual-uncased-sentiment`, Hugging Face
- LLM: `mistral-small-latest`, Mistral AI
- Libraries: Streamlit, pandas, scikit-learn, Plotly, Transformers, PyTorch, kagglehub

## Limitations
The personas are simulated by an LLM and may reflect its biases. The customer data
dates from around 2014 and covers US customers.
FocusFlash is a pre-testing tool, not a substitute for a real consumer study.