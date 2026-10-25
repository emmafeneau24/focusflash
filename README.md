# ⚡ FocusFlash
**A virtual consumer focus group in 60 seconds.**

FocusFlash lets a marketing team pre-test a product idea in about a minute, with a panel
of AI personas built from **real customer data**. Each persona reacts to the product, says
whether they would buy it and at what price, and can then be interviewed one-to-one.
The app ends with an actionable marketing summary.

---

## 🎯 The problem it solves
A traditional consumer study or focus group costs thousands of euros and takes weeks.
Many ideas are dropped, or launched blind, because testing them is too slow and expensive.

FocusFlash does not replace real consumer research. It makes it possible to **screen ideas
quickly and for free**, so that teams only invest in a real study for the most promising ones.

**Who it is for:** product managers, marketing teams, start-ups and SMEs without a market
research budget, and students working on product launch projects.

---

## ✨ Features
| Feature | What it does |
|---|---|
| **1. Segments** | Groups ~2,200 real customers into 3 to 6 segments (K-Means) and gives each one a meaningful name, such as *"Affluent · no kids · big spenders"* |
| **2. Panel** | One AI persona per segment reacts to the product: review, purchase intent, maximum price, likes and concerns |
| **A/B test** | Compares two versions of a product: each persona reviews both and states a preference |
| **3. Summary** | Verdict, priority target, recommended price, strengths, risks, recommendations and a slogan; results are weighted by segment size |
| **4. Interview a persona** | Ask any participant your own follow-up questions; they answer in character |
| **Consistency check** | An independent Hugging Face sentiment model scores every review, to check that the LLM's ratings match the tone of what it wrote |

---

## ⚙️ How it works
```
Kaggle customer data ──► scikit-learn K-Means ──► named segments
                                                       │
                                                       ▼
                                  Mistral AI: one persona per segment
                                                       │
                          ┌────────────────────────────┼───────────────────────┐
                          ▼                            ▼                       ▼
           Hugging Face sentiment check     Mistral AI: summary      Mistral AI: interview
```
1. **Data**: the *Customer Personality Analysis* dataset from Kaggle (real customers of a retailer:
   age, income, children, spending by category, online purchases, use of promotions).
2. **Segmentation**: features are standardised, then clustered with K-Means (`random_state=42`,
   so results are reproducible). Segment names are computed from each segment's profile compared
   to the whole customer base (income tertiles, children, promotion use, online share, spending, age).
3. **Personas**: each segment's statistical profile is sent to a Mistral LLM, which invents a
   credible consumer and reacts to the product, in a structured JSON format.
4. **Verification**: each review is scored from 1 to 5 stars by a pre-trained Hugging Face
   sentiment model, independently of the LLM.
5. **Summary and interview**: the LLM synthesises the panel's reactions, and can play any
   persona in a one-to-one interview.

---

## 🛠️ Installation
**Requirements:** Python 3.10 or later (tested with Python 3.13), and an internet connection.

1. **Get the code**
   ```
   git clone https://github.com/<your-username>/focusflash.git
   cd focusflash
   ```
   (or download the repository as a ZIP file from GitHub and unzip it)

2. **Install the dependencies**
   ```
   python -m pip install -r requirements.txt
   ```
   On Windows, `py` can be used instead of `python`. The installation takes a few minutes
   (PyTorch is large). Using a virtual environment is recommended.

3. **Data**: the dataset is already included in `data/marketing_campaign.csv`.
   If it is missing, download it from
   [Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)
   and place it in the `data/` folder, or let the app download it automatically with
   `kagglehub` (a Kaggle API key may be required).
   ⚠️ The file is tab-separated: do not open and re-save it with Excel.

4. **Get a free Mistral API key**
   - Create an account on [console.mistral.ai](https://console.mistral.ai).
   - In *Admin > Subscriptions*, choose the free **Experiment** plan ("Experiment for free").
     No credit card is needed; phone verification is required.
   - Go to *API Keys* > *Create new key* and copy the key (it is only shown once).
   - Keep the key private: never put it in the code or on GitHub.

---

## ▶️ Running the app
```
python -m streamlit run app.py
```
The app opens in your browser at http://localhost:8501. Keep the terminal open while using it.

The first launch takes longer: the libraries load, and the first panel downloads the
Hugging Face model (about 700 MB, only once).

---

## 📖 User guide
1. **Paste your Mistral API key** in the sidebar. You can also choose the number of segments (3 to 6).
2. **Tab 1 – Segments**: explore the customer segments, their size and profile.
3. **Tab 2 – Panel**: describe your product and its price, then click **Run the panel**.
   To compare two versions, switch on **Compare two versions (A/B test)** and fill in both.
   A panel takes about one minute.
4. **Tab 3 – Summary**: read the verdict, the weighted purchase intent, the recommended price,
   the priority target and the recommendations. In A/B mode, see which version wins and with whom.
5. **Tab 4 – Interview a persona**: choose a participant, then click a quick question or type
   your own.

**Tip:** test a product that should work and one that should flop
(e.g. *"a caviar yoghurt at €15 a pot"*) to see how the panel reacts.

---

## 🧯 Troubleshooting
| Problem | Solution |
|---|---|
| `Could not open requirements file` | The terminal is not in the project folder; on Windows, also check that the file is not called `requirements.txt.txt` |
| `command not found: streamlit` | Use `python -m streamlit run app.py` |
| `No module named 'torchvision'` | Run `python -m pip install torchvision` |
| `429 Too Many Requests` | The free plan limits the request rate. The app already pauses and retries automatically; if it persists, wait a minute. On our account, `mistral-small-latest` was rate-limited, which is why the app uses `open-mistral-nemo` |
| `Operation not permitted` (macOS) | Move the project out of Desktop/Documents/Downloads, or allow Terminal access in *System Settings > Privacy & Security > Files and Folders* |
| Python version too old | Install Python 3.10+ (or create an environment with `conda create -n focusflash python=3.11`) |

---

## 📁 Project structure
```
focusflash/
├── app.py                    # the whole application
├── requirements.txt          # Python libraries
├── README.md                 # this guide
├── prompts.md                # prompts used, in the app and with AI assistants
├── .gitignore                # files not sent to GitHub
├── .streamlit/
│   └── config.toml           # colour theme
└── data/
    └── marketing_campaign.csv
```

---

## 📚 Sources and tools
**Data**
- *Customer Personality Analysis*, Akash Patel (imakash3011), Kaggle:
  https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis

**Models**
- LLM: `open-mistral-nemo`, served by Mistral AI as `ministral-8b`, via the Mistral API: https://docs.mistral.ai
- Sentiment model: `nlptown/bert-base-multilingual-uncased-sentiment`, Hugging Face:
  https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment

**Python libraries**
- [Streamlit](https://github.com/streamlit/streamlit) (interface)
- [pandas](https://github.com/pandas-dev/pandas) (data processing)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn) (StandardScaler, K-Means)
- [Plotly](https://github.com/plotly/plotly.py) (charts)
- [Transformers](https://github.com/huggingface/transformers), [PyTorch](https://github.com/pytorch/pytorch) and torchvision (Hugging Face model)
- [Requests](https://github.com/psf/requests) (Mistral API calls)
- [kagglehub](https://github.com/Kaggle/kagglehub) (optional automatic data download)

**AI assistants**
Claude (Anthropic) was used to design and code the app. All prompts, in the app and with
the assistant, are listed in [`prompts.md`](prompts.md).

---

## ⚠️ Limitations and ethics
- **Simulated opinions**: the personas are generated by an LLM. They can reflect the model's
  biases and stereotypes, and are no substitute for real consumers. FocusFlash is a
  pre-testing tool, to be followed by real research for important decisions.
- **Data**: the customer data dates from around 2014 and covers US customers; spending is
  in dollars while prices are tested in euros. Results should be read as indications.
- **One persona per segment**: each segment is represented by a single persona, which
  simplifies the diversity within a segment.
- **Model**: a small LLM was used because of the free plan's limits. A larger model would give
  richer answers.
- **Privacy**: on Mistral's free Experiment plan, requests may be used to improve Mistral's models.
  Do not enter confidential product information.

---

## 🚀 Value at scale
If widely used, FocusFlash would let any company, including those without a research budget,
test dozens of ideas, prices or packaging options before investing. Teams could drop weak
ideas earlier, spend their research budget only on the most promising concepts, and bring
the customer's voice into decisions from day one.
