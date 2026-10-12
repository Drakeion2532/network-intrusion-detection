# Hands-on labs

Six short labs that take you from raw NSL-KDD rows to an explainable detector you can attack.
Run every lab from the repository root, e.g. `python -m labs.lab1_explore`.

| Lab | Script | You learn |
|---|---|---|
| 0 | – | Set up Python, clone, install, run tests (Windows PowerShell) |
| 1 | `labs.lab1_explore` | What a network flow is, the 5 attack categories, why the test set is hard |
| 2 | `labs.lab2_first_model [threshold]` | Train a Random Forest, read a confusion matrix, trade detection vs false alarms |
| 3 | `labs.lab3_zero_day [percentile]` | Why supervised models miss new attacks, and how an anomaly detector helps |
| 4 | `labs.lab4_explain [attack]` | Explain any alert (or miss) with SHAP |
| 5 | `labs.lab5_api_client` | Call the API like a SOC tool, then try to evade it |
| 6 | – | Open challenge: improve the detector and defend your result |
