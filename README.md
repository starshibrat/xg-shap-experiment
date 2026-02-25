What's this repository about?
===
XGBoost and SHAP already has their own dedicated library and both are integrated nicely. This repository is not exists in any means to replace but to learn on how they work by learning the algorithm. Also, I'm writing it for my undergraduate thesis. Hope everyone who found this repository helpful.


Directory and it's purpose
===

**mini_shap** : Tree Explainer mostly\
**tree** : Decision tree and Mini XGBoostClassifier\
**notebook** : Jupyter notebook for experiment and presentational purpose\
**objective** : Loss, gradient, and hessian logic/function\
**common** : Commonly used function (ex: sigmoid)

----
To be implemented
===
Status\
🟢 = done\
🔶 = on progress\
🔴 = halt\
🔨 = prototype (not sure done with it or no)

| Feature | Priority | Status | Last Update |
| --- | --- | --- | --- |
| Full/halfly Vectorized Tree SHAP | High | 🔨 | 2026.02.25 |
| Parallelize/Vectorize Tree Split | High | 🔴 | - |
| More Objective Function | Medium | 🔴 | - |
| More Parameter to Tweak | Low | 🔴 | - |
| Evaluation function | Medium | 🔴 | - |
| Sparsity aware split finding | Medium | 🔴 | - |
---
Development notes
===
2026.02.25 Faster tree explainer
---
The algorithm is still the same but using vectorization more than the former. The benchmark is not there yet, but I believe it's 50-100x faster with sacrificing the memory obviously. For now I put prototype status.