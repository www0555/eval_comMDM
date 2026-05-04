# Evaluation Pipeline for comMDM

This README explains how to reproduce the evaluation results for **comMDM** using a two-stage pipeline:
1. Generate motion results using the PriorMDM environment.
2. Evaluate the generated results using the InterGen (InterCLIP) evaluation framework.

---

## 1. Generate Results with comMDM

### PriorMDM Environment (for Generation)

First, set up the environment following the official PriorMDM project:

- 🔗 https://priormdm.github.io/priorMDM-page/

Make sure all dependencies are correctly installed before proceeding.
 

### Data and Model Paths

- Data directory:

./generate_result_comMDM/dataset/3dpw


- Model checkpoint path:

./generate_result_comMDM/save/modal/model000200000.pt

- SMPL models path:
./generate_result_comMDM/body_models/smpl


### Run Generation

Navigate to the generation directory and run:

```bash
cd ./generate_result_comMDM

python -m eval.generate_eval_results \
--model_path /data/cluster/www/eval_comMDM/generate_result_comMDM/save/modal/model.pt \
--batch_size 4
```

This step will generate motion results using the trained comMDM model.


## 2. Evaluation with InterGen (InterCLIP)
Environment Setup

Switch to the InterGen evaluation environment and follow the setup instructions:

🔗 https://tr3e.github.io/intergen-page/

Ensure that all dependencies and pretrained models required for evaluation are properly configured.

Run Evaluation

Navigate to the evaluation directory and execute:

```bash
cd ./eval_interclip

python -m tools.eval_comMDM

```
This will compute evaluation metrics based on the generated results.
