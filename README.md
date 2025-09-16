# MIRAGE-VC

This work implements a multi-agent framework for predicting the likelihood that an early-stage start-up will secure a Series-A round. The approach has three pillars:
1. **Path Selector** — Samples informative paths on the VC investment graph and builds task-oriented prompts; collects agent feedback and semantic embeddings to train a lightweight selector.
2. **Weight Generator** — Learns the historical importance of three expert perspectives (Path, Similar-Company, Lead-Investor) and produces a per-sample weight vector.
3. **Inference Pipeline** — Aggregates the three agents’ outputs with the per-sample weights and drives a final decision agent to produce the prediction and analysis.

## Directory Layout
~~~text
VC_LLM_agent_git/
├─ config/                     # Environment & runtime configuration
├─ mirage-vc/                  # Framework / shared utilities
│
├─ vc_data_process/            # VC data preprocessing
│  ├─ vc_graph_builid.ipynb    # Build VC investment graph from raw tables
│  └─ test_data_generate.ipynb # Time/industry sampling to produce final test set
│
├─ path_selector/              # Graph path selector
│  ├─ data_preprocess.ipynb    # Train/val/test sampling & grouping; build task prompts;
│  │                           # call agents for feedback; compute prompt embeddings
│  └─ train.py                 # Train selector and choose the best path per sample
│
├─ weight_generator/           # Importance learning across the three perspectives
│  ├─ data_preprocess.ipynb    # Split data; prepare three agents’ inputs/outputs;
│  │                           # generate company feature vectors
│  └─ train.py                 # Train the weight generator and output per-sample weights
│
└─ inference/                  # End-to-end inference pipeline
   └─ test_data_process.ipynb  # Combine three agents’ I/O and weights; construct the
                               # manager prompt and run final inference
~~~

## Module Notes
1) config/
- Stores environment and runtime settings (models, API keys, paths, concurrency, etc.).

2) mirage-vc/
- Framework code and reusable components: prompt templates, parallel/LLM utilities, helpers.

3) vc_data_process/
- VC graph construction: extract entities (companies, people) and relations (investments, roles) from raw tables to build a multi-edge, time-aware graph.
- Test set filtering: sample targets by time windows and industry rules to produce the final online evaluation set.

4) path_selector/
- Data preprocessing
- Sample and group train/validation/test examples (e.g., 3-hop random branching).
- Build task-oriented prompts for candidate paths; call the path agent for feedback.
- Convert prompts/feedback into semantic embeddings and persist for training.
- Training
- Learn to rank/score candidates within each group and select the best path per target company.

5) weight_generator/
- Data preprocessing
- Create train/validation/test splits.
- Prepare inputs/outputs for the three perspectives:
- Path: best path from the selector and its prompt/feedback;
- Similar-Company: semantically close companies and analyses;
- Lead-Investor: analyses tied to key investors of the target.
- Generate/align company feature vectors (structured attributes, temporal features, etc.).
- Training
- Learn to aggregate the three perspectives into a per-sample weight vector
w_path, w_sim, w_inv.

6) inference/
- Once the path selector and weight generator are trained and the test set is ready:
- Assemble prompts and inputs for the three agents;
- Collect their predictions and analyses;
- Combine them with the per-sample weights to build the Manager Analyst Prompt;
- Call the final decision agent to output a single prediction with an explanation.


## Standard Pipeline
1.	Build the VC investment graph
Run vc_data_process/vc_graph_builid.ipynb to construct the raw investment network (company/person nodes, time-stamped edges).
2.	Generate the test set
Run vc_data_process/test_data_generate.ipynb to sample by time window and industry and obtain the final test targets.
3.	Path-selector data preparation
In path_selector/data_preprocess.ipynb:
- Sample/group train/val/test examples;
- Build task-oriented prompts for candidate paths and collect agent feedback;
- Compute semantic embeddings and assemble the training data.
4.	Train the path selector
Run path_selector/train.py to obtain and save the best path per sample and selector checkpoints.
5.	Weight-generator data preparation
In weight_generator/data_preprocess.ipynb:
- Create train/val/test splits;
- Prepare inputs/outputs for the three perspectives (using the best path from Step 4, similar-company samples, and lead-investor analyses);
- Generate and align company feature vectors.
6.	Train the weight generator
Run weight_generator/train.py to learn the historical importance across perspectives and export w_path, w_sim, w_inv per sample.
7.	End-to-end inference
In inference/test_data_process.ipynb:
- Combine the three agents’ prompts/outputs with the learned weights;
- Construct the Manager Analyst Prompt;
- Invoke the final decision agent and save the prediction and analysis for every test sample.