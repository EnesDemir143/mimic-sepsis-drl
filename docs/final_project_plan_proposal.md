# Offline Reinforcement Learning for Sepsis Treatment using Conservative Q-Learning on MIMIC-IV

## 1. Project Motivation and Scope

This project studies offline reinforcement learning on a realistic sequential decision-making dataset from critical care. The main objective is to implement and evaluate Conservative Q-Learning (CQL) for a discrete-action offline RL problem using the MIMIC-IV v3.1 sepsis cohort.

The project is framed as a Computer Engineering / Reinforcement Learning methodology project, not as a clinical deployment study. Sepsis treatment is used because it provides a challenging benchmark setting for offline RL: decisions are sequential, patient trajectories are variable length, actions are partially supported by observational clinician behavior, and rewards are delayed and uncertain.

The project does not claim that the learned policy is clinically valid or suitable for real treatment guidance. Instead, the focus is on:

- constructing a valid offline RL problem from retrospective ICU data,
- training one main offline RL algorithm, Conservative Q-Learning,
- comparing shaped and sparse training reward designs,
- evaluating learned policies using retrospective off-policy evaluation,
- auditing temporal alignment, leakage, and behavior-policy support,
- reporting limitations clearly.

The main research question is:

> Can a discrete-action CQL agent be trained and evaluated consistently on a MIMIC-IV Sepsis-3 cohort under a carefully controlled offline RL protocol?

## 2. Problem Formulation as an MDP

The sepsis treatment task is formulated as a finite-horizon Markov Decision Process:

\[
M = (S, A, P, R, \gamma)
\]

where:

- \(S\) is the patient state space.
- \(A\) is the discrete treatment action space.
- \(P\) is the unknown transition dynamics induced by patient physiology and treatment effects.
- \(R\) is the reward function used for training or evaluation.
- \(\gamma\) is the discount factor.

Each episode corresponds to one ICU patient trajectory around estimated sepsis onset. The timestep is fixed to 4-hour intervals, and the analysis window is from 24 hours before onset to 48 hours after onset. Therefore, each episode has approximately 18 decision points, although episode lengths may vary depending on patient data availability.

### State \(s_t\)

Each state consists of 62 continuous clinical features, including:

- vital signs,
- laboratory measurements,
- ventilation-related variables,
- demographic variables,
- derived clinical features such as PF ratio, shock index, and hours since onset,
- missingness indicators.

The state representation is designed for retrospective modeling only. It is not assumed to be a complete representation of the patient’s true physiological state.

### Action \(a_t\)

The action space is a 5×5 discrete grid, giving 25 possible actions. The two action dimensions are:

- vasopressor dose bin,
- intravenous fluid amount bin.

Action bin thresholds are learned only from the training split and then applied unchanged to validation and test splits. This avoids using validation or test distribution information when defining the action space.

### Transition

Each transition is represented as:

\[
(s_t, a_t, r_t, s_{t+1}, done)
\]

where \(a_t\) corresponds to the treatment action in the correct 4-hour decision window, and \(s_{t+1}\) is the post-action next state.

### Reward

Two reward variants are used for training:

1. **SOFA-shaped reward**
   - intermediate reward based on change in SOFA score,
   - terminal reward of +15 for survival and -15 for death.

2. **Sparse reward**
   - no intermediate reward,
   - terminal reward of +15 for survival and -15 for death.

For final model selection and final reporting, both reward-trained policies are evaluated under one shared terminal survival/death utility. This prevents shaped and sparse models from being selected using incompatible reward scales.

### Discount Factor

The discount factor is fixed at \(\gamma = 0.99\). This is treated as an MDP design choice rather than a normal optimizer hyperparameter. With approximately 18 timesteps per episode, \(\gamma = 0.99\) preserves:

\[
0.99^{18} \approx 0.83
\]

of the terminal reward weight. This ensures that delayed survival/death outcomes remain influential. By comparison:

\[
0.95^{18} \approx 0.40
\]

\[
0.90^{18} \approx 0.15
\]

so lower discount factors would substantially downweight delayed terminal outcomes. Therefore, \(\gamma\) is fixed and not swept.

## 3. Dataset and Preprocessing Summary

### Dataset

The project uses the MIMIC-IV v3.1 database, restricted to adult ICU patients satisfying Sepsis-3 cohort criteria.

### Cohort

The cohort consists of adult ICU Sepsis-3 patients. Episodes are anchored around estimated sepsis onset and discretized into 4-hour intervals from onset -24h to onset +48h.

### Splits

The data are split at the patient level into:

- training set,
- validation set,
- test set.

No patient appears in more than one split. This is essential because ICU trajectories from the same patient are highly correlated, and patient-level leakage would invalidate evaluation.

### Preprocessing

The preprocessing pipeline includes:

- feature construction,
- missingness handling,
- imputation,
- scaling,
- action bin construction,
- replay buffer construction.

All imputation, scaling, and action-bin statistics must be fit only on the training split. Validation and test data must use the frozen preprocessing statistics learned from training.

### Replay Datasets

The shaped replay dataset has already been built. The sparse replay dataset still needs to be built before Stage 1. Before training, shaped and sparse replay datasets must be checked to ensure that they share:

- identical patient split definitions,
- identical feature preprocessing,
- identical action bins,
- identical transition indexing,
- identical terminal outcome definitions,
- different training reward definitions only.

## 4. Method: Conservative Q-Learning

The main algorithm is discrete-action Conservative Q-Learning (CQL).

CQL is suitable for offline RL because it explicitly penalizes overestimated Q-values for actions that are not well supported by the behavior policy. This is important in retrospective healthcare datasets, where many treatment actions may be rare or absent for particular patient states.

The learned Q-function is represented by a two-layer ReLU MLP:

- input: 62-dimensional state vector,
- hidden layers: [256, 256],
- activation: ReLU,
- output: Q-values for 25 discrete actions.

The policy selects actions by:

\[
\pi(s) = \arg\max_a Q(s, a)
\]

The CQL objective combines a standard temporal-difference learning loss with a conservative regularization penalty. In discrete-action form, the conservative term encourages the Q-values of out-of-distribution or weakly supported actions to be lower than actions observed in the dataset. The CQL alpha parameter controls the strength of this conservative penalty.

The project intentionally focuses on CQL only. Other offline RL methods may be briefly mentioned in related work, but they are not part of the main experimental protocol.

## 5. Hyperparameter and Fixed Design Choices

### Fixed Design Choices

- Algorithm: discrete-action Conservative Q-Learning.
- Network: 2-layer ReLU MLP.
- Hidden sizes: [256, 256].
- Optimizer: Adam.
- Batch size: 256.
- Epochs: up to 200.
- Discount factor: \(\gamma = 0.99\).
- Target update: hard target update every 10 epochs.
- Gradient clipping: 10.0.
- Action space: 25 discrete actions from 5 vasopressor bins × 5 IV fluid bins.
- No optimizer sweep.
- No gamma sweep.
- No architecture sweep.

### Final Hyperparameter Sweep

Reward variant:

- shaped,
- sparse.

Learning rate:

- 1e-4,
- 3e-4,
- 1e-3.

CQL alpha:

- 0.05,
- 0.1,
- 0.5,
- 1.0.

This gives:

\[
2 \text{ reward variants} \times 3 \text{ learning rates} \times 4 \text{ alpha values} = 24 \text{ Stage 1 configurations}
\]

### Justification

The sweep is intentionally limited and realistic for a course project. Reward variant is a central design choice because it changes the training signal. Learning rate is important for neural Q-learning stability. CQL alpha is the main CQL-specific hyperparameter because it controls the strength of conservative regularization.

Alpha 0.05 is included because offline CQL studies in sepsis-like settings have reported useful performance with conservative penalty values below 0.1. Larger values are included to test stronger conservatism.

Gamma is not swept because it defines the temporal weighting of the MDP objective. Given the short finite-horizon setting of approximately 18 timesteps, \(\gamma = 0.99\) is a defensible fixed design choice.

## 6. Two-Stage Model Selection Protocol

The project uses a two-stage protocol to separate broad exploratory screening from multi-seed confirmation.

### Stage 0 — Pre-sweep Audit

Before any CQL sweep, the following steps will be completed:

1. Build the sparse replay dataset.
2. Verify that shaped and sparse replay datasets use the same:
   - patient splits,
   - feature preprocessing,
   - action bins,
   - transition indexing,
   - terminal outcomes.
3. Confirm that action bins are fit only on the training split.
4. Confirm that imputation and scaling statistics are fit only on the training split.
5. Verify that validation FQE uses one common terminal survival/death evaluation reward.
6. Run the temporal alignment and leakage audit.
7. Confirm that no patient appears in multiple splits.
8. Confirm that mortality or discharge outcome variables do not leak into state features.

### Stage 1 — Broad Screening

Purpose:
Stage 1 is a single-seed exploratory screening phase. It is used to identify promising configurations, not to claim confirmed performance.

Protocol:

- Seed: 42.
- Configurations: 24 CQL runs.
- Each run trained for up to 200 epochs.
- Checkpoints saved during training.
- Each checkpoint evaluated on validation FQE under the shared terminal evaluation reward.
- The best checkpoint for each run is selected by validation FQE, not by final epoch.
- WIS, ESS, behavior-support mass, and low-support action rate are used only as diagnostics.
- The top 6 configurations are selected for Stage 2.

Important reporting rule:
Stage 1 results must be labeled as exploratory screening only. They should not be interpreted as confirmed performance.

### Stage 2 — Multi-seed Confirmation

Purpose:
Stage 2 confirms the most promising configurations under seed variability.

Protocol:

- Configurations: top 6 from Stage 1.
- Seeds: 42, 123, 456, 789, 1024.
- Total: 5 seeds per selected configuration. If the Stage 1 seed-42 run for a selected configuration was trained and evaluated using the same checkpoint-selection protocol, it will be reused as one of the five Stage 2 seeds. Otherwise, it will be rerun under the Stage 2 protocol.
- Each run trained for up to 200 epochs.
- Checkpoints evaluated on validation FQE under the shared terminal evaluation reward.
- Best checkpoint selected by validation FQE for each seed.
- Report:
  - mean validation FQE,
  - standard deviation across seeds,
  - patient-level bootstrap 95% confidence intervals,
  - WIS,
  - ESS,
  - behavior-support diagnostics,
  - low-support action rate.

Final selection:
The final selected CQL configuration is chosen using validation results only. The held-out test set is not used during configuration selection, checkpoint selection, or model debugging.

## 7. Evaluation Protocol

### Primary Evaluation

The primary metric is Fitted Q Evaluation (FQE), computed using the common terminal survival/death utility. Patient-level bootstrap 95% confidence intervals will be reported.

The key methodological correction is that shaped-trained and sparse-trained models must not be selected under different reward scales. Reward variant is a training design choice; final validation and test evaluation use the same terminal utility.

### Secondary Evaluation

The project will also report:

- Weighted Importance Sampling (WIS),
- Effective Sample Size (ESS),
- behavior-support mass,
- low-support action rate,
- clinician exact-bin agreement,
- clinician adjacent-bin agreement,
- clinician vs CQL action heatmaps.

For WIS and ESS computation, behavior policy probabilities will be estimated from the training data using the implemented behavior cloning model or empirical action frequencies conditioned on the discretized state/action representation. WIS will be treated as a diagnostic metric because it can have high variance when the learned policy differs substantially from clinician behavior.

### Final Test Evaluation

The held-out test set is used only once after final model selection. The final test evaluates the selected CQL policy using:

- FQE with patient-level bootstrap 95% CI,
- WIS,
- ESS,
- behavior-support mass,
- low-support action rate,
- exact-bin clinician agreement,
- adjacent-bin clinician agreement,
- action heatmaps.

All test results are interpreted as retrospective off-policy estimates, not clinical proof.

## 8. Leakage and Temporal Alignment Controls

Before final evaluation, the pipeline will be audited for temporal consistency and data leakage.

### Temporal Alignment Checks

- \(s_t\) contains only information available before the corresponding treatment action.
- \(a_t\) corresponds to the treatment action in the correct 4-hour decision window.
- \(s_{t+1}\) is the post-action next state.
- \(r_t\) is assigned to the correct transition.
- Terminal reward is assigned only at terminal transitions.
- Intermediate SOFA-shaped reward is aligned with the correct change in patient state.

### Leakage Controls

- Action bins are fit only on the training split.
- Imputation statistics are fit only on the training split.
- Scaling statistics are fit only on the training split.
- No patient appears in multiple splits.
- Validation and test preprocessing use frozen training statistics.
- Mortality, discharge status, or future outcome indicators do not appear in state features.
- The sparse and shaped replay datasets differ only in reward definition, not in splits or preprocessing.

These checks are essential because offline RL is highly sensitive to temporal leakage. Even small leakage can cause inflated OPE estimates and misleading policy comparisons.

## 9. Expected Figures and Tables

### Expected Figures

1. **Cohort flow diagram**
   - shows cohort construction from MIMIC-IV to final Sepsis-3 ICU episodes.

2. **Clinician vs CQL action heatmap**
   - compares action frequencies over the 5×5 vasopressor/fluid grid.

3. **CQL training curves**
   - TD loss,
   - CQL loss,
   - conservative gap,
   - Q-value trends.

4. **Episode reward curves**
   - shaped and sparse training reward trends.

5. **Support diagnostics**
   - behavior-support mass histogram,
   - low-support action rate.

6. **Clinician agreement**
   - exact-bin agreement,
   - adjacent-bin agreement,
   - optionally stratified by severity.

7. **Reward decomposition example**
   - one patient trajectory showing SOFA trajectory and shaped/sparse reward signals.

8. **Learning rate × alpha comparison**
   - validation FQE ± bootstrap confidence interval.

### Expected Tables

1. **Stage 1 screening results table**
   - reward variant,
   - learning rate,
   - CQL alpha,
   - best validation FQE,
   - WIS,
   - ESS,
   - support diagnostics.

2. **Stage 2 multi-seed confirmation table**
   - selected configurations,
   - mean validation FQE,
   - standard deviation,
   - bootstrap 95% CI,
   - support diagnostics.

3. **Final test results table**
   - selected CQL policy only,
   - FQE,
   - WIS,
   - ESS,
   - support diagnostics,
   - clinician agreement.

4. **Cohort characteristics table**, if space allows
   - stratified by mortality.

5. **Clinician action distribution table**, if space allows
   - 5×5 action counts and percentages.

## 10. Limitations and Ethical Considerations

### POMDP and Observability Limitations

Although the project is formulated as an MDP, the true clinical decision process is partially observable. Important variables may be missing, delayed, noisy, or unmeasured. Examples include physician judgment, bedside observations, latent severity, treatment intent, contraindications, and hospital-specific practice patterns. Therefore, the learned state representation should be understood as an approximation, not a complete patient state.

### Confounding

The dataset is retrospective and observational. Clinician actions are not randomly assigned. Treatment decisions are influenced by severity, clinical judgment, and unmeasured factors. Therefore, the learned policy may reflect confounding patterns in the data.

### Support Mismatch

Offline RL policies may select actions that are rare or unsupported in the behavior data. CQL is used specifically to reduce overestimation of unsupported actions, but this does not eliminate support mismatch. Behavior-support mass and low-support action rate will be reported to identify policies that appear poorly supported by the dataset.

### OPE Uncertainty

FQE and WIS are retrospective off-policy evaluation estimates. They are not equivalent to prospective clinical validation. WIS can have high variance when learned policies differ from clinician behavior, and FQE depends on function approximation assumptions.

### No Clinical Deployment Claim

The project does not claim that the learned CQL policy can guide real treatment. The results are only valid as a retrospective offline RL experiment under the assumptions of the constructed dataset and evaluation protocol.

### Ethics and Data Access

The project uses MIMIC-IV through PhysioNet access procedures, including credentialing, CITI training, and the required Data Use Agreement. Patient data are de-identified. The project is for academic research and course evaluation only.

## 11. Final Execution Plan

The final project will proceed as follows:

1. **Complete Stage 0 audit**
   - build sparse replay dataset,
   - verify shaped/sparse replay consistency,
   - verify common terminal evaluation reward,
   - audit temporal alignment,
   - audit train-only preprocessing,
   - audit patient-level splits,
   - audit leakage from future outcome variables.

2. **Run Stage 1 screening**
   - train 24 CQL configurations with seed 42,
   - save checkpoints,
   - evaluate checkpoints using validation FQE under common terminal utility,
   - select the best checkpoint per run,
   - report Stage 1 as exploratory only,
   - choose top 6 configurations.

3. **Run Stage 2 confirmation**
   - train top 6 configurations using seeds 42, 123, 456, 789, and 1024,
   - select best checkpoint per run by validation FQE,
   - report mean validation FQE,
   - report seed variability,
   - compute patient-level bootstrap 95% confidence intervals,
   - use support diagnostics to identify unsupported policies.

4. **Select final model**
   - choose final configuration using validation results only,
   - do not inspect test results during selection.

5. **Run final test evaluation once**
   - evaluate selected CQL policy,
   - report FQE, WIS, ESS, support diagnostics, clinician agreement, and action heatmaps.

6. **Prepare final report**
   - emphasize CQL as the main method,
   - clearly separate Stage 1 screening from Stage 2 confirmation,
   - include gamma justification,
   - discuss POMDP and observability limitations,
   - include ethics and data access,
   - include at least 10 IEEE-style references,
   - avoid clinical deployment claims.

---

This project is designed as an offline RL methodology study. The learned policies are evaluated retrospectively using OPE and should not be interpreted as clinically validated treatment recommendations.
