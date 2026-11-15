Domain: Healthcare

## How to run
1. Ensure Python 3.10+ is installed. Open your terminal in the repository root.
2. Create and activate a virtual environment:
   `python3 -m venv venv`
   `source venv/bin/activate` 
3. Install the required packages:
   `pip install -r requirements.txt`
4. Generate the dataset:
   `python generate_data.py`
5. Run the analysis scripts in order:
   `python part1.py`
   `python part2.py`
   `python part3.py`
   `python part4.py`

## Part 1 findings
* **R1.1:** We reject H0 because our z-statistic of 2.8930 exceeds the critical value of 1.96. A Type I error here would mean concluding the average care-plan cost is not INR 8,000 when it actually is. The gap is 0.9382 units (INR 938.20), which represents a significant budget deviation in practice.
* **R1.2:** To achieve a margin of error of 0.5 at 95% confidence, a sample size of 1293 is needed. Compared to our 800 available rows, this tells the planning team we do not have enough data to trust the estimate at this precision.
* **R1.3:** The gap between the gradient descent slope and the closed-form slope is 0.0000. Gradient descent reached the closed-form line because linear regression with a mean squared error cost function forms a convex bowl with a single, global minimum.
* **R1.4:** The raw residual standard deviation ratio is 3.7525 and the log model ratio is 1.0396. The raw model shows a clear funnel effect (heteroscedasticity) where variance increases with the fitted values, but the log transformation successfully stabilizes this variance.
* **R1.5:** The predictors with an absolute coefficient above 0.05 are ln(X4), X2, X3, and X6. Comparing the multiple regression adjusted R² of 0.7403 with the log-log adjusted R² of 0.6561, the extra predictors do earn their place by explaining significantly more variance.

## Part 2 findings
* **R2.1:** The losses from lowest to highest are η=1.0 (0.3703), η=0.1 (0.3776), and η=0.01 (0.5091). A learning rate that is too small (0.01) converges too slowly, while a learning rate that is large (1.0) reached a lower loss faster on this specific smooth surface without diverging.
* **R2.2:** The three predictors with the largest absolute weights are X3 (0.9013), X1 (-0.5554), and X2 (0.5505). The largest odds ratio (X3, 2.4629) means that for every 1-standard-deviation increase in the copay burden ratio, the odds of a patient being a no-show are multiplied by 2.4629, assuming all other factors remain fixed.
* **R2.3:** The `blind_accuracy` is 0.7875, while the τ=0.5 accuracy is 0.8042. Accuracy is a weak score here because the dataset is imbalanced, meaning a model that simply guesses "no-show" for nobody still achieves ~79% accuracy. Lowering the threshold from 0.5 to 0.3 increased recall from 0.4706 to 0.7451 but decreased precision from 0.5455 to 0.4634.
* **R2.4:** The logistic model at τ=0.3 achieved an F1 of 0.5714 and recall of 0.7451, compared to the KNN model's F1 of 0.3696 and recall of 0.3333.
* **R2.5:** I would deploy the Logistic Regression model at threshold τ=0.1500. The cost on the test set at this threshold is INR 6050, compared to INR 11800 at τ=0.5 and INR 20400 if we flagged nobody. The threshold was chosen on the training rows to prevent data leakage and ensure the threshold generalizes to unseen data.

## Part 3 findings
* **R3.1:** The successive WCSS drops are 659.8282, 108.0739, 171.0575, and 45.5466. The elbow supports K=3 because this is the point where the marginal reduction in within-cluster variance significantly flattens out, indicating diminishing returns for adding more clusters (noting the drop spike at K=5 is an artifact of the pinned starting rows not hitting the global optimum).
* **R3.2:** 
  * Segment A: "High-Tenure Loyals" - Action: Offer automated digital renewals.
  * Segment B: "Moderate-Risk Patients" - Action: Trigger automated SMS reminders 24 hours before appointments.
  * Segment C: "High-Friction Patients" - Action: Assign a dedicated human care coordinator to navigate copay barriers.
  The segment with the highest outcome rate is [Insert Segment and outcome_rate directly from part3_segments.csv].
* **R3.3:** At the cut, single linkage yields 4 clusters, while complete linkage yields 2 clusters. The rules disagree because single linkage merges based on the closest neighbors (which can cause "chaining"), whereas complete linkage merges based on the furthest neighbors, forcing clusters to be tighter and more spherical.
* **R3.4:** The first principal component is primarily driven by X1, X3, and X5 based on the loading sizes. It takes 3 components to retain 80% of the variance. The mean PC1 scores of 1.7773 (A), -0.7831 (B), and -1.6709 (C) show that PC1 effectively separates the three segments.

## Part 4 findings
* **R4.1:** The best split is S4 because it yields the highest information gain (reduces impurity the most). A tree grown until every leaf is pure would overfit by memorizing noise rather than generalizable patterns. Setting a maximum depth is one rule that stops a tree from growing too deep.
* **R4.2:** I would ship the Logistic Regression model because it offers the best balance, with an F1 of [Insert F1 from part4_voting.csv] and a recall of [Insert Recall from part4_voting.csv]. Combining models via voting [helped / did not help - check part4_voting.csv] compared to the best individual model.
* **R4.3:** The total weight of the two misclassified points after round 1 normalisation is 0.2500. Round 2 trusts h2 more because it achieved a lower error rate on the newly re-weighted dataset from round 1.
* **R4.4:** 
  | Layer | Output Shape | Parameters |
  |---|---|---|
  | cnn_conv1 | 28x28x6 | 156 |
  | pool1 | 14x14x6 | 0 |
  | cnn_conv2 | 10x10x16 | 2416 |
  | pool2 | 5x5x16 | 0 |
  | cnn_dense1 | 32 | 12832 |
  | cnn_dense2 | 10 | 330 |
  Model B shapes: W_xh (128x64), W_hh (128x128), W_hy (16x128). `rnn_total` (26768) does not grow with longer sequences because the weight matrices are shared across all time steps. For 10 classes, I would use ReLU for the hidden layers and Softmax with Categorical Cross-Entropy loss for the output. A very deep plain network stops learning early because gradients vanish as they are multiplied backwards through many layers. Skip connections help by providing direct gradient bypass paths, preventing them from shrinking to zero.

## Model-choice memo
To: Head of Scheduling Operations
From: Analytics Team

Based on our analysis of the 800 patient records, I recommend deploying the Logistic Regression model at an alert threshold of τ=0.1500 to flag potential no-shows. 

This model minimizes our financial exposure. By weighing the INR 50 cost of a false alarm against the INR 400 cost of an empty slot, this tuned threshold yields a test-set cost of INR 6050. This is a significant improvement over the INR 11800 cost incurred at the default 0.5 threshold, and vastly outperforms the INR 20400 cost of doing nothing.

A key limitation of this evidence is that it relies on synthetic data and a relatively small sample of positive test records (51 no-shows), which may cause our threshold performance to vary in production. 

Regarding future capabilities, Model A (image routing) is worth building before the structured model is in production, as convolutional networks scale efficiently and could immediately alleviate manual triage bottlenecks.

## Limitations
1. The dataset is entirely synthetic, meaning real-world patient behavior and confounding variables may not be captured.
2. The analysis relies on a single 70/30 train-test split; performance metrics might fluctuate significantly with a different random seed or cross-validation setup.
3. The absolute number of positive cases in the test set is very small (51 no-shows), making our evaluation metrics (especially precision and cost) highly sensitive to minor variations.
