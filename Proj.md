## Early Prevention of Diabetes Using Explainable Machine Learning

## Abstract
Diabetes, a prominent lifestyle disease, presents a significant challenge to global public health, with a rising number of individuals affected worldwide. The integration of machine learning models into healthcare offers a promising avenue for improving early detection and intervention strategies. This study investigates the effectiveness of three machine learning models Random Forest (RF), Logistic Regression (LR), and Support Vector Machine (SVM) in predicting diabetes using a dataset of 520 samples with 16 features. The primary focus is on enhancing model interpretability through post-hoc explainability techniques such as SHAP (Shapley Additive Explanations) and LIME (Local Interpretable Model-agnostic Explanations), which provide valuable insights into the factors driving model predictions.
To address class imbalance, the dataset undergoes preprocessing using the Synthetic Minority Over-sampling Technique (SMOTE), ensuring a balanced distribution of classes. The models are evaluated based on metrics such as AUC, F1 score, sensitivity, and specificity. The Random Forest model emerges as the most effective, achieving an AUC score of 0.99 and an F1 score of 0.95. Notably, the analysis identifies polyuria, polydipsia, age, and gender as significant predictors of early diabetes, highlighting the model's ability to discern critical risk factors.
The utilization of explainable AI techniques enhances the transparency and trustworthiness of machine learning models in healthcare. SHAP values offer both global and local explanations of model predictions, underscoring the importance of symptoms like polyuria and polydipsia in diagnosing diabetes. LIME provides localized explanations that further aid clinical decision-making by illustrating individual prediction factors. Despite the computational demands of these techniques, the study demonstrates their potential to support endocrinologists in identifying at-risk individuals and implementing preventive measures, ultimately contributing to more effective diabetes management.
Keywords: Explainable AI in diabetes, Early Prediction in diabetes, AI in healthcare



---

## Introduction 
Modern society faces a significant burden from life-threatening diseases that can often be effectively controlled if diagnosed early. According to IDF[1], by 2045, approximately 783 million adults globally, will be living with diabetes. 

Building upon predictive ML models, we employed post hoc explainability techniques to enhance interpretability and decision-making in the early detection of diabetes. This approach not only facilitates early detection of diabetes but also sheds light on the key variables driving these conditions, offering a deeper understanding of its causative factors and potentially guiding more effective intervention strategies.

In this study, we have constructed three predictive ML techniques, including RF, LR, and SVM to detect diabetes associated with sixteen predictors. Furthermore, we aim to answer the following three RQs:
	
1.	What are the practical features of a task that predict diabetes?
2.	Which ML model – has the highest ROC-AUC scores for predicting diabetes?
3.	Can XAI be utilized to assist endocrinologists in predicting diabetes outcomes more effectively?
---
## Conclusion
Polyuria and Polydipsia are well-known early signs of diabetes. In this study, we applied the SMOTE technique to three different machine-learning algorithms to identify these early indicators. Unlike previous research, we also integrated interpretable machine learning methods, specifically SHAP and LIME, to explain how our models arrived at predictions and to determine the relationships between target variables and predictors. Our findings revealed that the Random Forest (RF) model performed best, achieving an impressive F1 score of 95% and an AUC score of 0.99. These insights can aid endocrinologists in identifying individuals at risk of developing diabetes based on early symptoms, offering a preventive approach in diabetes management.
 
Throughout the project, an extensive analysis of explainable AI (XAI) techniques was conducted, though several limitations emerged. One significant challenge was scalability, particularly with SHAP, which becomes computationally intensive when applied to large datasets or complex models. Additionally, LIME, while popular for its simplicity, exhibited variability in its explanations, raising concerns about consistency and reliability. Addressing these limitations could be a focus of future work, such as exploring hybrid explainability approaches that combine different XAI methods to leverage their strengths while mitigating weaknesses. Furthermore, applying these techniques to alternative domains, like finance or criminal justice, could provide valuable insights into their generalizability and effectiveness in various fields.
 
Overall, this project highlights the crucial role of explainable AI in making machine learning models transparent, trustworthy, and actionable. By systematically evaluating LIME and SHAP, we have created a clear framework for selecting appropriate XAI techniques based on the needs of specific applications. As machine learning continues to influence high-stakes decision-making environments, explainability becomes increasingly vital. This project emphasizes the need for ongoing research and development to ensure that AI systems not only deliver powerful performance but also offer understandable and justifiable outcomes.
 




