EconoCausal: Dynamic Pricing via Double Machine Learning

Project Overview
EconoCausal is a data science project that uses Causal AI and Double Machine Learning (DML) to estimate how different discount offers affect customer purchasing behavior.
Traditional machine learning predicts which customers are likely to purchase. EconoCausal goes further by estimating the potential effect of offering a higher discount to a customer.

The project compares two discount options:
$10 discount
$20 discount
It identifies customers who may respond positively to the higher discount and helps allocate a limited marketing budget to those customers.

Project Objectives
Analyze and preprocess retail customer data.
Build a causal Directed Acyclic Graph (DAG) using DoWhy.
Estimate customer-level treatment effects using EconML.
Identify customers who may respond positively to higher discounts.
Optimize discount allocation based on a specified budget.
Evaluate uplift and present results through a React dashboard.

Technologies Used
Python — Data processing and model development
Pandas & NumPy — Data manipulation
Scikit-learn — Machine learning models
DoWhy — Causal graph development and causal analysis
EconML — Double Machine Learning and treatment-effect estimation

Key Features
1. Causal Analysis
Uses causal reasoning and a DAG to represent relationships between customer characteristics, discount offers, and purchase outcomes.
2. Double Machine Learning
Uses EconML's LinearDML approach with machine learning models to estimate the treatment effect of a $20 discount compared with a $10 discount.
3. Customer Targeting
Ranks customers based on their estimated treatment effects to identify those who may benefit from the higher discount.
4. Budget Optimization
Selects eligible customers based on estimated treatment effects and a user-specified budget. The current implementation assumes an incremental cost of $10 per customer when moving from a $10 discount to a $20 discount.
5. Uplift Analysis
Evaluates how effectively the model ranks customers according to their estimated response to the discount intervention.
6. Interactive Dashboard
Provides dashboard sections for:
Dashboard overview
Campaign data upload
Budget optimization
Causal analysis
Uplift analysis

Project Workflow
Explore the available customer and retail data.
Clean and preprocess the data.
Prepare the dataset for causal analysis.
Define the causal relationships using a DAG.
Train a Double Machine Learning model.
Estimate individual customer treatment effects.
Rank customers based on estimated effects.
Optimize the discount budget.
Evaluate uplift performance.
Display results through the web dashboard.
Flask — Backend API
React & Vite — Interactive frontend dashboard
Matplotlib — Data visualization
