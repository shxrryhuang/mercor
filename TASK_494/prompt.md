You are a data analyst for a nationwide retail chain that tracks inventory turnover across stores. The dataset includes the number of units sold per store in Q2, Q3, and Q4. Your goal is to implement a forecasting function that estimates how many units each store is expected to sell in Q1 next year. Your colleague provided a notebook that calculates growth percentages between Q3 and Q4 but doesn’t forecast future trends.

Your task is to:

1. Implement a function called predict_turnover() that:

Introduces a new variable avg_turnover, representing the 3-quarter moving average of the company’s total units sold (Q2–Q4).
(sum all stores’ sales for Q2, Q3, and Q4, then divide by 3.)

-Introduces another new variable predicted_turnover, computed using linear regression (numpy.polyfit) based on the total units from Q2–Q4, and extrapolated to the next quarter (x = 4 → Q1 next year).

Returns a dictionary formatted as: {'avg_turnover': value, 'predicted_turnover': value}

2. Round all numerical outputs to two decimal places.

3. Retain the original growth_pct computation in the notebook.