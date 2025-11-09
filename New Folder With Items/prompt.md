You are working as a data analyst for a climate research institute. Your colleague prepared a notebook that analyzes average monthly temperatures for several U.S. cities (NYC, LA, Chicago, Miami, Dallas) from October through December. Your task is to extend the notebook with a temperature forecasting feature that predicts next month’s (January) average temperature trend.

You must:

1. Create a function called forecast_temp() that:

-Defines a new variable rolling_avg, which represents the 3-month rolling average of the mean temperatures across all cities for October–December.
(Hint: take the mean of the column averages for oct_temp, nov_temp, and dec_temp.)

-Defines another new variable predicted_temp, calculated by performing simple linear regression using numpy.polyfit on the October–December averages
and extrapolating to the next month (x = 4 → January).

-Returns a dictionary formatted as: {'rolling_avg': value, 'predicted_temp': value}

2. Ensure values are rounded to two decimal places.

3. Preserve all existing temperature calculations in the notebook.
