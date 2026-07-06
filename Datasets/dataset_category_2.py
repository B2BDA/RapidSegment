import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

n_rows = 250000

# 1. Generate Customer ID
customer_ids = [f"C{i:07d}" for i in range(1, n_rows + 1)]

# Define categorical choices and initial baseline probabilities
# Employment Type (High IV)
emp_choices = ['Full-Time', 'Part-Time', 'Self-Employed', 'Unemployed']
# Housing Status (High IV)
house_choices = ['Own', 'Mortgaged', 'Rent', 'Other']
# Missed Payment Pattern
payment_choices = ['None', 'Occasional', 'Frequent']
# Prior Bankruptcy
bankruptcy_choices = ['No', 'Yes']

# Vectorized random choice generation
emp = np.random.choice(emp_choices, size=n_rows, p=[0.50, 0.20, 0.18, 0.12])
house = np.random.choice(house_choices, size=n_rows, p=[0.35, 0.35, 0.25, 0.05])
payment = np.random.choice(payment_choices, size=n_rows, p=[0.55, 0.30, 0.15])
bankruptcy = np.random.choice(bankruptcy_choices, size=n_rows, p=[0.88, 0.12])

# Assign risk probabilities based on combinations to guarantee High IV for Employment and Housing
# Let's create a logit score
# High risk for Unemployed (2.5), Self-Employed (1.2)
emp_risk = np.select(
    [emp == 'Unemployed', emp == 'Self-Employed', emp == 'Part-Time', emp == 'Full-Time'],
    [2.5, 1.2, 0.4, -1.5]
)

# High risk for Rent (1.8), Other (1.5)
house_risk = np.select(
    [house == 'Rent', house == 'Other', house == 'Mortgaged', house == 'Own'],
    [1.8, 1.5, -0.5, -1.8]
)

# Other controls
pay_risk = np.select(
    [payment == 'Frequent', payment == 'Occasional', payment == 'None'],
    [1.5, 0.2, -1.2]
)

bank_risk = np.select(
    [bankruptcy == 'Yes', bankruptcy == 'No'],
    [1.0, -0.2]
)

# Combine risk with a logistic distribution perturbation
logit = -1.0 + emp_risk + house_risk + pay_risk + bank_risk
prob = 1 / (1 + np.exp(-logit))

# Determine Default Status (Target)
default_status = np.random.binomial(1, prob)

# Construct DataFrame
df_large = pd.DataFrame({
    'Customer_ID': customer_ids,
    'Employment_Type': emp,
    'Housing_Status': house,
    'Missed_Payment_Pattern': payment,
    'Prior_Bankruptcy': bankruptcy,
    'Default_Status': default_status
})

# Verify distribution/default rates to ensure high IV
print("Emp Type Default Rates:\n", df_large.groupby('Employment_Type')['Default_Status'].mean())
print("\nHousing Status Default Rates:\n", df_large.groupby('Housing_Status')['Default_Status'].mean())

# Export to parquet (or compressed CSV) to fit into memory and file limits easily
parquet_filename = 'credit_card_default_2.5M.parquet'
df_large.to_parquet(parquet_filename, compression='snappy', index=False)
print(f"\nSuccessfully saved {n_rows} rows to {parquet_filename}")