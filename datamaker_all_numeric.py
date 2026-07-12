import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import uuid
from tqdm import tqdm # Install with: pip install tqdm

def create_credit_card_dataset(file_path: str, total_rows: int = 5_000_00, batch_size: int = 1_000_000):
    """
    Generates a 5M row credit card application dataset in chunks to prevent memory overflow.
    """
    
    # 1. Define the Schema
    schema = pa.schema([
        ('application_id', pa.string()),
        ('age', pa.int32()),
        ('annual_income', pa.float32()),
        ('credit_score', pa.int32()),
        ('existing_debt', pa.float32()),
        ('employment_type', pa.int8()), # 0: Salaried, 1: Self-Employed
        ('city_tier', pa.int8()),       # 1, 2, 3
        ('years_employed', pa.int8()),
        ('number_of_cards', pa.int8()),
        ('previous_defaults', pa.int8()),
        ('loan_to_income_ratio', pa.float32()),
        ('requested_limit', pa.float32()),
        ('home_ownership', pa.int8()),
        ('gender', pa.int8()),
        ('education_level', pa.int8()),
        ('dependents', pa.int8()),
        ('account_balance', pa.float32()),
        ('monthly_spend', pa.float32()),
        ('application_channel', pa.int8()),
        ('is_approved', pa.int8())      # Target variable (0 or 1)
    ])

    # 2. Setup Parquet Writer
    writer = pq.ParquetWriter(file_path, schema, compression='snappy')

    print(f"Starting generation of {total_rows:,} rows...")

    for i in tqdm(range(0, total_rows, batch_size)):
        current_batch = min(batch_size, total_rows - i)
        
        # 3. Generate Random Data for the batch
        data = {
            'application_id': [str(uuid.uuid4()) for _ in range(current_batch)],
            'age': np.random.randint(18, 75, current_batch),
            'annual_income': np.random.uniform(20000, 200000, current_batch).astype(np.float32),
            'credit_score': np.random.randint(300, 850, current_batch),
            'existing_debt': np.random.uniform(0, 50000, current_batch).astype(np.float32),
            'employment_type': np.random.randint(0, 2, current_batch),
            'city_tier': np.random.randint(1, 4, current_batch),
            'years_employed': np.random.randint(0, 40, current_batch),
            'number_of_cards': np.random.randint(0, 6, current_batch),
            'previous_defaults': np.random.choice([0, 1], p=[0.95, 0.05], size=current_batch),
            'loan_to_income_ratio': np.random.uniform(0, 0.8, current_batch).astype(np.float32),
            'requested_limit': np.random.uniform(1000, 50000, current_batch).astype(np.float32),
            'home_ownership': np.random.randint(0, 3, current_batch),
            'gender': np.random.randint(0, 2, current_batch),
            'education_level': np.random.randint(0, 4, current_batch),
            'dependents': np.random.randint(0, 5, current_batch),
            'account_balance': np.random.uniform(0, 100000, current_batch).astype(np.float32),
            'monthly_spend': np.random.uniform(0, 10000, current_batch).astype(np.float32),
            'application_channel': np.random.randint(0, 3, current_batch),
        }
        
        # 4. Generate Target Variable (Logic-based)
        # Higher credit score and lower LTI ratio increases approval probability
        score = (data['credit_score'] / 850) - (data['loan_to_income_ratio'])
        probs = 1 / (1 + np.exp(-(score * 10 - 5))) # Sigmoid-like logic
        data['is_approved'] = np.random.binomial(1, probs).astype(np.int8)

        # 5. Convert to Arrow Table and Write
        table = pa.Table.from_pydict(data, schema=schema)
        writer.write_table(table)

    writer.close()
    print("Dataset generation complete. Saved to:", file_path)

if __name__ == "__main__":
    create_credit_card_dataset("credit_app_data.parquet")