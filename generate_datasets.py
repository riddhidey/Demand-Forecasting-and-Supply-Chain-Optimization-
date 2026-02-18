import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_dataset(store_id, sku_id, start_date, num_weeks, base_demand, filename):
    dates = [start_date + timedelta(weeks=i) for i in range(num_weeks)]
    units_sold = np.random.poisson(lam=base_demand, size=num_weeks)
    df = pd.DataFrame({
        "week": [d.strftime("%y/%m/%d") for d in dates],
        "store_id": [store_id] * num_weeks,
        "sku_id": [sku_id] * num_weeks,
        "units_sold": units_sold,
        "is_featured_sku": np.random.choice([0, 1], size=num_weeks),
        "is_display_sku": np.random.choice([0, 1], size=num_weeks),
    })
    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")

# Generate 3 datasets
start = datetime(2022, 1, 1)
generate_dataset("ELEC001", "TV101", start, 52, 120, "electronics_store.csv")
generate_dataset("GROC002", "MILK202", start, 52, 180, "grocery_store.csv")
generate_dataset("FASH003", "JEANS303", start, 52, 95, "fashion_store.csv")
