"""
generate_sample.py — Generates data/sample.xlsx for training exercises.

Creates a realistic sales workbook with 3 sheets:
    - Sales    : 200 transaction rows
    - Clients  : 40 client records
    - Products : 20 product records

Run:
    python generate_sample.py
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_PATH = Path("data/sample.xlsx")

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------
REGIONS = ["North", "South", "East", "West", "Central"]
CATEGORIES = ["Electronics", "Office Supplies", "Furniture", "Software", "Services"]
PRODUCTS = [
    ("P001", "Laptop Pro 15", "Electronics", 1299.99),
    ("P002", "Wireless Mouse", "Electronics", 29.99),
    ("P003", "USB-C Hub", "Electronics", 49.99),
    ("P004", "Standing Desk", "Furniture", 499.99),
    ("P005", "Ergonomic Chair", "Furniture", 349.99),
    ("P006", "Monitor 27\"", "Electronics", 399.99),
    ("P007", "Stapler Set", "Office Supplies", 14.99),
    ("P008", "Notebook Pack", "Office Supplies", 9.99),
    ("P009", "Printer Ink", "Office Supplies", 24.99),
    ("P010", "CRM License", "Software", 89.99),
    ("P011", "Security Suite", "Software", 149.99),
    ("P012", "Cloud Storage", "Software", 19.99),
    ("P013", "Projector", "Electronics", 699.99),
    ("P014", "Webcam HD", "Electronics", 79.99),
    ("P015", "Keyboard Mech", "Electronics", 129.99),
    ("P016", "Filing Cabinet", "Furniture", 189.99),
    ("P017", "Desk Lamp", "Furniture", 59.99),
    ("P018", "Training Basic", "Services", 299.99),
    ("P019", "Training Advanced", "Services", 599.99),
    ("P020", "Consulting/hr", "Services", 149.99),
]

FIRST_NAMES = ["Alice", "Bob", "Carlos", "Diana", "Elena", "Frank", "Grace",
               "Hugo", "Irina", "James", "Karen", "Luis", "María", "Nora",
               "Oscar", "Paula", "Quinn", "Rafael", "Sara", "Tom"]
LAST_NAMES = ["Smith", "García", "Chen", "Johnson", "Williams", "Martínez",
              "Brown", "Davis", "Wilson", "López", "Anderson", "Taylor",
              "Thomas", "Hernández", "Moore", "Jackson", "Martín", "Lee",
              "Pérez", "Thompson"]
COMPANIES = [
    "TechCorp SA", "GlobalTrade SL", "InnovateTech", "DataSystems Inc",
    "CloudFirst", "DigitalEdge", "SmartSolutions", "FutureBiz Corp",
    "NexGen Ltd", "AlphaBeta SA", "Quadrant Group", "Pinnacle SL",
    "Horizon Tech", "Vertex Solutions", "Apex Digital", "Streamline Co",
    "CoreBusiness", "ProActive Ltd", "SynergyTech", "Catalyst Group",
    "Momentum SA", "Blueprint Corp", "Catalyst SL", "Keystone Inc",
    "Bridgepoint", "Crossroads Tech", "Lighthouse SA", "Meridian Corp",
    "Paragon Tech", "Summit Solutions", "Triton Group", "Vanguard SL",
    "Windmill Corp", "Xenith Inc", "Yellowstone SA", "Zenith Digital",
    "Acme Corp", "Bellwether Ltd", "Cascade SL", "Deltaforce Inc",
]

# ---------------------------------------------------------------------------
# Generate Products sheet
# ---------------------------------------------------------------------------
def make_products_df() -> pd.DataFrame:
    rows = []
    for pid, name, cat, price in PRODUCTS:
        rows.append({
            "product_id": pid,
            "product_name": name,
            "category": cat,
            "unit_price": price,
            "stock_units": random.randint(0, 500),
            "supplier": random.choice(["Supplier A", "Supplier B", "Supplier C"]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Generate Clients sheet
# ---------------------------------------------------------------------------
def make_clients_df() -> pd.DataFrame:
    rows = []
    for i, company in enumerate(COMPANIES):
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        rows.append({
            "client_id": f"C{i + 1:03d}",
            "company": company,
            "contact_name": f"{first} {last}",
            "email": f"{first.lower()}.{last.lower()}@{company.replace(' ', '').lower()[:10]}.com",
            "region": random.choice(REGIONS),
            "segment": random.choice(["SMB", "Enterprise", "Public Sector"]),
            "since_year": random.randint(2018, 2024),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Generate Sales sheet
# ---------------------------------------------------------------------------
def make_sales_df(clients: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")

    for i in range(200):
        client = clients.sample(1).iloc[0]
        product = products.sample(1).iloc[0]
        quantity = random.randint(1, 20)
        unit_price = product["unit_price"]
        discount = random.choice([0.0, 0.0, 0.0, 0.05, 0.10, 0.15])
        total = round(quantity * unit_price * (1 - discount), 2)
        sale_date = random.choice(dates)

        rows.append({
            "sale_id": f"S{i + 1:04d}",
            "date": sale_date,
            "quarter": f"Q{(sale_date.month - 1) // 3 + 1}",
            "client_id": client["client_id"],
            "company": client["company"],
            "region": client["region"],
            "segment": client["segment"],
            "product_id": product["product_id"],
            "product_name": product["product_name"],
            "category": product["category"],
            "quantity": quantity,
            "unit_price": unit_price,
            "discount": discount,
            "total_revenue": total,
            "salesperson": random.choice(FIRST_NAMES),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Generating sample data...")
    products = make_products_df()
    clients = make_clients_df()
    sales = make_sales_df(clients, products)

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        sales.to_excel(writer, sheet_name="Sales", index=False)
        clients.to_excel(writer, sheet_name="Clients", index=False)
        products.to_excel(writer, sheet_name="Products", index=False)

    print(f"Created: {OUTPUT_PATH.resolve()}")
    print(f"  Sales:    {len(sales)} rows")
    print(f"  Clients:  {len(clients)} rows")
    print(f"  Products: {len(products)} rows")


if __name__ == "__main__":
    main()
