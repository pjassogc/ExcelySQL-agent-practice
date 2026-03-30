# generate_db.py
from pathlib import Path
import sqlite3
import pandas as pd

from generate_sample import make_products_df, make_clients_df, make_sales_df

DB_PATH = Path("data/sample.db")


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Generando datos de ejemplo...")
    products = make_products_df()
    clients = make_clients_df()
    sales = make_sales_df(clients, products)

    # SQLite no tiene tipo DATE nativo; guardamos como string ISO
    sales["date"] = sales["date"].dt.strftime("%Y-%m-%d")

    conn = sqlite3.connect(DB_PATH)
    products.to_sql("products", conn, if_exists="replace", index=False)
    clients.to_sql("clients", conn, if_exists="replace", index=False)
    sales.to_sql("sales", conn, if_exists="replace", index=False)
    conn.close()

    print(f"Creada: {DB_PATH.resolve()}")
    print(f"  sales:    {len(sales)} filas")
    print(f"  clients:  {len(clients)} filas")
    print(f"  products: {len(products)} filas")


if __name__ == "__main__":
    main()