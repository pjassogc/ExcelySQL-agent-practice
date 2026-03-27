"""
generate_db.py — Genera data/sample.db (SQLite) para los ejercicios de formación.

Reutiliza las funciones generadoras de generate_sample.py y escribe
los mismos datos en tres tablas SQLite: sales, clients, products.

Uso:
    python generate_db.py
"""

import sqlite3
from pathlib import Path

from generate_sample import make_clients_df, make_products_df, make_sales_df

DB_PATH = Path("data/sample.db")


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Generando datos de ejemplo...")
    products = make_products_df()
    clients = make_clients_df()
    sales = make_sales_df(clients, products)

    # SQLite no tiene tipo DATE nativo; guardamos como string ISO 8601
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
