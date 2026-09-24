"""
Tests for Vizpilot core package using standard library unittest
"""
import os
import shutil
import sqlite3
import tempfile
import unittest
import pandas as pd

from src.agent import build_graph, get_database_tables, _load_dataframe
from src.sandbox import static_safety_check, UnsafeCodeError


class TestVizpilot(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_static_safety_check(self):
        """Verify dangerous functions are caught by static denylist."""
        safe_code = "fig = px.bar(df, x='category', y='sales')"
        static_safety_check(safe_code)  # Should not raise

        unsafe_codes = [
            "import os; os.system('rm -rf /')",
            "import subprocess; subprocess.run(['ls'])",
            "open('some_file.txt', 'w')",
            "eval('1 + 1')",
        ]
        for code in unsafe_codes:
            with self.assertRaises(UnsafeCodeError):
                static_safety_check(code)

    def test_sqlite_database_tables_and_loading(self):
        """Verify SQLite database inspection and table loading."""
        db_file = os.path.join(self.test_dir, "test.db")
        conn = sqlite3.connect(db_file)
        pd.DataFrame({"id": [1, 2], "name": ["A", "B"]}).to_sql("users", conn, index=False)
        pd.DataFrame({"id": [10, 20], "total": [100.0, 200.0]}).to_sql("orders", conn, index=False)
        conn.close()

        tables = get_database_tables(db_file)
        self.assertIn("users", tables)
        self.assertIn("orders", tables)

        df_users = _load_dataframe(db_file, table_name="users")
        self.assertEqual(len(df_users), 2)
        self.assertIn("name", df_users.columns)

        df_orders = _load_dataframe(db_file, table_name="orders")
        self.assertEqual(len(df_orders), 2)
        self.assertIn("total", df_orders.columns)

    def test_build_graph(self):
        """Verify LangGraph state machine compiles successfully."""
        app = build_graph()
        self.assertIsNotNone(app)


if __name__ == "__main__":
    unittest.main()
