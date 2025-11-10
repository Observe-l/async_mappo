import datetime
import mysql.connector
from typing import List, Dict, Any, Optional

class MySQLRunLogger:
    """Manage per-run SUMO simulation logging into a fresh MySQL database.

    Creates a new database named sumo_YYYYMMDD_HHMMSS at init, then creates one table per truck.
    Table schema columns:
      sim_time VARCHAR(16)   -- HH:MM:SS
      rul FLOAT
      driving_distance_km FLOAT
      state VARCHAR(32)
      destination VARCHAR(32)
      loaded_goods VARCHAR(32)
      weight FLOAT
      total_transported FLOAT
      decision_flag TINYINT  -- 1 if this row inserted due to RL decision, else 0 (periodic)
    """
    def __init__(self, user: str, password: str, host: str = '127.0.0.1', port: int = 3306,
                 database: Optional[str] = None, prefix: str = 'sumo_'):
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.db_name = database if database else f'{prefix}{ts}'
        self._conn = mysql.connector.connect(user=user, password=password, host=host, port=port)
        self._conn.autocommit = True
        self._cursor = self._conn.cursor()
        # Quote database identifier with backticks to be safe
        self._cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{self.db_name}`")
        self._cursor.execute(f"USE `{self.db_name}`")
        self.tables_created: Dict[str, bool] = {}

    def ensure_tables(self, truck_ids: List[str]):
        for tid in truck_ids:
            if self.tables_created.get(tid):
                continue
            ddl = (
                f"CREATE TABLE IF NOT EXISTS `{tid}` ("
                "id INT AUTO_INCREMENT PRIMARY KEY,"
                "sim_time VARCHAR(16),"
                "rul FLOAT,"
                "driving_distance_km FLOAT,"
                "state VARCHAR(32),"
                "destination VARCHAR(32),"
                "loaded_goods VARCHAR(32),"
                "weight FLOAT,"
                "total_transported FLOAT,"
                "decision_flag TINYINT"
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
            )
            self._cursor.execute(ddl)
            self.tables_created[tid] = True

    def insert(self, truck_id: str, row: Dict[str, Any]):
        self.ensure_tables([truck_id])
        cols = ["sim_time","rul","driving_distance_km","state","destination","loaded_goods","weight","total_transported","decision_flag"]
        values = [row.get(c) for c in cols]
        placeholders = ','.join(['%s'] * len(values))
        sql = f"INSERT INTO `{truck_id}` ({','.join(cols)}) VALUES ({placeholders})"
        self._cursor.execute(sql, values)

    def get_database_name(self) -> str:
        return self.db_name

    def close(self):
        try:
            self._cursor.close()
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass
