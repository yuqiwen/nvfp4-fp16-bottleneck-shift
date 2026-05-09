import sqlite3
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python query_nsys_sqlite.py /path/to/report.sqlite")

    db = Path(sys.argv[1]).expanduser()
    con = sqlite3.connect(str(db))
    cur = con.cursor()

    print(f"===FILE===\n{db}")
    print("===STAGES===")
    for text, ms in cur.execute(
        "SELECT text, (end-start)/1000000.0 "
        "FROM NVTX_EVENTS "
        "WHERE text LIKE 'LLM_PREFILL%' OR text LIKE 'LLM_GENERATION%' "
        "ORDER BY start"
    ):
        print(f"{text}\t{ms:.3f} ms")

    print("===ITER_STATS===")
    n, avg_ms, min_ms, max_ms = cur.execute(
        "SELECT COUNT(*), AVG((end-start)/1000000.0), "
        "MIN((end-start)/1000000.0), MAX((end-start)/1000000.0) "
        "FROM NVTX_EVENTS WHERE text LIKE 'Decode_Iter%'"
    ).fetchone()
    print(f"n={n}, avg_ms={avg_ms:.3f}, min_ms={min_ms:.3f}, max_ms={max_ms:.3f}")

    print("===FIRST_ITERS===")
    for text, ms in cur.execute(
        "SELECT text, (end-start)/1000000.0 "
        "FROM NVTX_EVENTS WHERE text LIKE 'Decode_Iter%' "
        "ORDER BY start LIMIT 8"
    ):
        print(f"{text}\t{ms:.3f} ms")

    print("===LAST_ITERS===")
    for text, ms in cur.execute(
        "SELECT text, (end-start)/1000000.0 "
        "FROM NVTX_EVENTS WHERE text LIKE 'Decode_Iter%' "
        "ORDER BY start DESC LIMIT 8"
    ):
        print(f"{text}\t{ms:.3f} ms")

    con.close()


if __name__ == "__main__":
    main()
