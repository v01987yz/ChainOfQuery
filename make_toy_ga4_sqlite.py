import argparse
import sqlite3
from pathlib import Path
import random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tmp/ga4_toy.db")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        out.unlink()

    con = sqlite3.connect(str(out))
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE events (
        event_date TEXT,         -- YYYYMMDD
        user_pseudo_id TEXT,
        event_name TEXT
    );
    """)

    # Create fake users
    users = [f"u{i:03d}" for i in range(1, 101)]

    # Insert some events for 20210101..20210107
    dates = ["20210101","20210102","20210103","20210104","20210105","20210106","20210107"]
    for d in dates:
        active_count = random.randint(20, 60)
        active_users = random.sample(users, active_count)
        for u in active_users:
            cur.execute(
                "INSERT INTO events(event_date, user_pseudo_id, event_name) VALUES (?,?,?)",
                (d, u, "user_engagement")
            )

    con.commit()
    con.close()
    print(f"[OK] wrote toy db: {out}")

if __name__ == "__main__":
    main()
