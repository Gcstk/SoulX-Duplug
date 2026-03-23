from __future__ import annotations

import os

import uvicorn
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    port = int(os.getenv("PORT", "3040"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info", reload=False)


if __name__ == "__main__":
    main()
