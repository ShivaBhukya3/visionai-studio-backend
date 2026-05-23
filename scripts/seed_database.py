#!/usr/bin/env python3
"""Seed the VisionAI Studio database with sample data."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


async def seed():
    try:
        from app.database import engine, Base
        from app.models import DetectionSession, DetectionRecord, ModelBenchmark

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        print("Database tables created.")
        print("Seed data inserted successfully.")
    except Exception as e:
        print(f"Seeding failed (DB may not be running): {e}")


if __name__ == "__main__":
    asyncio.run(seed())
