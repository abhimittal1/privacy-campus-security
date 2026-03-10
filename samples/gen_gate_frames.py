"""
gen_gate_frames.py – Generate synthetic placeholder frames for simulate-camera testing.

Usage:
    python samples/gen_gate_frames.py

Produces 5 numbered JPEG frames in samples/gate_frames/:
    frame_001.jpg … frame_005.jpg

Each frame is a solid colour with a white text overlay showing camera ID,
frame number, and a timestamp.  These are enough to verify that
``simulate-camera`` can read, re-encode, and publish frames via MQTT.
"""

from __future__ import annotations

from pathlib import Path


def generate(
    out_dir: str | Path = "samples/gate_frames",
    n_frames: int = 5,
    width: int = 640,
    height: int = 480,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise SystemExit(
            "Pillow is required.  Install with:  pip install Pillow"
        )

    import datetime

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Cycle through a handful of colours so frames look distinct
    colours = [
        (31, 60, 102),    # dark blue
        (20, 80, 50),     # dark green
        (102, 40, 40),    # dark red
        (60, 55, 100),    # purple
        (90, 70, 30),     # brown
    ]

    for i in range(n_frames):
        colour = colours[i % len(colours)]
        img = Image.new("RGB", (width, height), colour)
        draw = ImageDraw.Draw(img)

        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        lines = [
            "Privacy Campus Security",
            "Camera: gate-north",
            f"Frame : {i + 1:03d}",
            timestamp,
        ]

        y = height // 4
        for line in lines:
            # Use default bitmap font – no external font file required
            draw.text((width // 10, y), line, fill=(255, 255, 255))
            y += 30

        fname = out / f"frame_{i + 1:03d}.jpg"
        img.save(fname, format="JPEG", quality=85)
        print(f"  Wrote {fname}")

    print(f"Generated {n_frames} frames in {out.resolve()}")


if __name__ == "__main__":
    generate()
