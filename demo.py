import os
import argparse
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, default="images/dog.png")
    p.add_argument("--prompt", type=str, default="Describe the image in detail.")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--pretrained", type=str, default="lavida-ckpts/lavida-llada-hd")
    p.add_argument("--merge-layer", type=int, default=3)
    p.add_argument("--merge-ratio", type=float, default=0.8)
    p.add_argument("--merge-adaptive", type=int, default=1)
    p.add_argument("--merge-window", type=float, default=0.2)
    a = p.parse_args()

    os.environ["LLADA_MERGE_LAYER"] = str(a.merge_layer)
    os.environ["LLADA_MERGE_RATIO"] = str(a.merge_ratio)
    os.environ["LLADA_MERGE_RATIO_ADAPTIVE"] = str(int(a.merge_adaptive))

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))

    from src import lavida

    ans, elapsed = lavida.run(
        image_path=a.image,
        prompt=a.prompt,
        temperature=a.temperature,
        pretrained=a.pretrained,
    )

    print(ans)
    print(f"Time: {elapsed:.4f}s")


if __name__ == "__main__":
    main()