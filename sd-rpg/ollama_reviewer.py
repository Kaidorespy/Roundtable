"""
Ollama Image Reviewer
Uses LLaVA or other vision models to review generated images.
"""

import json
import base64
import urllib.request
from pathlib import Path
from typing import Optional


class OllamaReviewer:
    def __init__(self, host="127.0.0.1", port=11434, model="llava"):
        self.base_url = f"http://{host}:{port}"
        self.model = model

    def is_running(self) -> bool:
        """Check if Ollama is accessible."""
        try:
            urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=2)
            return True
        except:
            return False

    def list_models(self) -> list[str]:
        """List available models."""
        try:
            response = urllib.request.urlopen(f"{self.base_url}/api/tags")
            data = json.loads(response.read())
            return [m["name"] for m in data.get("models", [])]
        except:
            return []

    def review_image(self, image_path: str, prompt: str,
                     question: str = None) -> str:
        """
        Review an image against its prompt.

        Args:
            image_path: Path to the image file
            prompt: The prompt that was used to generate it
            question: Custom question (default asks about accuracy)

        Returns:
            The model's assessment
        """
        if not question:
            question = f"""Look at this AI-generated image. It was generated with the prompt: "{prompt}"

Rate how well the image matches the prompt on a scale of 1-10, and briefly explain why.
Also note any obvious problems (weird anatomy, text artifacts, etc).

Format:
Score: X/10
Match: [brief explanation]
Issues: [any problems, or "none"]"""

        # Read and encode image
        img_data = Path(image_path).read_bytes()
        img_b64 = base64.b64encode(img_data).decode('utf-8')

        payload = {
            "model": self.model,
            "prompt": question,
            "images": [img_b64],
            "stream": False
        }

        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"}
        )

        response = urllib.request.urlopen(req, timeout=120)
        result = json.loads(response.read())
        return result.get("response", "No response")

    def describe_image(self, image_path: str) -> str:
        """Just describe what's in an image."""
        img_data = Path(image_path).read_bytes()
        img_b64 = base64.b64encode(img_data).decode('utf-8')

        payload = {
            "model": self.model,
            "prompt": "Describe this image in detail. What do you see?",
            "images": [img_b64],
            "stream": False
        }

        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"}
        )

        response = urllib.request.urlopen(req, timeout=120)
        result = json.loads(response.read())
        return result.get("response", "No response")

    def batch_review(self, generations: list[tuple[str, str]],
                     threshold: float = 7.0) -> list[dict]:
        """
        Review multiple images, flag ones below threshold.

        Args:
            generations: List of (image_path, prompt) tuples
            threshold: Minimum acceptable score

        Returns:
            List of review results
        """
        results = []
        for img_path, prompt in generations:
            try:
                review = self.review_image(img_path, prompt)
                # Try to parse score
                score = None
                for line in review.split('\n'):
                    if 'score:' in line.lower():
                        try:
                            score = float(line.split('/')[0].split(':')[-1].strip())
                        except:
                            pass

                results.append({
                    "image": img_path,
                    "prompt": prompt,
                    "review": review,
                    "score": score,
                    "passed": score is None or score >= threshold
                })
            except Exception as e:
                results.append({
                    "image": img_path,
                    "prompt": prompt,
                    "review": f"Error: {e}",
                    "score": None,
                    "passed": False
                })

        return results


if __name__ == "__main__":
    reviewer = OllamaReviewer()

    if reviewer.is_running():
        print("Ollama is running!")
        models = reviewer.list_models()
        print(f"Available models: {models}")
    else:
        print("Ollama not running at localhost:11434")
