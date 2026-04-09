"""
ComfyUI API Client
Talks to ComfyUI's websocket API to queue prompts and retrieve images.
"""

import json
import uuid
import urllib.request
import urllib.parse
from pathlib import Path
import websocket
import base64

class ComfyClient:
    def __init__(self, host="127.0.0.1", port=8188):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws"
        self.client_id = str(uuid.uuid4())

    def is_running(self) -> bool:
        """Check if ComfyUI is accessible."""
        try:
            urllib.request.urlopen(f"{self.base_url}/system_stats", timeout=2)
            return True
        except:
            return False

    def queue_prompt(self, workflow: dict) -> str:
        """Queue a workflow and return the prompt_id."""
        payload = {
            "prompt": workflow,
            "client_id": self.client_id
        }
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            f"{self.base_url}/prompt",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        response = urllib.request.urlopen(req)
        result = json.loads(response.read())
        return result.get("prompt_id")

    def get_history(self, prompt_id: str) -> dict:
        """Get the execution history for a prompt."""
        url = f"{self.base_url}/history/{prompt_id}"
        response = urllib.request.urlopen(url)
        return json.loads(response.read())

    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """Download an image from ComfyUI's output."""
        params = urllib.parse.urlencode({
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type
        })
        url = f"{self.base_url}/view?{params}"
        response = urllib.request.urlopen(url)
        return response.read()

    def generate_image(self, prompt: str, negative: str = "", width: int = 1024, height: int = 1024,
                       steps: int = 20, cfg: float = 7.0, seed: int = -1,
                       checkpoint: str = None, sampler: str = "dpmpp_2m_sde",
                       scheduler: str = "karras", clip_skip: int = 2,
                       loras: list = None) -> list[bytes]:
        """
        Generate an image using a basic txt2img workflow.
        Returns list of image bytes.

        Args:
            loras: List of dicts with 'name', 'weight', 'enabled' keys
        """
        import random
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        # Filter to only enabled LoRAs
        active_loras = [l for l in (loras or []) if l.get('enabled', True)]

        # Basic SDXL workflow
        workflow = self._build_txt2img_workflow(
            prompt=prompt,
            negative=negative,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            seed=seed,
            checkpoint=checkpoint,
            sampler=sampler,
            scheduler=scheduler,
            clip_skip=clip_skip,
            loras=active_loras
        )

        # Queue and wait for result
        prompt_id = self.queue_prompt(workflow)
        return self._wait_for_images(prompt_id)

    def _build_txt2img_workflow(self, prompt: str, negative: str, width: int, height: int,
                                 steps: int, cfg: float, seed: int, checkpoint: str = None,
                                 sampler: str = "dpmpp_2m_sde", scheduler: str = "karras",
                                 clip_skip: int = 2, loras: list = None) -> dict:
        """Build a basic txt2img workflow JSON with optional LoRA support."""

        # Default checkpoint - user should set this
        ckpt = checkpoint or "sd_xl_base_1.0.safetensors"

        # Start building the workflow
        workflow = {
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": ckpt
                }
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "batch_size": 1,
                    "height": height,
                    "width": width
                }
            },
        }

        # Track where model and clip are coming from (for chaining LoRAs)
        model_source = ["4", 0]  # CheckpointLoader model output
        clip_source = ["4", 1]   # CheckpointLoader clip output

        # Add LoRA nodes if any are specified
        loras = loras or []
        for i, lora in enumerate(loras):
            lora_name = lora.get('name', '')
            lora_weight = float(lora.get('weight', 1.0))

            if not lora_name:
                continue

            # Add .safetensors extension if not present
            if not lora_name.endswith('.safetensors') and not lora_name.endswith('.pt'):
                lora_name += '.safetensors'

            node_id = f"lora_{i}"
            workflow[node_id] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": lora_name,
                    "strength_model": lora_weight,
                    "strength_clip": lora_weight,
                    "model": model_source,
                    "clip": clip_source
                }
            }

            # Update sources for next LoRA or final nodes
            model_source = [node_id, 0]
            clip_source = [node_id, 1]

            print(f"[comfy] Added LoRA: {lora_name} @ {lora_weight}")

        # Now add the rest of the workflow, using the final model/clip sources
        workflow["10"] = {
            "class_type": "CLIPSetLastLayer",
            "inputs": {
                "clip": clip_source,
                "stop_at_clip_layer": -clip_skip
            }
        }

        workflow["6"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["10", 0],
                "text": prompt
            }
        }

        workflow["7"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["10", 0],
                "text": negative
            }
        }

        workflow["3"] = {
            "class_type": "KSampler",
            "inputs": {
                "cfg": cfg,
                "denoise": 1.0,
                "latent_image": ["5", 0],
                "model": model_source,  # Use final model (after LoRAs)
                "negative": ["7", 0],
                "positive": ["6", 0],
                "sampler_name": sampler,
                "scheduler": scheduler,
                "seed": seed,
                "steps": steps
            }
        }

        workflow["8"] = {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2]
            }
        }

        workflow["9"] = {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "sd-rpg",
                "images": ["8", 0]
            }
        }

        return workflow

    def _wait_for_images(self, prompt_id: str, timeout: int = 300) -> list[bytes]:
        """Wait for generation to complete and return image bytes."""
        import time

        print(f"[comfy] Waiting for prompt_id: {prompt_id}")
        ws = websocket.create_connection(f"{self.ws_url}?clientId={self.client_id}")

        try:
            start = time.time()
            completed = False
            while time.time() - start < timeout:
                msg = ws.recv()
                if isinstance(msg, str):
                    data = json.loads(msg)
                    msg_type = data.get("type")
                    if msg_type == "executing":
                        exec_data = data.get("data", {})
                        node = exec_data.get("node")
                        msg_prompt_id = exec_data.get("prompt_id")
                        if node is None:
                            print(f"[comfy] Execution done signal for prompt: {msg_prompt_id}")
                            if msg_prompt_id == prompt_id:
                                completed = True
                                break
                    elif msg_type == "executed":
                        # Alternative completion signal
                        exec_data = data.get("data", {})
                        msg_prompt_id = exec_data.get("prompt_id")
                        print(f"[comfy] 'executed' signal for prompt: {msg_prompt_id}")
                        if msg_prompt_id == prompt_id:
                            completed = True
                            break

            if not completed:
                print(f"[comfy] WARNING: Timed out waiting for completion signal")

            # Retry fetching history until it appears (ComfyUI can be slow to update)
            images = []
            for attempt in range(30):  # Try up to 30 times (30 seconds for slow models like Flux)
                time.sleep(1)  # Wait 1 second between attempts
                history = self.get_history(prompt_id)

                if prompt_id in history:
                    outputs = history[prompt_id].get("outputs", {})
                    print(f"[comfy] Found history on attempt {attempt + 1}, output nodes: {list(outputs.keys())}")
                    for node_id, node_output in outputs.items():
                        if "images" in node_output:
                            for img_info in node_output["images"]:
                                print(f"[comfy] Found image: {img_info['filename']}")
                                img_data = self.get_image(
                                    img_info["filename"],
                                    img_info.get("subfolder", ""),
                                    img_info.get("type", "output")
                                )
                                images.append(img_data)
                    break
                else:
                    print(f"[comfy] Attempt {attempt + 1}: history not ready yet...")

            if not images:
                print(f"[comfy] WARNING: No images found after 30 attempts!")

            print(f"[comfy] Returning {len(images)} images")
            return images
        finally:
            ws.close()


# Quick test
if __name__ == "__main__":
    client = ComfyClient()
    if client.is_running():
        print("ComfyUI is running!")
    else:
        print("ComfyUI not detected at localhost:8188")
