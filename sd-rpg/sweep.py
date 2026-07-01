"""
Parameter Sweep Tool
Generate a grid of images varying steps, CFG, sampler to find optimal settings.
"""

import os
import random
import itertools
from datetime import datetime
from pathlib import Path

from comfy_client import ComfyClient
from settings import Settings

# Output folder
SWEEP_DIR = Path(__file__).parent / "sweeps"
SWEEP_DIR.mkdir(exist_ok=True)


class ParameterSweep:
    def __init__(self):
        self.settings = Settings.load()
        self.client = ComfyClient(
            host=self.settings.comfy_host,
            port=self.settings.comfy_port
        )

    def sweep(self, prompt: str,
              steps_list: list[int] = None,
              cfg_list: list[float] = None,
              sampler_list: list[str] = None,
              seed: int = None,
              width: int = 1024,
              height: int = 1024):
        """
        Run a parameter sweep and generate comparison grid.
        """
        if not self.client.is_running():
            print("ComfyUI not running!")
            return

        # Defaults
        if steps_list is None:
            steps_list = [20, 25, 28, 32]
        if cfg_list is None:
            cfg_list = [4.0, 5.0, 6.0, 7.0]
        if sampler_list is None:
            sampler_list = ["dpmpp_2m_sde"]  # Optimal for Illustrious

        # Use fixed seed for fair comparison
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        print(f"Parameter Sweep")
        print(f"Prompt: {prompt}")
        print(f"Seed: {seed} (fixed for comparison)")
        print(f"Steps: {steps_list}")
        print(f"CFG: {cfg_list}")
        print(f"Samplers: {sampler_list}")
        print()

        # Create sweep folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_folder = SWEEP_DIR / f"sweep_{timestamp}"
        sweep_folder.mkdir(exist_ok=True)

        results = []
        total = len(steps_list) * len(cfg_list) * len(sampler_list)
        current = 0

        for sampler in sampler_list:
            for steps in steps_list:
                for cfg in cfg_list:
                    current += 1
                    print(f"[{current}/{total}] steps={steps}, cfg={cfg}, sampler={sampler}...", end="", flush=True)

                    try:
                        # Build workflow with specific sampler
                        workflow = self._build_workflow(
                            prompt=prompt,
                            negative=self.settings.default_negative,
                            steps=steps,
                            cfg=cfg,
                            seed=seed,
                            sampler=sampler,
                            width=width,
                            height=height
                        )

                        prompt_id = self.client.queue_prompt(workflow)
                        images = self.client._wait_for_images(prompt_id)

                        if images:
                            filename = f"s{steps}_cfg{cfg}_{sampler}.png"
                            filepath = sweep_folder / filename
                            filepath.write_bytes(images[0])
                            print(f" saved")

                            results.append({
                                "steps": steps,
                                "cfg": cfg,
                                "sampler": sampler,
                                "filename": filename
                            })
                        else:
                            print(" no image")

                    except Exception as e:
                        print(f" error: {e}")

        # Generate comparison HTML
        self._generate_grid_html(sweep_folder, results, prompt, seed, steps_list, cfg_list, sampler_list)

        print(f"\nSweep complete! {len(results)} images generated.")
        print(f"Open: {sweep_folder / 'grid.html'}")

        # Auto-open
        import webbrowser
        webbrowser.open(f"file://{sweep_folder / 'grid.html'}")

        return sweep_folder

    def _build_workflow(self, prompt, negative, steps, cfg, seed, sampler, width, height):
        """Build workflow with specific sampler."""
        ckpt = self.settings.default_checkpoint or "realismIllustriousBy_v50FP16.safetensors"

        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": cfg,
                    "denoise": 1.0,
                    "latent_image": ["5", 0],
                    "model": ["4", 0],
                    "negative": ["7", 0],
                    "positive": ["6", 0],
                    "sampler_name": sampler,
                    "scheduler": "karras",
                    "seed": seed,
                    "steps": steps
                }
            },
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
            "10": {
                "class_type": "CLIPSetLastLayer",
                "inputs": {
                    "clip": ["4", 1],
                    "stop_at_clip_layer": -2
                }
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["10", 0],
                    "text": prompt
                }
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["10", 0],
                    "text": negative
                }
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                }
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "sweep",
                    "images": ["8", 0]
                }
            }
        }

    def _generate_grid_html(self, folder, results, prompt, seed, steps_list, cfg_list, sampler_list):
        """Generate an HTML grid for comparison with split-view preview."""

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Parameter Sweep</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            background: #1a1a1a;
            color: #eee;
            font-family: monospace;
            margin: 0;
            padding: 0;
            height: 100vh;
            overflow: hidden;
        }}
        .container {{
            display: flex;
            height: 100vh;
        }}
        .left-panel {{
            width: 50%;
            overflow-y: auto;
            padding: 20px;
            border-right: 1px solid #333;
        }}
        .right-panel {{
            width: 50%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background: #111;
        }}
        h1 {{ color: #fff; margin-top: 0; font-size: 1.5em; }}
        .info {{
            background: #2a2a2a;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 15px;
            font-size: 12px;
        }}
        .sampler-section {{
            margin-bottom: 30px;
        }}
        .sampler-section h2 {{
            font-size: 1em;
            margin-bottom: 10px;
        }}
        table {{
            border-collapse: collapse;
        }}
        th {{
            background: #333;
            padding: 8px;
            text-align: center;
            font-size: 11px;
        }}
        td {{
            padding: 4px;
            text-align: center;
            vertical-align: top;
        }}
        td img {{
            width: 120px;
            height: 120px;
            object-fit: cover;
            border-radius: 4px;
            cursor: pointer;
            transition: outline 0.1s;
        }}
        td img:hover {{
            outline: 2px solid #4a9eff;
        }}
        td img.selected {{
            outline: 3px solid #00ff88;
        }}
        .label {{
            font-size: 10px;
            color: #666;
            margin-top: 3px;
        }}
        #preview {{
            max-width: 90%;
            max-height: 70vh;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }}
        #preview-info {{
            margin-top: 15px;
            text-align: center;
            font-size: 14px;
        }}
        #preview-info .params {{
            color: #4a9eff;
            font-size: 18px;
            margin-bottom: 5px;
        }}
        .placeholder {{
            color: #555;
            font-size: 16px;
        }}
        #use-btn {{
            margin-top: 15px;
            padding: 10px 25px;
            background: #4a9eff;
            border: none;
            border-radius: 6px;
            color: #fff;
            font-size: 14px;
            cursor: pointer;
            font-family: monospace;
        }}
        #use-btn:hover {{
            background: #3a8eef;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="left-panel">
            <h1>Parameter Sweep</h1>
            <div class="info">
                <div><strong>Prompt:</strong> {prompt[:80]}{'...' if len(prompt) > 80 else ''}</div>
                <div><strong>Seed:</strong> {seed}</div>
            </div>
"""

        for sampler in sampler_list:
            html += f"""
            <div class="sampler-section">
                <h2>Sampler: {sampler}</h2>
                <table>
                    <tr>
                        <th></th>
"""
            for cfg in cfg_list:
                html += f"                        <th>CFG {cfg}</th>\n"
            html += "                    </tr>\n"

            for steps in steps_list:
                html += f"                    <tr>\n                        <th>{steps}s</th>\n"
                for cfg in cfg_list:
                    match = next((r for r in results if r["steps"] == steps and r["cfg"] == cfg and r["sampler"] == sampler), None)
                    if match:
                        html += f"""                        <td>
                            <img src="{match['filename']}"
                                 onclick="selectImage(this, '{match['filename']}', {steps}, {cfg}, '{sampler}')"
                                 alt="s{steps} cfg{cfg}">
                        </td>
"""
                    else:
                        html += "                        <td>-</td>\n"
                html += "                    </tr>\n"

            html += """                </table>
            </div>
"""

        html += """
        </div>
        <div class="right-panel">
            <div id="preview-container">
                <p class="placeholder">Click an image to preview</p>
            </div>
            <div id="preview-info" style="display:none;">
                <div class="params" id="params-display"></div>
                <div id="filename-display" style="color:#888;"></div>
                <button id="use-btn" onclick="copySettings()">Copy settings command</button>
            </div>
        </div>
    </div>

    <script>
        let currentParams = {};

        function selectImage(el, filename, steps, cfg, sampler) {
            // Remove previous selection
            document.querySelectorAll('td img.selected').forEach(img => img.classList.remove('selected'));
            // Select this one
            el.classList.add('selected');

            // Update preview
            const container = document.getElementById('preview-container');
            container.innerHTML = '<img id="preview" src="' + filename + '" onclick="window.open(\\'' + filename + '\\')">';

            // Update info
            document.getElementById('preview-info').style.display = 'block';
            document.getElementById('params-display').textContent = steps + ' steps, CFG ' + cfg + ', ' + sampler;
            document.getElementById('filename-display').textContent = filename;

            currentParams = {steps, cfg, sampler};
        }

        function copySettings() {
            const cmd = '/set steps ' + currentParams.steps + ' && /set cfg ' + currentParams.cfg;
            navigator.clipboard.writeText(cmd).then(() => {
                document.getElementById('use-btn').textContent = 'Copied!';
                setTimeout(() => {
                    document.getElementById('use-btn').textContent = 'Copy settings command';
                }, 1500);
            });
        }
    </script>
</body>
</html>
"""
        (folder / "grid.html").write_text(html)


def quick_sweep():
    """Run a quick default sweep."""
    sweep = ParameterSweep()

    print("Quick Parameter Sweep")
    print("=" * 40)
    prompt = input("Prompt: ").strip()
    if not prompt:
        prompt = "a dragon in a cave, fantasy art"

    # Quick defaults: 4x4 grid
    sweep.sweep(
        prompt=prompt,
        steps_list=[15, 20, 25, 30],
        cfg_list=[5.0, 7.0, 7.5, 9.0]
    )


def custom_sweep():
    """Interactive custom sweep."""
    sweep = ParameterSweep()

    print("Custom Parameter Sweep")
    print("=" * 40)

    prompt = input("Prompt: ").strip()
    if not prompt:
        prompt = "a dragon in a cave, fantasy art"

    steps_input = input("Steps (comma-separated, e.g., 15,20,25,30): ").strip()
    steps_list = [int(x.strip()) for x in steps_input.split(",")] if steps_input else [15, 20, 25, 30]

    cfg_input = input("CFG (comma-separated, e.g., 5,7,7.5,9): ").strip()
    cfg_list = [float(x.strip()) for x in cfg_input.split(",")] if cfg_input else [5.0, 7.0, 7.5, 9.0]

    sampler_input = input("Samplers (comma-separated, e.g., euler,euler_ancestral,dpmpp_2m): ").strip()
    sampler_list = [x.strip() for x in sampler_input.split(",")] if sampler_input else ["euler"]

    seed_input = input("Seed (or Enter for random): ").strip()
    seed = int(seed_input) if seed_input else None

    sweep.sweep(
        prompt=prompt,
        steps_list=steps_list,
        cfg_list=cfg_list,
        sampler_list=sampler_list,
        seed=seed
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "custom":
        custom_sweep()
    else:
        quick_sweep()
