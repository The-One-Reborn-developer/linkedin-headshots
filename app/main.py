import torch
import subprocess
import logging

from diffusers import FluxPipeline


# Set to True if you want to use the GPU
NVIDIA_CARD = False
TORCH_TYPE = torch.float32


def manage_swapfile(action: str, swapfile_path="/swapfile"):
    """
    Enable or disable the swapfile.

    :param action: 'on' to enable, 'off' to disable.
    :param swapfile_path: Path to the swapfile.
    :return: True if successful, False otherwise.
    """
    try:
        if action == "off":
            logging.info("Turning off the swapfile...")
            subprocess.run(["sudo", "swapoff", swapfile_path], check=True)
            logging.info("Swapfile turned off.")
        elif action == "on":
            logging.info("Turning on the swapfile...")
            subprocess.run(["sudo", "swapon", swapfile_path], check=True)
            logging.info("Swapfile turned on.")
        else:
            logging.error(
                "Invalid action for manage_swapfile: Use 'on' or 'off'.")
            return False
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to {action} swapfile: {e}")
        return False


def load_pipe(attempts=3) -> FluxPipeline:
    if NVIDIA_CARD:
        TORCH_TYPE = torch.float16

    for attempt in range(attempts):
        try:
            logging.info("Loading pipe...")
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-schnell",
                torch_dtype=TORCH_TYPE,
                low_cpu_mem_usage=True,
            )
            logging.info("Pipe created.")
            return pipe
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} to load pipe failed: {e}")
    logging.error(f"Failed to load pipe after {attempts} attempts.")
    return None


def enable_cpu_offload(pipe, attempts=3) -> True:
    for attempt in range(attempts):
        try:
            logging.info("Enabling CPU offload...")
            pipe.enable_model_cpu_offload()
            logging.info("CPU offload enabled.")
            return True
        except Exception as e:
            logging.warning(
                f"Attempt {attempt + 1} to enable CPU offload failed: {e}")
    logging.error(f"Failed to enable CPU offload after {attempts} attempts.")
    return False


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    pipe = load_pipe()
    if pipe is None:
        logging.warning("Failed to load pipe. Exiting...")
        return

    logging.info("Pipeline loaded successfully.")

    if NVIDIA_CARD:
        enable_cpu_offload_result = enable_cpu_offload(pipe)
        if not enable_cpu_offload_result:
            logging.warning("Failed to enable CPU offload. Exiting...")
            return

    logging.info("Generating image...")
    prompt = 'A wolf-man in a businessman suit in an office'
    image = pipe(
        prompt,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        num_images_per_prompt=1
    ).images[0]
    image.save("output.png")
    logging.info("Image saved successfully.")

    logging.info("Cleaning up the swapfile...")
    manage_swapfile("off")
    manage_swapfile("on")
    logging.info("Swapfile cleaned up.")


if __name__ == "__main__":
    main()
