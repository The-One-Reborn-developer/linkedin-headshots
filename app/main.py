import torch
import logging

from diffusers import FluxPipeline


def load_pipe(attempts=3) -> FluxPipeline:
    for attempt in range(attempts):
        try:
            logging.info("Loading pipe...")
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-schnell",
                torch_dtype=torch.float16,
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

    enable_cpu_offload_result = enable_cpu_offload(pipe)
    if not enable_cpu_offload_result:
        logging.warning("Failed to enable CPU offload. Exiting...")
        return

    logging.info("Pipeline loaded successfully.")

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


if __name__ == "__main__":
    main()
