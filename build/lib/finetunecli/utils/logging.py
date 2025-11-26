import logging

def get_logger(name="finetune_cli"):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )
    return logging.getLogger(name)
