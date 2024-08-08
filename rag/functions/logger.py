import logging

class CustomLogger:
    RAG_LEVEL_NUM = 35  # Custom log level number

    def __init__(self, name: str, log_file: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Define a new log level called "RAG"
        logging.addLevelName(self.RAG_LEVEL_NUM, "RAG")

        # Create a custom handler for the RAG log level
        self.rag_handler = logging.FileHandler(log_file)
        self.rag_handler.setLevel(self.RAG_LEVEL_NUM)

        # Create a formatter and set it for the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.rag_handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(self.rag_handler)

    def log(self, message: str, *args, **kwargs):
        if self.logger.isEnabledFor(self.RAG_LEVEL_NUM):
            self.logger._log(self.RAG_LEVEL_NUM, message, args, **kwargs)


# Usage example:
if __name__ == "__main__":
    custom_logger = CustomLogger("my_custom_logger", "rag_log.log")
    custom_logger.log("This is a RAG level log message!")
