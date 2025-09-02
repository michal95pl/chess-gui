class Logger:

    file_logger = open("gui.log", "a")

    @staticmethod
    def log(message: str):
        Logger.file_logger.write(message + "\n")
        Logger.file_logger.flush()