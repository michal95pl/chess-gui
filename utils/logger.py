class Logger:

    @staticmethod
    def log(message: str):
        print(message)
        with open("gui.log", "a") as f:
            f.write(message + "\n")

    @staticmethod
    def reset_log():
        open("gui.log", "w").close()