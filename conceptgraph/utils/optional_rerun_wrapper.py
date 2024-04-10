import logging

class OptionalReRun:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config_use_rerun = None
            cls._instance._rerun = None
        return cls._instance

    def set_use_rerun(self, config_use_rerun):
        self._config_use_rerun = config_use_rerun
        if self._config_use_rerun and self._rerun is None:
            try:
                import rerun as rr
                self._rerun = rr
                logging.info("rerun is installed. Using rerun for logging.")
            except ImportError:
                logging.info("rerun is not installed. Not using rerun for logging.")
        else:
            logging.info("rerun functionality is disabled in the config. Not using rerun for logging.")

    def __getattr__(self, name):
        def method(*args, **kwargs):
            if self._config_use_rerun and self._rerun:
                func = getattr(self._rerun, name, None)
                if func:
                    return func(*args, **kwargs)
                else:
                    logging.debug(f"'{name}' is not a valid rerun method.")
            else:
                if not self._config_use_rerun:
                    logging.debug(f"Skipping optional rerun call to '{name}' because rerun usage is disabled.")
                elif self._rerun is None:
                    logging.debug(f"Skipping optional rerun call to '{name}' because rerun is not installed.")
        return method
