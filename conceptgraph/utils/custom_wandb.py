import logging

class OptionalWandB:
    '''
    A class for optionally integrating Weights & Biases (wandb) into your projects. It's designed to 
    allow projects to function with or without wandb installed, making the logging functionality 
    flexible and optional. This is particularly useful in environments where wandb is not available 
    or in scenarios where you want to run your code without wandb logging. The class follows the 
    Singleton design pattern to ensure a single, consistent state is maintained throughout the 
    application's lifetime.

    How It Works:
    - On first use, it attempts to configure itself based on the provided settings, specifically 
      whether to use wandb or not.
    - If wandb usage is enabled and the wandb library is installed, it will act as a proxy, forwarding 
      calls to the wandb library.
    - If wandb is not installed or its usage is disabled, calls to wandb methods (like log, init) will 
      be silently ignored, allowing your code to run without modifications.

    Example Usage:
    --------------

    Normally, you would use wandb directly in your project like so:

    ```python
    import wandb
    wandb.init(project="my_project", config=my_config)
    wandb.log({"metric": value})
    wandb.finish()
    ```

    With OptionalWandB, you can replace the wandb usage as follows:

    ```python
    from custom_wandb import OptionalWandB

    # In your main script, instantiate the OptionalWandB singleton and
    # set whether to use wandb based on your project configuration
    optional_wandb = OptionalWandB()
    optional_wandb.set_use_wandb(cfg.use_wandb)

    # The rest of your code can use optional_wandb as if it was the wandb library
    optional_wandb.init(project="my_project", config=my_config)
    optional_wandb.log({"metric": value})
    optional_wandb.finish()
    ```
    
    ```python
    # As with normal wandb, you may want to use the optional_wandb in another script as well
    # for example in a utils.py file, and there you'll want to do the import
    # but also initialize the singleton instance of OptionalWandB at the start of the file
    # here you don't need to set the use_wandb flag, as it will be set in the main script
    from custom_wandb import OptionalWandB
    optional_wandb = OptionalWandB()
    
    # Then you can use the optional_wandb instance in your functions
    # As you normally would with wandb
    def do_something():
        value = 42
        optional_wandb.log({"metric": value})
    ```

    In the above example, if `cfg.use_wandb` is True and the wandb library is installed, 
    `optional_wandb` will forward calls to the wandb library. If wandb is not installed or 
    `cfg.use_wandb` is False, these method calls will do nothing but can still be included 
    in your code without causing errors.

    Attributes:
        _instance: Stores the singleton instance of OptionalWandB.
        _config_use_wandb (bool): Determines if wandb is to be used, based on user configuration.
        _wandb (module): Reference to the wandb module if it's installed and enabled.
    '''
    
    _instance = None
    
    def __new__(cls):
        '''
        Ensures that only one instance of the OptionalWandB class is created. 
        This method is called before an object is instantiated.

        Parameters:
            use_wandb (bool, optional): Indicates whether to use wandb. This parameter is not used in the current implementation but is kept for compatibility.

        Returns:
            The singleton instance of the OptionalWandB class.
        '''
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config_use_wandb = None
            cls._instance._wandb = None
            # cls._instance._initialized = False
        return cls._instance

    def set_use_wandb(self, config_use_wandb):
        '''
        Configures the OptionalWandB instance to use or not use wandb based on the provided configuration. 
        Attempts to import the wandb module if enabled and logs the outcome.

        Parameters:
            config_use_wandb (bool): True to enable wandb functionality, False to disable it.
        '''
        self._config_use_wandb = config_use_wandb
        if self._config_use_wandb and self._wandb is None:
            try:
                import wandb
                self._wandb = wandb
                logging.info("wandb is installed. Using wandb for logging.")
            except ImportError:
                logging.info("wandb is not installed. Not using wandb for logging.")
        else:
            logging.info("wandb functionality is disabled in the config. Not using wandb for logging.")

    def __getattr__(self, name):
        '''
        Provides a way to dynamically call wandb methods if wandb is configured for use and installed. 
        If the conditions are not met, logs a message instead of performing the operation. This method 
        is automatically called when an attempt is made to access an attribute that doesn't exist in the 
        OptionalWandB instance.

        Parameters:
            name (str): The name of the method being accessed.

        Returns:
            A method that either calls the corresponding wandb method or logs a message, depending on 
            the wandb usage configuration and installation status.
        '''
        def method(*args, **kwargs):
            if self._config_use_wandb and self._wandb:
                # if self._initialized:
                func = getattr(self._wandb, name, None)
                if func:
                    return func(*args, **kwargs)
                else:
                    logging.debug(f"'{name}' is not a valid wandb method.")
                # else:
                #     logging.info(f"Skipping optional wandb call to '{name}' because wandb is not initialized.")
            else:
                if not self._config_use_wandb:
                    logging.debug(f"Skipping optional wandb call to '{name}' because wandb usage is disabled.")
                elif self._wandb is None:
                    logging.debug(f"Skipping optional wandb call to '{name}' because wandb is not installed.")
        return method