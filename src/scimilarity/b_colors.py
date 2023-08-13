class BColors:
    """A class that stores color codes for printing to the console.


    Example:
        Print some text in cyan color::

            from scimilarity import BColors
            print(f"{BColors.OKCYAN}This text is now cyan!{BColors.ENDC}")

    Attributes:
        HEADER (string): character code for a header string.
        OKBLUE (string): character code for a blue string.
        OKCYAN (string): character code for a cyan string.
        OKGREEN (string): character code for a green string.
        WARNING (string): character code for a warning string.
        FAIL (string): character code for a fail red string.
        ENDC (string): character code for the end of the character formating.
        BOLD (string): character code for a bold string.
        UNDERLINE (string): character code for underlining a string.
    """

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
