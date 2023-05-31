import logging
import logger2

# Define a custom logging level
ACTION = logging.INFO + 1
logging.addLevelName(ACTION, "ACTION")

# Create a custom log format with a newline character
ACTION_log_format = '%(levelname)s - %(name)s: %(message)s\n'
logging.basicConfig(filename='logfile.log',level=logging.NOTSET, format=ACTION_log_format)

# Log a message at the custom level
logger = logging.getLogger(__name__)
logger.info("Prueba de mensaje")

logger2.func()
logger2.perform_action()
# Output:
# ACTION - __main__: This is a custom log message.
# <empty line>
