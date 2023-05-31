
import logging
from tqdm import tqdm as log_progress



logger = logging.getLogger(__name__)
print(logger)

def func():
    logger.info( "This is a ksjdbcjsdhcbcustom log message.")



def perform_action():
    logger.info("Performing an action in mymodule.")

    # Instantiate tqdm with the logger as the file argument
    N = 10
    pbar = log_progress(range(N))
    pbar.set_description("Loss: %s " % 100)
    for i in pbar:
        print(i)
        pbar.set_description("Loss: {:6.4e}".format(N/(i+1)))
    logger.info(pbar.n)
