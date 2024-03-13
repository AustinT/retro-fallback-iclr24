"""Main script to run search algorithms and analyze results."""

import logging
import sys

from retro_fallback_iclr24.iclr24_experiments import analyze_results, run_search
from retro_fallback_iclr24.retro_fallback import logger as rfb_logger

if __name__ == "__main__":

    # Set up loggers
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        filemode="w",
    )
    rfb_logger.setLevel(logging.DEBUG - 1)  # show every iteration
    analyze_results.logger.setLevel(logging.DEBUG)
    run_search.logger.setLevel(logging.DEBUG)

    # Run search and analyze results
    run_search.run_search_and_analyze_results()
