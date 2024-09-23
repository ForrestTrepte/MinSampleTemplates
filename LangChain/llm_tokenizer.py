import logging

logger = logging.getLogger(__name__)
tokenizer_model_names_that_have_already_been_warned = set()


def get_fallback_tokenizer_name(model_name: str, e: Exception) -> str:
    """
    Sometimes experimental models are released before they are supported in the tokenizer libraries.
    When this fails, this method is used to emit a warning and return a default model name that can be used to count tokens.
    Note that the fallback may not produce an accurate token count for this particular model.

    Args:
        model_name (str): The name of the model to look up.

    Returns:
        str: Fallback model name that is supported by a tokenizer.
    """
    fallback_model_name = "gpt-4o"
    if not model_name in tokenizer_model_names_that_have_already_been_warned:
        logger.warning(
            f"Failed to get encoding for model {model_name}, falling back to {fallback_model_name}"
        )
        logger.debug(e)
        tokenizer_model_names_that_have_already_been_warned.add(model_name)

    return fallback_model_name
